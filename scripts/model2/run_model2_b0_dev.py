#!/usr/bin/env python
"""Resumable unsteered Model2 B0 generation on the 1341 development prompts."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
PHASE = ROOT / "revision/model2/phase21"
OUT = PHASE / "outputs"
DEV = ROOT / "data/cyberseceval/dev_prompts.json"
PROTOCOL = OUT / "model2_b0_dev_protocol.json"
OUTPUT = OUT / "model2_b0_dev_outputs.json"
CHECKPOINT = OUT / "model2_b0_dev_checkpoint.json"
RUN_MANIFEST = OUT / "model2_b0_dev_run_manifest.json"
FINAL_MANIFEST = OUT / "model2_b0_dev_generation_manifest.json"
MODEL_ID = "google/gemma-2-9b-it"
MODEL_REVISION = "11c9b309abf73637e4b6f9a3fa1e92e615547819"
RUNNER = Path(__file__).resolve()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def capture_rng(torch: Any) -> dict[str, Any]:
    return {
        "cpu": base64.b64encode(bytes(torch.get_rng_state().tolist())).decode(),
        "cuda": [base64.b64encode(bytes(x.tolist())).decode() for x in torch.cuda.get_rng_state_all()],
    }


def restore_rng(torch: Any, state: dict[str, Any]) -> None:
    torch.set_rng_state(torch.tensor(list(base64.b64decode(state["cpu"])), dtype=torch.uint8))
    cuda = [torch.tensor(list(base64.b64decode(x)), dtype=torch.uint8) for x in state["cuda"]]
    require(len(cuda) == torch.cuda.device_count(), "checkpoint CUDA RNG device count mismatch")
    torch.cuda.set_rng_state_all(cuda)


def validate_inputs(resume: bool) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    require(PROTOCOL.is_file(), "frozen B0 protocol is missing")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    require(protocol.get("status") == "FROZEN_BEFORE_GENERATION", "protocol is not frozen")
    require(protocol["runner"]["sha256"] == sha256_file(RUNNER), "runner changed after protocol freeze")
    require(protocol["source"]["sha256"] == sha256_file(DEV), "development source hash mismatch")
    require(protocol["source"]["record_count"] == 1341, "protocol development count mismatch")
    require(protocol["model"] == {"id": MODEL_ID, "revision": MODEL_REVISION, "dtype": "bfloat16", "device": "cuda:0", "quantization": False, "offload": False}, "model protocol mismatch")
    require(protocol["generation"] == {"seed": 42, "temperature": 0.2, "top_p": 0.95, "max_new_tokens": 512, "do_sample": True, "decode_generated_tokens_only": True, "skip_special_tokens": True}, "generation protocol mismatch")
    rows = json.loads(DEV.read_text(encoding="utf-8"))
    require(len(rows) == 1341, "development source count mismatch")
    ids = [int(x["prompt_id"]) for x in rows]
    require(len(set(ids)) == 1341, "duplicate development prompt ID")
    records: list[dict[str, Any]] = []
    prior = None
    if CHECKPOINT.exists():
        require(resume, "checkpoint exists; rerun with --resume")
        prior = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
        require(prior.get("schema_version") == "phase21_model2_b0_checkpoint_v1", "checkpoint schema mismatch")
        require(prior.get("condition", {}).get("protocol_sha256") == sha256_file(PROTOCOL), "checkpoint protocol hash mismatch")
        require(prior.get("condition", {}).get("runner_sha256") == sha256_file(RUNNER), "checkpoint runner hash mismatch")
        records = prior.get("records", [])
        require([int(x["prompt_id"]) for x in records] == ids[:len(records)], "checkpoint is not a source-order prefix")
        require(len({int(x["prompt_id"]) for x in records}) == len(records), "duplicate checkpoint prompt ID")
        require(isinstance(prior.get("rng_state"), dict), "checkpoint RNG state missing")
    else:
        require(not resume, "--resume supplied without checkpoint")
    return rows, protocol, records, prior


def preflight(resume: bool) -> dict[str, Any]:
    rows, protocol, records, prior = validate_inputs(resume)
    return {
        "schema_version": "phase21_model2_b0_preflight_v1",
        "status": "PASS_APPROVED",
        "model_loaded": False,
        "generation_run": False,
        "scanner_run": False,
        "sae_loaded": False,
        "causal_intervention_run": False,
        "heldout_accessed": False,
        "source_count": len(rows),
        "completed_checkpoint_records": len(records),
        "resume_required": prior is not None,
        "protocol_path": str(PROTOCOL.relative_to(ROOT)).replace("\\", "/"),
        "protocol_sha256": sha256_file(PROTOCOL),
        "runner_sha256": sha256_file(RUNNER),
        "output_path": str(OUTPUT.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_path": str(CHECKPOINT.relative_to(ROOT)).replace("\\", "/"),
        "condition": {"method": "B0", "seed": 42, "chat_template": True, "settings": protocol["generation"]},
    }


def generate(resume: bool) -> None:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    require(torch.cuda.is_available(), "CUDA is unavailable")
    rows, protocol, records, prior = validate_inputs(resume)
    ids = [int(x["prompt_id"]) for x in rows]
    condition = {
        "method": "B0", "seed": 42, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "protocol_sha256": sha256_file(PROTOCOL), "runner_sha256": sha256_file(RUNNER),
        "source_sha256": sha256_file(DEV),
    }
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    os.environ["HF_HUB_OFFLINE"] = "1"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, device_map={"": "cuda:0"}, local_files_only=True,
    ).eval()
    parameter_devices = sorted({str(x.device) for x in model.parameters()})
    require(parameter_devices == ["cuda:0"], f"unauthorized offload/device map: {parameter_devices}")
    if prior is not None:
        restore_rng(torch, prior["rng_state"])
    start = len(records)
    prior_elapsed = float(prior.get("elapsed_seconds", 0.0)) if prior else 0.0
    started = time.perf_counter()
    run_manifest = {
        "schema_version": "phase21_model2_b0_run_manifest_v1", "status": "IN_PROGRESS",
        "condition": condition, "completed_records": start, "source_count": 1341,
        "model_load_seconds": time.perf_counter() - load_started,
        "transformers_version": transformers.__version__, "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0), "parameter_devices": parameter_devices,
        "failure_count": sum(x["generation_status"] == "FAILED" for x in records),
    }
    atomic_json(RUN_MANIFEST, run_manifest)

    frozen_by_index = protocol["records"]
    for source_index in range(start, len(rows)):
        row = rows[source_index]
        frozen = frozen_by_index[source_index]
        require(int(row["prompt_id"]) == int(frozen["prompt_id"]), "protocol/source prompt ID mismatch")
        source_prompt = row["test_case_prompt"]
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": source_prompt}], tokenize=False,
            add_generation_prompt=True,
        )
        require(hashlib.sha256(rendered.encode()).hexdigest() == frozen["rendered_prompt_sha256"], "rendered prompt drift")
        generated_text = ""
        generated_token_count = 0
        status = "FAILED"
        failure = None
        input_token_count = 0
        try:
            inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to("cuda:0")
            input_token_count = int(inputs["input_ids"].shape[-1])
            require(input_token_count == int(frozen["rendered_input_token_count"]), "tokenization drift")
            with torch.inference_mode():
                output = model.generate(
                    **inputs, max_new_tokens=512, temperature=0.2, top_p=0.95,
                    do_sample=True, pad_token_id=tokenizer.pad_token_id,
                )
            generated = output[0, input_token_count:]
            generated_token_count = int(generated.shape[0])
            generated_text = tokenizer.decode(generated, skip_special_tokens=True)
            status = "SUCCESS"
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            torch.cuda.empty_cache()
        records.append({
            "schema_version": "phase21_model2_b0_record_v1",
            "prompt_id": int(row["prompt_id"]), "source_index": source_index,
            "source_cwe": row.get("cwe_identifier", ""), "cwe_id": row.get("cwe_identifier", ""),
            "language": row.get("language", ""), "prompt_text": source_prompt,
            "source_prompt_sha256": frozen["source_prompt_sha256"],
            "chat_template_used": True, "rendered_prompt_sha256": frozen["rendered_prompt_sha256"],
            "input_token_count": input_token_count,
            "generated_text": generated_text, "generated_code": generated_text,
            "generated_token_count": generated_token_count,
            "generation_status": status, "failure_status": failure,
            "empty_status": "EMPTY_OR_WHITESPACE" if not generated_text.strip() else "NONEMPTY",
            "method": "B0", "seed": 42,
            "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
            "tokenizer_id": MODEL_ID, "tokenizer_revision": MODEL_REVISION,
            "generation_settings": protocol["generation"],
            "sae_intervention_applied": False, "causal_steering_applied": False,
        })
        elapsed = prior_elapsed + time.perf_counter() - started
        should_checkpoint = len(records) % 10 == 0 or len(records) == len(rows)
        if should_checkpoint:
            checkpoint = {
                "schema_version": "phase21_model2_b0_checkpoint_v1", "condition": condition,
                "records": records, "rng_state": capture_rng(torch), "elapsed_seconds": elapsed,
            }
            atomic_json(CHECKPOINT, checkpoint)
            run_manifest["completed_records"] = len(records)
            run_manifest["failure_count"] = sum(x["generation_status"] == "FAILED" for x in records)
            run_manifest["elapsed_seconds"] = elapsed
            atomic_json(RUN_MANIFEST, run_manifest)
            print(f"B0 {len(records)}/1341 failures={run_manifest['failure_count']} elapsed_min={elapsed/60:.1f}", flush=True)

    require([int(x["prompt_id"]) for x in records] == ids, "final output source order mismatch")
    require(len(set(ids)) == 1341, "final duplicate IDs")
    atomic_json(OUTPUT, records)
    output_hash = sha256_file(OUTPUT)
    final_manifest = {
        "schema_version": "phase21_model2_b0_generation_manifest_v1",
        "status": "COMPLETE_WITH_PRESERVED_FAILURES" if any(x["generation_status"] == "FAILED" for x in records) else "COMPLETE",
        "condition": condition, "source_record_count": 1341, "output_record_count": len(records),
        "unique_prompt_id_count": len({int(x["prompt_id"]) for x in records}),
        "source_order_preserved": [int(x["prompt_id"]) for x in records] == ids,
        "heldout_ids_used": False, "duplicate_ids": False,
        "success_count": sum(x["generation_status"] == "SUCCESS" for x in records),
        "failure_count": sum(x["generation_status"] == "FAILED" for x in records),
        "empty_or_whitespace_count": sum(x["empty_status"] == "EMPTY_OR_WHITESPACE" for x in records),
        "generation_output": {"path": "revision/model2/phase21/outputs/model2_b0_dev_outputs.json", "sha256": output_hash},
        "protocol": {"path": "revision/model2/phase21/outputs/model2_b0_dev_protocol.json", "sha256": sha256_file(PROTOCOL)},
        "runner": {"path": "revision/model2/phase21/scripts/run_model2_b0_dev.py", "sha256": sha256_file(RUNNER)},
        "rendered_prompt_aggregate_sha256": protocol["prompt_interface"]["rendered_prompt_aggregate_sha256"],
        "completed_prompt_ids_source_order": ids,
        "sae_intervention_applied": False, "causal_steering_applied": False,
        "scanner_run": False, "heldout_accessed": False,
    }
    atomic_json(FINAL_MANIFEST, final_manifest)
    run_manifest.update({"status": "COMPLETE", "completed_records": 1341, "output_sha256": output_hash, "generation_manifest_sha256": sha256_file(FINAL_MANIFEST)})
    atomic_json(RUN_MANIFEST, run_manifest)
    print(json.dumps({"status": final_manifest["status"], "output_sha256": output_hash, "manifest_sha256": sha256_file(FINAL_MANIFEST), "success": final_manifest["success_count"], "failures": final_manifest["failure_count"]}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    pf = preflight(args.resume)
    print(json.dumps(pf, indent=2))
    if args.preflight_only:
        return 0
    require(not OUTPUT.exists(), "immutable final output already exists")
    generate(args.resume)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"PHASE21_MODEL2_B0_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
