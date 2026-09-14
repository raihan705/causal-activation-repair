#!/usr/bin/env python
"""Resumable unsteered Model3 B0 generation on the frozen 22C population."""

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
PHASE = ROOT / "revision/model3/phase22"
OUT = PHASE / "outputs"
DEV = ROOT / "data/cyberseceval/dev_prompts.json"
POPULATION = OUT / "phase22c_source_population.json"
PROTOCOL = OUT / "phase22c_b0_protocol.json"
OUTPUT = OUT / "phase22c_b0_dev_outputs.json"
CHECKPOINT = OUT / "phase22c_b0_checkpoint.json"
RUN_MANIFEST = OUT / "phase22c_b0_run_manifest.json"
FINAL_MANIFEST = OUT / "phase22c_b0_generation_manifest.json"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
EXPECTED_COUNT = 180
RUNNER = Path(__file__).resolve()


def model_snapshot_path() -> Path:
    path = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots" / MODEL_REVISION
    require(path.is_dir() and (path / "config.json").is_file(), f"pinned local model snapshot is incomplete: {path}")
    return path


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


def load_inputs(resume: bool) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    require(POPULATION.is_file() and PROTOCOL.is_file(), "frozen 22C population/protocol missing")
    population = json.loads(POPULATION.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    require(population.get("status") == "FROZEN_BEFORE_GENERATION", "population is not frozen")
    require(protocol.get("status") == "FROZEN_BEFORE_GENERATION", "protocol is not frozen")
    require(protocol["runner"]["sha256"] == sha256_file(RUNNER), "runner changed after protocol freeze")
    require(protocol["source_population"]["sha256"] == sha256_file(POPULATION), "population hash mismatch")
    require(protocol["source"]["sha256"] == sha256_file(DEV), "development source hash mismatch")
    require(protocol["source_population"]["record_count"] == EXPECTED_COUNT, "population count mismatch")
    require(protocol["model"] == {"id": MODEL_ID, "revision": MODEL_REVISION, "dtype": "bfloat16", "device": "cuda:0", "quantization": False, "offload": False}, "model protocol mismatch")
    require(protocol["generation"] == {"seed": 42, "temperature": 0.2, "top_p": 0.95, "max_new_tokens": 512, "do_sample": True, "decode_generated_tokens_only": True, "skip_special_tokens": True, "batch_size": 1}, "generation protocol mismatch")
    records_source = population["records"]
    require(len(records_source) == EXPECTED_COUNT, "frozen population record count mismatch")
    ids = [int(x["prompt_id"]) for x in records_source]
    require(len(set(ids)) == EXPECTED_COUNT, "duplicate population prompt ID")
    prior = None
    records: list[dict[str, Any]] = []
    if CHECKPOINT.exists():
        require(resume, "checkpoint exists; rerun with --resume")
        prior = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
        require(prior.get("schema_version") == "phase22c_model3_b0_checkpoint_v1", "checkpoint schema mismatch")
        require(prior.get("condition", {}).get("protocol_sha256") == sha256_file(PROTOCOL), "checkpoint protocol hash mismatch")
        require(prior.get("condition", {}).get("runner_sha256") == sha256_file(RUNNER), "checkpoint runner hash mismatch")
        records = prior.get("records", [])
        require([int(x["prompt_id"]) for x in records] == ids[:len(records)], "checkpoint is not a source-order prefix")
        require(len({int(x["prompt_id"]) for x in records}) == len(records), "duplicate checkpoint prompt ID")
        require(isinstance(prior.get("rng_state"), dict), "checkpoint RNG state missing")
    else:
        require(not resume, "--resume supplied without checkpoint")
    return records_source, protocol, records, prior


def preflight(resume: bool) -> dict[str, Any]:
    source, protocol, records, prior = load_inputs(resume)
    return {
        "schema_version": "phase22c_model3_b0_preflight_v1",
        "status": "PASS_APPROVED",
        "model_loaded": False,
        "generation_run": False,
        "scanner_run": False,
        "sae_loaded": False,
        "heldout_accessed": False,
        "source_count": len(source),
        "completed_checkpoint_records": len(records),
        "resume_required": prior is not None,
        "protocol_path": str(PROTOCOL.relative_to(ROOT)).replace("\\", "/"),
        "protocol_sha256": sha256_file(PROTOCOL),
        "runner_sha256": sha256_file(RUNNER),
        "output_path": str(OUTPUT.relative_to(ROOT)).replace("\\", "/"),
        "condition": {"method": "B0", "seed": 42, "chat_template": True, "settings": protocol["generation"]},
    }


def generate(resume: bool) -> None:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    require(torch.cuda.is_available(), "CUDA is unavailable")
    source, protocol, records, prior = load_inputs(resume)
    ids = [int(x["prompt_id"]) for x in source]
    condition = {
        "method": "B0", "seed": 42, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "protocol_sha256": sha256_file(PROTOCOL), "runner_sha256": sha256_file(RUNNER),
        "source_population_sha256": sha256_file(POPULATION), "source_sha256": sha256_file(DEV),
    }
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    os.environ["HF_HUB_OFFLINE"] = "1"
    snapshot = model_snapshot_path()
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    require(tokenizer.pad_token_id is not None, "pinned tokenizer has no pad token")
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, device_map={"": "cuda:0"}, local_files_only=True,
    ).eval()
    parameter_devices = sorted({str(x.device) for x in model.parameters()})
    require(parameter_devices == ["cuda:0"], f"unauthorized offload/device map: {parameter_devices}")
    require(next(model.parameters()).dtype == torch.bfloat16, "model dtype mismatch")
    if prior is not None:
        restore_rng(torch, prior["rng_state"])
    start = len(records)
    prior_elapsed = float(prior.get("elapsed_seconds", 0.0)) if prior else 0.0
    started = time.perf_counter()
    run_manifest = {
        "schema_version": "phase22c_model3_b0_run_manifest_v1", "status": "IN_PROGRESS",
        "condition": condition, "completed_records": start, "source_count": EXPECTED_COUNT,
        "model_load_seconds": time.perf_counter() - load_started,
        "transformers_version": transformers.__version__, "torch_version": torch.__version__,
        "platform": platform.platform(), "gpu": torch.cuda.get_device_name(0),
        "parameter_devices": parameter_devices,
        "failure_count": sum(x["generation_status"] == "FAILED" for x in records),
    }
    atomic_json(RUN_MANIFEST, run_manifest)

    frozen = protocol["records"]
    for population_index in range(start, len(source)):
        row = source[population_index]
        fixed = frozen[population_index]
        require(int(row["prompt_id"]) == int(fixed["prompt_id"]), "protocol/population prompt ID mismatch")
        prompt = row["test_case_prompt"]
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
        )
        require(hashlib.sha256(rendered.encode()).hexdigest() == fixed["rendered_prompt_sha256"], "rendered prompt drift")
        generated_text = ""
        generated_token_count = 0
        status = "FAILED"
        failure = None
        input_token_count = 0
        try:
            inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to("cuda:0")
            input_token_count = int(inputs["input_ids"].shape[-1])
            require(input_token_count == int(fixed["rendered_input_token_count"]), "tokenization drift")
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
            "schema_version": "phase22c_model3_b0_record_v1",
            "prompt_id": int(row["prompt_id"]), "source_index": int(row["source_index"]),
            "population_index": population_index, "source_cwe": row["cwe_identifier"],
            "cwe_id": row["cwe_identifier"], "language": row["language"],
            "prompt_text": prompt, "source_prompt_sha256": row["source_prompt_sha256"],
            "chat_template_used": True, "rendered_prompt_sha256": fixed["rendered_prompt_sha256"],
            "input_token_count": input_token_count, "generated_text": generated_text,
            "generated_code": generated_text, "generated_token_count": generated_token_count,
            "generation_status": status, "failure_status": failure,
            "empty_status": "EMPTY_OR_WHITESPACE" if not generated_text.strip() else "NONEMPTY",
            "method": "B0", "seed": 42, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
            "tokenizer_id": MODEL_ID, "tokenizer_revision": MODEL_REVISION,
            "generation_settings": protocol["generation"], "sae_intervention_applied": False,
            "causal_steering_applied": False,
        })
        elapsed = prior_elapsed + time.perf_counter() - started
        if len(records) % 10 == 0 or len(records) == len(source):
            checkpoint = {
                "schema_version": "phase22c_model3_b0_checkpoint_v1", "condition": condition,
                "records": records, "rng_state": capture_rng(torch), "elapsed_seconds": elapsed,
            }
            atomic_json(CHECKPOINT, checkpoint)
            run_manifest.update({
                "completed_records": len(records), "elapsed_seconds": elapsed,
                "failure_count": sum(x["generation_status"] == "FAILED" for x in records),
            })
            atomic_json(RUN_MANIFEST, run_manifest)
            print(f"Model3 B0 {len(records)}/{EXPECTED_COUNT} failures={run_manifest['failure_count']} elapsed_min={elapsed/60:.1f}", flush=True)

    require([int(x["prompt_id"]) for x in records] == ids, "final output source order mismatch")
    atomic_json(OUTPUT, records)
    output_hash = sha256_file(OUTPUT)
    final_manifest = {
        "schema_version": "phase22c_model3_b0_generation_manifest_v1",
        "status": "COMPLETE_WITH_PRESERVED_FAILURES" if any(x["generation_status"] == "FAILED" for x in records) else "COMPLETE",
        "condition": condition, "source_record_count": EXPECTED_COUNT, "output_record_count": len(records),
        "unique_prompt_id_count": len(set(ids)), "source_order_preserved": [int(x["prompt_id"]) for x in records] == ids,
        "heldout_ids_used": False, "duplicate_ids": False,
        "success_count": sum(x["generation_status"] == "SUCCESS" for x in records),
        "failure_count": sum(x["generation_status"] == "FAILED" for x in records),
        "empty_or_whitespace_count": sum(x["empty_status"] == "EMPTY_OR_WHITESPACE" for x in records),
        "generation_output": {"path": "revision/model3/phase22/outputs/phase22c_b0_dev_outputs.json", "sha256": output_hash},
        "source_population": {"path": "revision/model3/phase22/outputs/phase22c_source_population.json", "sha256": sha256_file(POPULATION)},
        "protocol": {"path": "revision/model3/phase22/outputs/phase22c_b0_protocol.json", "sha256": sha256_file(PROTOCOL)},
        "runner": {"path": "revision/model3/phase22/scripts/run_phase22c_b0.py", "sha256": sha256_file(RUNNER)},
        "rendered_prompt_aggregate_sha256": protocol["prompt_interface"]["rendered_prompt_aggregate_sha256"],
        "completed_prompt_ids_source_order": ids, "sae_intervention_applied": False,
        "causal_steering_applied": False, "scanner_run": False, "heldout_accessed": False,
    }
    atomic_json(FINAL_MANIFEST, final_manifest)
    run_manifest.update({"status": "COMPLETE", "completed_records": EXPECTED_COUNT, "output_sha256": output_hash, "generation_manifest_sha256": sha256_file(FINAL_MANIFEST)})
    atomic_json(RUN_MANIFEST, run_manifest)
    print(json.dumps({"status": final_manifest["status"], "output_sha256": output_hash, "manifest_sha256": sha256_file(FINAL_MANIFEST), "success": final_manifest["success_count"], "failures": final_manifest["failure_count"]}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(preflight(args.resume), indent=2))
    if args.preflight_only:
        return 0
    require(not OUTPUT.exists(), "immutable final output already exists")
    generate(args.resume)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"PHASE22C_MODEL3_B0_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
