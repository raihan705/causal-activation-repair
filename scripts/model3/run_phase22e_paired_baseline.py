#!/usr/bin/env python
"""Generate the prospectively frozen paired B0 controls for Model3 Stage 22E."""

from __future__ import annotations

import argparse
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
PROTOCOL = OUT / "phase22e_causal_validation_protocol.json"
OUTPUT = OUT / "phase22e_paired_baseline_generations.json"
CHECKPOINT = OUT / "phase22e_paired_baseline_checkpoint.json"
RUN_MANIFEST = OUT / "phase22e_paired_baseline_run_manifest.json"
FINAL_MANIFEST = OUT / "phase22e_paired_baseline_generation_manifest.json"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
EXPECTED_COUNT = 40
RUNNER = Path(__file__).resolve()


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def model_snapshot_path() -> Path:
    path = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots" / MODEL_REVISION
    require(path.is_dir() and (path / "config.json").is_file(), f"pinned model snapshot missing: {path}")
    return path


def validity(text: str) -> dict[str, Any]:
    stripped = text.strip()
    markers = ("def ", "int ", "char ", "return", "if ", "for ", "{", "(")
    marker = next((x for x in markers if x in text), None)
    valid = len(stripped) >= 10 and marker is not None
    return {"is_valid": valid, "invalid_reason": None if valid else ("EMPTY_OR_TOO_SHORT" if len(stripped) < 10 else "NO_CODE_MARKER"), "stripped_character_count": len(stripped), "matched_marker": marker}


def load_inputs(resume: bool) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    require(PROTOCOL.is_file(), "frozen Stage 22E protocol missing")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    require(protocol["status"] == "FROZEN_BEFORE_PAIRED_BASELINE_GENERATION", "causal protocol not frozen")
    require(protocol["implementation"]["paired_baseline_runner_sha256"] == sha256_file(RUNNER), "baseline runner drift")
    plan = protocol["paired_baseline_prompts"]
    require(len(plan) == EXPECTED_COUNT and len({int(x["prompt_id"]) for x in plan}) == EXPECTED_COUNT, "paired baseline plan count/uniqueness mismatch")
    records: list[dict[str, Any]] = []
    if CHECKPOINT.exists():
        require(resume, "checkpoint exists; use --resume")
        checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
        require(checkpoint["protocol_sha256"] == sha256_file(PROTOCOL), "checkpoint protocol drift")
        require(checkpoint["runner_sha256"] == sha256_file(RUNNER), "checkpoint runner drift")
        records = checkpoint["records"]
        require([x["record_id"] for x in records] == [x["record_id"] for x in plan[:len(records)]], "checkpoint not exact plan prefix")
    else:
        require(not resume, "--resume supplied without checkpoint")
    return protocol, plan, records


def preflight(resume: bool) -> dict[str, Any]:
    protocol, plan, records = load_inputs(resume)
    return {
        "schema_version": "phase22e_paired_baseline_preflight_v1", "status": "PASS_APPROVED",
        "expected_records": len(plan), "completed_checkpoint_records": len(records),
        "resume_required": bool(records), "protocol_sha256": sha256_file(PROTOCOL),
        "runner_sha256": sha256_file(RUNNER), "model_loaded": False,
        "generation_run": False, "sae_loaded": False, "scanner_run": False,
        "steering_run": False, "heldout_used": False,
    }


def run(resume: bool) -> None:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    require(torch.cuda.is_available(), "CUDA unavailable")
    protocol, plan, records = load_inputs(resume)
    snapshot = model_snapshot_path()
    os.environ["HF_HUB_OFFLINE"] = "1"
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map={"": "cuda:0"}, local_files_only=True,
    ).eval()
    devices = sorted({str(x.device) for x in model.parameters()})
    require(devices == ["cuda:0"], f"unauthorized model offload: {devices}")
    prior_elapsed = float(json.loads(CHECKPOINT.read_text(encoding="utf-8")).get("elapsed_seconds", 0.0)) if CHECKPOINT.exists() else 0.0
    started = time.perf_counter()
    run_state = {
        "schema_version": "phase22e_paired_baseline_run_v1", "status": "IN_PROGRESS",
        "expected_records": EXPECTED_COUNT, "completed_records": len(records),
        "protocol_sha256": sha256_file(PROTOCOL), "runner_sha256": sha256_file(RUNNER),
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "gpu": torch.cuda.get_device_name(0),
        "parameter_devices": devices, "torch_version": torch.__version__,
        "transformers_version": transformers.__version__, "platform": platform.platform(),
        "sae_loaded": False, "steering_run": False, "scanner_run": False, "heldout_used": False,
    }
    atomic_json(RUN_MANIFEST, run_state)
    for index in range(len(records), len(plan)):
        item = plan[index]
        rendered = tokenizer.apply_chat_template([{"role": "user", "content": item["source_prompt"]}], tokenize=False, add_generation_prompt=True)
        require(sha256_text(rendered) == item["rendered_prompt_sha256"], f"rendered prompt drift: {item['prompt_id']}")
        inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to("cuda:0")
        require(int(inputs["input_ids"].shape[-1]) == item["input_token_count"], f"input token drift: {item['prompt_id']}")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        generated_text = ""
        generated_count = 0
        status = "FAILED"
        exception = None
        try:
            with torch.inference_mode():
                output = model.generate(
                    **inputs, max_new_tokens=512, temperature=0.2, top_p=0.95,
                    do_sample=True, pad_token_id=tokenizer.pad_token_id,
                )
            suffix = output[0, inputs["input_ids"].shape[-1]:]
            generated_count = int(suffix.shape[0])
            generated_text = tokenizer.decode(suffix, skip_special_tokens=True)
            status = "SUCCESS"
        except Exception as exc:
            exception = {"type": type(exc).__name__, "message": str(exc)}
            torch.cuda.empty_cache()
        records.append({
            "schema_version": "phase22e_paired_baseline_record_v1", "record_index": index,
            "record_id": item["record_id"], "target_cwe": item["target_cwe"],
            "prompt_id": item["prompt_id"], "population": item["population"],
            "source_index": item["source_index"], "language": item["language"],
            "source_prompt_sha256": item["source_prompt_sha256"],
            "rendered_prompt_sha256": item["rendered_prompt_sha256"],
            "input_token_count": item["input_token_count"], "method": "PAIRED_B0",
            "seed": 42, "seed_reset_immediately_before_generation": True,
            "generation_settings": protocol["generation"], "generated_text": generated_text,
            "generated_code": generated_text, "generated_text_sha256": sha256_text(generated_text),
            "generated_token_count": generated_count, "generation_status": status,
            "validity": validity(generated_text), "exception": exception,
            "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
            "sae_intervention_applied": False, "causal_steering_applied": False,
            "protocol_sha256": sha256_file(PROTOCOL),
        })
        if len(records) % 5 == 0 or len(records) == EXPECTED_COUNT:
            elapsed = prior_elapsed + time.perf_counter() - started
            atomic_json(CHECKPOINT, {
                "schema_version": "phase22e_paired_baseline_checkpoint_v1",
                "protocol_sha256": sha256_file(PROTOCOL), "runner_sha256": sha256_file(RUNNER),
                "completed_records": len(records), "records": records, "elapsed_seconds": elapsed,
            })
            run_state.update({"completed_records": len(records), "elapsed_seconds": elapsed, "failure_count": sum(x["generation_status"] != "SUCCESS" for x in records)})
            atomic_json(RUN_MANIFEST, run_state)
            print(f"PHASE22E PAIRED B0 {len(records)}/{EXPECTED_COUNT} failures={run_state['failure_count']} elapsed_min={elapsed/60:.1f}", flush=True)
    atomic_json(OUTPUT, {"schema_version": "phase22e_paired_baseline_generations_v1", "status": "COMPLETE", "protocol_sha256": sha256_file(PROTOCOL), "records": records})
    output_hash = sha256_file(OUTPUT)
    final = {
        "schema_version": "phase22e_paired_baseline_generation_manifest_v1", "status": "COMPLETE",
        "protocol_sha256": sha256_file(PROTOCOL), "runner_sha256": sha256_file(RUNNER),
        "record_count": len(records), "success_count": sum(x["generation_status"] == "SUCCESS" for x in records),
        "failure_count": sum(x["generation_status"] != "SUCCESS" for x in records),
        "invalid_count": sum(not x["validity"]["is_valid"] for x in records),
        "output_path": "revision/model3/phase22/outputs/phase22e_paired_baseline_generations.json",
        "output_sha256": output_hash, "record_ids": [x["record_id"] for x in records],
        "prompt_ids": [x["prompt_id"] for x in records], "sae_loaded": False,
        "steering_run": False, "scanner_run": False, "heldout_used": False,
    }
    atomic_json(FINAL_MANIFEST, final)
    run_state.update({"status": "COMPLETE", "completed_records": EXPECTED_COUNT, "output_sha256": output_hash, "generation_manifest_sha256": sha256_file(FINAL_MANIFEST)})
    atomic_json(RUN_MANIFEST, run_state)
    print(json.dumps(final, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(preflight(args.resume), indent=2))
    if not args.preflight_only:
        require(not OUTPUT.exists(), "immutable paired baseline output already exists")
        run(args.resume)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"PHASE22E_BASELINE_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
