#!/usr/bin/env python
"""Run new-strength generations for frozen Model3 Stage 22F calibration."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import run_phase22e_causal_steering as shared

ROOT = Path(__file__).resolve().parents[4]
PHASE = ROOT / "revision/model3/phase22"
OUT = PHASE / "outputs"
PROTOCOL = OUT / "phase22f_strength_calibration_protocol.json"
CAUSAL_PROTOCOL = OUT / "phase22e_causal_validation_protocol.json"
EXECUTION = OUT / "phase22f_strength_execution_protocol.json"
OUTPUT = OUT / "phase22f_strength_generations.json"
CHECKPOINT = OUT / "phase22f_strength_generation_checkpoint.json"
RUN_MANIFEST = OUT / "phase22f_strength_generation_run_manifest.json"
FINAL_MANIFEST = OUT / "phase22f_strength_generation_manifest.json"
RUNNER = Path(__file__).resolve()
SHARED = Path(shared.__file__).resolve()
EXPECTED_PROTOCOL_SHA256 = "20e57079a57a65fe4569132c09b05b649a98627c85fe9c6effd8b3c20aac279c"
EXPECTED_CAUSAL_PROTOCOL_SHA256 = "e0fcf227460e7408471caabb1d695fa1905bc51073a1581776aa1772b13932f4"
EXPECTED_COUNT = 1083
TARGET_ORDER = {"CWE-120": 0, "CWE-327": 1, "CWE-89": 2}


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    return shared.sha256_file(path)


def sha256_text(value: str) -> str:
    return shared.sha256_text(value)


def canonical_sha256(value: Any) -> str:
    return shared.canonical_sha256(value)


def atomic_json(path: Path, value: Any) -> None:
    shared.atomic_json(path, value)


def build_worklist(protocol: dict[str, Any], causal_protocol: dict[str, Any]) -> list[dict[str, Any]]:
    prompt_by_key = {(x["target_cwe"], int(x["prompt_id"])): x for x in causal_protocol["paired_baseline_prompts"]}
    routes = sorted(protocol["route_specs"], key=lambda x: (int(x["layer"]), TARGET_ORDER[x["target_cwe"]], int(x["statistical_rank"]), int(x["feature_id"])))
    worklist: list[dict[str, Any]] = []
    for route in routes:
        new_strengths = sorted((x for x in route["strength_conditions"] if x["source"] == "NEW_STAGE22F_GENERATION"), key=lambda x: float(x["multiplier"]))
        require([x["multiplier"] for x in new_strengths] == [0.5, 1.5, 2.0], "new multiplier drift")
        for strength in new_strengths:
            for arm, prompt_ids in (("QUALIFIED_UNSAFE", route["qualified_unsafe_prompt_ids"]), ("QUALIFIED_SAFE", route["qualified_safe_prompt_ids"])):
                for prompt_id in prompt_ids:
                    prompt = prompt_by_key[(route["target_cwe"], int(prompt_id))]
                    multiplier = float(strength["multiplier"])
                    worklist.append({
                        "record_index": len(worklist),
                        "record_id": f"CAL|{route['target_cwe']}|L{route['layer']}|F{route['feature_id']}|M{multiplier:.1f}|{arm}|P{prompt_id}",
                        "target_cwe": route["target_cwe"], "layer": int(route["layer"]),
                        "feature_id": int(route["feature_id"]), "route_key": route["route_key"],
                        "statistical_rank": int(route["statistical_rank"]), "composite_score": float(route["composite_score"]),
                        "screening_alpha": float(route["screening_alpha"]), "strength_multiplier": multiplier,
                        "calibration_alpha": float(strength["alpha"]), "arm": arm, "prompt_id": int(prompt_id),
                        "source_index": int(prompt["source_index"]), "language": prompt["language"],
                        "source_prompt": prompt["source_prompt"], "source_prompt_sha256": prompt["source_prompt_sha256"],
                        "rendered_prompt_sha256": prompt["rendered_prompt_sha256"], "input_token_count": int(prompt["input_token_count"]),
                    })
    require(len(worklist) == EXPECTED_COUNT and len({x["record_id"] for x in worklist}) == EXPECTED_COUNT, "worklist count/uniqueness mismatch")
    return worklist


def load_inputs(resume: bool) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    for path in (PROTOCOL, CAUSAL_PROTOCOL, EXECUTION):
        require(path.is_file(), f"missing frozen input: {path}")
    require(sha256_file(PROTOCOL) == EXPECTED_PROTOCOL_SHA256, "strength protocol drift")
    require(sha256_file(CAUSAL_PROTOCOL) == EXPECTED_CAUSAL_PROTOCOL_SHA256, "causal protocol drift")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    causal_protocol = json.loads(CAUSAL_PROTOCOL.read_text(encoding="utf-8"))
    execution = json.loads(EXECUTION.read_text(encoding="utf-8"))
    require(protocol["status"] == "FROZEN_BEFORE_STRENGTH_GENERATION", "strength protocol not frozen")
    require(execution["status"] == "FROZEN_BEFORE_STRENGTH_GENERATION", "execution protocol not frozen")
    require(execution["runner_sha256"] == sha256_file(RUNNER), "strength runner drift")
    require(execution["shared_helper_sha256"] == sha256_file(SHARED), "shared helper drift")
    require(execution["strength_protocol_sha256"] == sha256_file(PROTOCOL), "execution-to-strength protocol mismatch")
    worklist = build_worklist(protocol, causal_protocol)
    require(execution["worklist_count"] == len(worklist) and execution["worklist_canonical_sha256"] == canonical_sha256(worklist), "execution worklist mismatch")
    records: list[dict[str, Any]] = []
    if CHECKPOINT.exists():
        require(resume, "checkpoint exists; use --resume")
        checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
        require(checkpoint["execution_protocol_sha256"] == sha256_file(EXECUTION), "checkpoint execution protocol drift")
        require(checkpoint["runner_sha256"] == sha256_file(RUNNER), "checkpoint runner drift")
        records = checkpoint["records"]
        require([x["record_id"] for x in records] == [x["record_id"] for x in worklist[:len(records)]], "checkpoint is not exact worklist prefix")
    else:
        require(not resume, "--resume supplied without checkpoint")
    return protocol, causal_protocol, execution, worklist, records


def preflight(resume: bool) -> dict[str, Any]:
    protocol, _causal, execution, worklist, records = load_inputs(resume)
    shared.model_snapshot_path()
    for layer in (7, 15, 23):
        shared.sae_weight_path(layer)
    return {
        "schema_version": "phase22f_strength_generation_preflight_v1", "status": "PASS_APPROVED",
        "expected_records": len(worklist), "completed_checkpoint_records": len(records),
        "resume_required": bool(records), "strength_protocol_sha256": sha256_file(PROTOCOL),
        "execution_protocol_sha256": sha256_file(EXECUTION), "runner_sha256": sha256_file(RUNNER),
        "shared_helper_sha256": sha256_file(SHARED), "worklist_canonical_sha256": execution["worklist_canonical_sha256"],
        "new_multipliers": protocol["strength_grid"]["new_generation_multipliers"],
        "model_loaded": False, "sae_loaded": False, "generation_run": False,
        "scanner_run": False, "stage22g_started": False, "heldout_used": False,
    }


def run(resume: bool) -> None:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    require(torch.cuda.is_available(), "CUDA unavailable")
    protocol, causal_protocol, _execution, worklist, records = load_inputs(resume)
    snapshot = shared.model_snapshot_path()
    os.environ["HF_HUB_OFFLINE"] = "1"
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(snapshot, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={"": "cuda:0"}, local_files_only=True).eval()
    devices = sorted({str(x.device) for x in model.parameters()})
    require(devices == ["cuda:0"], f"unauthorized model offload: {devices}")
    prior_elapsed = float(json.loads(CHECKPOINT.read_text(encoding="utf-8")).get("elapsed_seconds", 0.0)) if CHECKPOINT.exists() else 0.0
    started = time.perf_counter()
    run_state = {
        "schema_version": "phase22f_strength_generation_run_v1", "status": "IN_PROGRESS",
        "expected_records": EXPECTED_COUNT, "completed_records": len(records),
        "strength_protocol_sha256": sha256_file(PROTOCOL), "execution_protocol_sha256": sha256_file(EXECUTION),
        "runner_sha256": sha256_file(RUNNER), "shared_helper_sha256": sha256_file(SHARED),
        "model_id": shared.MODEL_ID, "model_revision": shared.MODEL_REVISION, "sae_revision": shared.SAE_REVISION,
        "gpu": torch.cuda.get_device_name(0), "parameter_devices": devices,
        "torch_version": torch.__version__, "transformers_version": transformers.__version__, "platform": platform.platform(),
        "scanner_run": False, "stage22g_started": False, "heldout_used": False,
    }
    atomic_json(RUN_MANIFEST, run_state)
    current_layer: int | None = None
    directions: dict[int, Any] = {}
    handle: Any = None
    steering_vector: Any = None

    def steering_hook(_module: Any, _args: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        require(steering_vector is not None, "steering vector not set")
        hidden[:, -1, :].add_(steering_vector.to(dtype=hidden.dtype))
        return output

    try:
        for index in range(len(records), len(worklist)):
            item = worklist[index]
            layer = int(item["layer"])
            if layer != current_layer:
                if handle is not None:
                    handle.remove()
                directions.clear()
                gc.collect()
                torch.cuda.empty_cache()
                feature_ids = sorted({int(x["feature_id"]) for x in worklist if int(x["layer"]) == layer})
                directions = shared.load_decoder_directions(layer, feature_ids, "cuda:0")
                handle = model.model.layers[layer].register_forward_hook(steering_hook)
                current_layer = layer
            steering_vector = directions[int(item["feature_id"])] * float(item["calibration_alpha"])
            require(bool(torch.isfinite(steering_vector).all()), f"nonfinite steering vector: {item['record_id']}")
            rendered = tokenizer.apply_chat_template([{"role": "user", "content": item["source_prompt"]}], tokenize=False, add_generation_prompt=True)
            require(sha256_text(rendered) == item["rendered_prompt_sha256"], f"rendered prompt drift: {item['record_id']}")
            inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to("cuda:0")
            require(int(inputs["input_ids"].shape[-1]) == item["input_token_count"], f"input token drift: {item['record_id']}")
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            generated_text = ""
            generated_count = 0
            status = "FAILED"
            exception = None
            try:
                with torch.inference_mode():
                    output = model.generate(**inputs, max_new_tokens=512, temperature=0.2, top_p=0.95, do_sample=True, pad_token_id=tokenizer.pad_token_id)
                suffix = output[0, inputs["input_ids"].shape[-1]:]
                generated_count = int(suffix.shape[0])
                generated_text = tokenizer.decode(suffix, skip_special_tokens=True)
                status = "SUCCESS"
            except Exception as exc:
                exception = {"type": type(exc).__name__, "message": str(exc)}
                torch.cuda.empty_cache()
            records.append({
                "schema_version": "phase22f_strength_generation_record_v1", "record_index": index,
                "record_id": item["record_id"], "target_cwe": item["target_cwe"], "layer": layer,
                "feature_id": item["feature_id"], "route_key": item["route_key"],
                "statistical_rank": item["statistical_rank"], "composite_score": item["composite_score"],
                "screening_alpha": item["screening_alpha"], "strength_multiplier": item["strength_multiplier"],
                "calibration_alpha": item["calibration_alpha"], "direction": "POSITIVE",
                "intervention": "DIRECT_LAST_TOKEN_RESIDUAL_ADDITION_EVERY_HOOK_CALL",
                "arm": item["arm"], "prompt_id": item["prompt_id"], "source_index": item["source_index"],
                "language": item["language"], "source_prompt_sha256": item["source_prompt_sha256"],
                "rendered_prompt_sha256": item["rendered_prompt_sha256"], "input_token_count": item["input_token_count"],
                "method": "MODEL3_STRENGTH_CALIBRATION", "seed": 42, "seed_reset_immediately_before_generation": True,
                "generation_settings": causal_protocol["generation"], "generated_text": generated_text,
                "generated_code": generated_text, "generated_text_sha256": sha256_text(generated_text),
                "generated_token_count": generated_count, "generation_status": status,
                "validity": shared.validity(generated_text), "exception": exception,
                "model_id": shared.MODEL_ID, "model_revision": shared.MODEL_REVISION,
                "sae_revision": shared.SAE_REVISION, "sae_weight_sha256": shared.SAE_SHA256[layer],
                "causal_steering_applied": True, "scanner_run": False,
                "execution_protocol_sha256": sha256_file(EXECUTION),
            })
            if len(records) % 5 == 0 or len(records) == EXPECTED_COUNT:
                elapsed = prior_elapsed + time.perf_counter() - started
                atomic_json(CHECKPOINT, {
                    "schema_version": "phase22f_strength_generation_checkpoint_v1",
                    "execution_protocol_sha256": sha256_file(EXECUTION), "runner_sha256": sha256_file(RUNNER),
                    "completed_records": len(records), "records": records, "elapsed_seconds": elapsed,
                })
                run_state.update({
                    "completed_records": len(records), "elapsed_seconds": elapsed,
                    "failure_count": sum(x["generation_status"] != "SUCCESS" for x in records),
                    "invalid_count": sum(not x["validity"]["is_valid"] for x in records),
                })
                atomic_json(RUN_MANIFEST, run_state)
                print(f"PHASE22F STRENGTH {len(records)}/{EXPECTED_COUNT} L{layer} failures={run_state['failure_count']} invalid={run_state['invalid_count']} elapsed_min={elapsed/60:.1f}", flush=True)
            del inputs
    finally:
        if handle is not None:
            handle.remove()
    atomic_json(OUTPUT, {"schema_version": "phase22f_strength_generations_v1", "status": "COMPLETE", "execution_protocol_sha256": sha256_file(EXECUTION), "records": records})
    output_hash = sha256_file(OUTPUT)
    final = {
        "schema_version": "phase22f_strength_generation_manifest_v1", "status": "COMPLETE",
        "strength_protocol_sha256": sha256_file(PROTOCOL), "execution_protocol_sha256": sha256_file(EXECUTION),
        "runner_sha256": sha256_file(RUNNER), "shared_helper_sha256": sha256_file(SHARED),
        "record_count": len(records), "success_count": sum(x["generation_status"] == "SUCCESS" for x in records),
        "failure_count": sum(x["generation_status"] != "SUCCESS" for x in records),
        "invalid_count": sum(not x["validity"]["is_valid"] for x in records),
        "output_path": "revision/model3/phase22/outputs/phase22f_strength_generations.json", "output_sha256": output_hash,
        "record_ids_canonical_sha256": canonical_sha256([x["record_id"] for x in records]),
        "strength_generation_run": True, "scanner_run": False, "stage22g_started": False, "heldout_used": False,
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
        require(not OUTPUT.exists() and not FINAL_MANIFEST.exists(), "immutable strength output already exists")
        run(args.resume)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"PHASE22F_STRENGTH_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
