#!/usr/bin/env python
"""Run frozen Stage 22E Model3 causal-candidate steering locally."""

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

ROOT = Path(__file__).resolve().parents[4]
PHASE = ROOT / "revision/model3/phase22"
OUT = PHASE / "outputs"
PROTOCOL = OUT / "phase22e_causal_validation_protocol.json"
DENOMINATOR = OUT / "phase22e_paired_denominator_manifest.json"
EXECUTION = OUT / "phase22e_steering_execution_protocol.json"
OUTPUT = OUT / "phase22e_steered_generations.json"
CHECKPOINT = OUT / "phase22e_steered_generation_checkpoint.json"
RUN_MANIFEST = OUT / "phase22e_steered_generation_run_manifest.json"
FINAL_MANIFEST = OUT / "phase22e_steered_generation_manifest.json"
RUNNER = Path(__file__).resolve()

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
SAE_REVISION = "c37e53c4bb07127ad17ab88f28b93d4e87142e59"
EXPECTED_PROTOCOL_SHA256 = "e0fcf227460e7408471caabb1d695fa1905bc51073a1581776aa1772b13932f4"
EXPECTED_DENOMINATOR_SHA256 = "30921b90579dbb9d350c256f90f720332bf41d12f6c4ce6c730ea61240ec7dde"
EXPECTED_COUNT = 1170
TARGET_ORDER = {"CWE-120": 0, "CWE-327": 1, "CWE-89": 2}
SAE_SHA256 = {
    7: "e9c88dc39dd89bc80bcea688c91739a37d1ec55bd7bf676f9cfb4bbf847fbd73",
    15: "1339c52258e95bc64535a90e45067187722e659e0ee03c976db813614b29ab5b",
    23: "a593d0da4a10cde4674b48749bf573b14e40119c1cff34cf2d499f334940ef4b",
}


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def model_snapshot_path() -> Path:
    path = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots" / MODEL_REVISION
    require(path.is_dir() and (path / "config.json").is_file(), f"pinned model snapshot missing: {path}")
    return path


def sae_weight_path(layer: int) -> Path:
    path = Path.home() / ".cache/huggingface/hub/models--andyrdt--saes-qwen2.5-7b-instruct/snapshots" / SAE_REVISION / f"resid_post_layer_{layer}/trainer_0/ae.pt"
    require(path.is_file(), f"pinned SAE weight missing: {path}")
    require(sha256_file(path) == SAE_SHA256[layer], f"SAE weight hash mismatch at layer {layer}")
    return path


def validity(text: str) -> dict[str, Any]:
    stripped = text.strip()
    markers = ("def ", "int ", "char ", "return", "if ", "for ", "{", "(")
    marker = next((x for x in markers if x in text), None)
    valid = len(stripped) >= 10 and marker is not None
    return {
        "is_valid": valid,
        "invalid_reason": None if valid else ("EMPTY_OR_TOO_SHORT" if len(stripped) < 10 else "NO_CODE_MARKER"),
        "stripped_character_count": len(stripped),
        "matched_marker": marker,
    }


def build_worklist(protocol: dict[str, Any], denominator: dict[str, Any]) -> list[dict[str, Any]]:
    prompt_by_key = {(x["target_cwe"], int(x["prompt_id"])): x for x in protocol["paired_baseline_prompts"]}
    assignments = sorted(
        protocol["candidate_assignments"],
        key=lambda x: (int(x["layer"]), TARGET_ORDER[x["target_cwe"]], int(x["statistical_rank"])),
    )
    worklist: list[dict[str, Any]] = []
    for assignment in assignments:
        target = assignment["target_cwe"]
        target_denominator = denominator["targets"][target]
        arms = (
            ("QUALIFIED_UNSAFE", target_denominator["qualified_unsafe_prompt_ids"]),
            ("QUALIFIED_SAFE", target_denominator["qualified_safe_prompt_ids"]),
        )
        for arm, prompt_ids in arms:
            for prompt_id in prompt_ids:
                prompt = prompt_by_key[(target, int(prompt_id))]
                worklist.append({
                    "record_index": len(worklist),
                    "record_id": f"STEER|{target}|L{assignment['layer']}|F{assignment['feature_id']}|{arm}|P{prompt_id}",
                    "target_cwe": target,
                    "layer": int(assignment["layer"]),
                    "feature_id": int(assignment["feature_id"]),
                    "statistical_rank": int(assignment["statistical_rank"]),
                    "composite_score": float(assignment["composite_score"]),
                    "screening_alpha": float(assignment["screening_alpha"]),
                    "route_key": assignment["route_key"],
                    "arm": arm,
                    "prompt_id": int(prompt_id),
                    "source_index": int(prompt["source_index"]),
                    "language": prompt["language"],
                    "source_prompt": prompt["source_prompt"],
                    "source_prompt_sha256": prompt["source_prompt_sha256"],
                    "rendered_prompt_sha256": prompt["rendered_prompt_sha256"],
                    "input_token_count": int(prompt["input_token_count"]),
                })
    require(len(worklist) == EXPECTED_COUNT, f"worklist count mismatch: {len(worklist)}")
    require(len({x["record_id"] for x in worklist}) == EXPECTED_COUNT, "duplicate steering record IDs")
    return worklist


def load_inputs(resume: bool) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    for path in (PROTOCOL, DENOMINATOR, EXECUTION):
        require(path.is_file(), f"missing frozen input: {path}")
    require(sha256_file(PROTOCOL) == EXPECTED_PROTOCOL_SHA256, "causal protocol drift")
    require(sha256_file(DENOMINATOR) == EXPECTED_DENOMINATOR_SHA256, "paired denominator drift")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    denominator = json.loads(DENOMINATOR.read_text(encoding="utf-8"))
    execution = json.loads(EXECUTION.read_text(encoding="utf-8"))
    require(denominator["status"] == "FROZEN_BEFORE_STEERING" and denominator["all_targets_evaluable"], "denominator gate not passed")
    require(execution["status"] == "FROZEN_BEFORE_STEERED_GENERATION", "execution protocol not frozen")
    require(execution["runner_sha256"] == sha256_file(RUNNER), "steering runner drift")
    require(execution["causal_protocol_sha256"] == EXPECTED_PROTOCOL_SHA256, "execution-to-causal protocol mismatch")
    require(execution["paired_denominator_sha256"] == EXPECTED_DENOMINATOR_SHA256, "execution-to-denominator mismatch")
    worklist = build_worklist(protocol, denominator)
    require(execution["worklist_count"] == EXPECTED_COUNT, "frozen worklist count mismatch")
    require(execution["worklist_canonical_sha256"] == canonical_sha256(worklist), "frozen worklist hash mismatch")
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
    return protocol, denominator, execution, worklist, records


def preflight(resume: bool) -> dict[str, Any]:
    _protocol, denominator, execution, worklist, records = load_inputs(resume)
    model_snapshot_path()
    for layer in (7, 15, 23):
        sae_weight_path(layer)
    return {
        "schema_version": "phase22e_steered_generation_preflight_v1",
        "status": "PASS_APPROVED",
        "expected_records": len(worklist),
        "completed_checkpoint_records": len(records),
        "resume_required": bool(records),
        "execution_protocol_sha256": sha256_file(EXECUTION),
        "causal_protocol_sha256": sha256_file(PROTOCOL),
        "paired_denominator_sha256": sha256_file(DENOMINATOR),
        "runner_sha256": sha256_file(RUNNER),
        "worklist_canonical_sha256": execution["worklist_canonical_sha256"],
        "qualified_counts": {target: {"unsafe": value["qualified_unsafe_count"], "safe": value["qualified_safe_count"]} for target, value in denominator["targets"].items()},
        "model_loaded": False,
        "sae_loaded": False,
        "generation_run": False,
        "scanner_run": False,
        "heldout_used": False,
    }


def load_decoder_directions(layer: int, feature_ids: list[int], device: str) -> dict[int, Any]:
    import torch
    state = torch.load(sae_weight_path(layer), map_location="cpu", weights_only=True, mmap=True)
    weight = state["decoder.weight"]
    require(tuple(weight.shape) == (3584, 131072), f"decoder shape mismatch at layer {layer}")
    directions = {feature_id: weight[:, feature_id].detach().float().to(device).contiguous() for feature_id in feature_ids}
    for feature_id, direction in directions.items():
        require(bool(torch.isfinite(direction).all()), f"nonfinite decoder direction L{layer}/F{feature_id}")
        require(abs(float(direction.norm().item()) - 1.0) <= 1e-4, f"decoder direction norm drift L{layer}/F{feature_id}")
    del weight, state
    gc.collect()
    return directions


def run(resume: bool) -> None:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    require(torch.cuda.is_available(), "CUDA unavailable")
    protocol, denominator, execution, worklist, records = load_inputs(resume)
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
        "schema_version": "phase22e_steered_generation_run_v1", "status": "IN_PROGRESS",
        "expected_records": EXPECTED_COUNT, "completed_records": len(records),
        "execution_protocol_sha256": sha256_file(EXECUTION), "causal_protocol_sha256": sha256_file(PROTOCOL),
        "paired_denominator_sha256": sha256_file(DENOMINATOR), "runner_sha256": sha256_file(RUNNER),
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "sae_revision": SAE_REVISION,
        "gpu": torch.cuda.get_device_name(0), "parameter_devices": devices,
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "platform": platform.platform(), "scanner_run": False, "heldout_used": False,
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
                    handle = None
                directions.clear()
                gc.collect()
                torch.cuda.empty_cache()
                feature_ids = sorted({int(x["feature_id"]) for x in worklist if int(x["layer"]) == layer})
                directions = load_decoder_directions(layer, feature_ids, "cuda:0")
                handle = model.model.layers[layer].register_forward_hook(steering_hook)
                current_layer = layer
            steering_vector = directions[int(item["feature_id"])] * float(item["screening_alpha"])
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
                "schema_version": "phase22e_steered_generation_record_v1",
                "record_index": index, "record_id": item["record_id"],
                "target_cwe": item["target_cwe"], "layer": layer,
                "feature_id": item["feature_id"], "statistical_rank": item["statistical_rank"],
                "composite_score": item["composite_score"], "route_key": item["route_key"],
                "screening_alpha": item["screening_alpha"], "direction": "POSITIVE",
                "intervention": "DIRECT_LAST_TOKEN_RESIDUAL_ADDITION_EVERY_HOOK_CALL",
                "arm": item["arm"], "prompt_id": item["prompt_id"],
                "source_index": item["source_index"], "language": item["language"],
                "source_prompt_sha256": item["source_prompt_sha256"],
                "rendered_prompt_sha256": item["rendered_prompt_sha256"],
                "input_token_count": item["input_token_count"], "method": "MODEL3_CANDIDATE_SCREEN",
                "seed": 42, "seed_reset_immediately_before_generation": True,
                "generation_settings": protocol["generation"], "generated_text": generated_text,
                "generated_code": generated_text, "generated_text_sha256": sha256_text(generated_text),
                "generated_token_count": generated_count, "generation_status": status,
                "validity": validity(generated_text), "exception": exception,
                "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                "sae_revision": SAE_REVISION, "sae_weight_sha256": SAE_SHA256[layer],
                "causal_steering_applied": True, "scanner_run": False,
                "execution_protocol_sha256": sha256_file(EXECUTION),
            })
            if len(records) % 5 == 0 or len(records) == EXPECTED_COUNT:
                elapsed = prior_elapsed + time.perf_counter() - started
                atomic_json(CHECKPOINT, {
                    "schema_version": "phase22e_steered_generation_checkpoint_v1",
                    "execution_protocol_sha256": sha256_file(EXECUTION), "runner_sha256": sha256_file(RUNNER),
                    "completed_records": len(records), "records": records, "elapsed_seconds": elapsed,
                })
                run_state.update({
                    "completed_records": len(records), "elapsed_seconds": elapsed,
                    "failure_count": sum(x["generation_status"] != "SUCCESS" for x in records),
                    "invalid_count": sum(not x["validity"]["is_valid"] for x in records),
                })
                atomic_json(RUN_MANIFEST, run_state)
                print(f"PHASE22E STEER {len(records)}/{EXPECTED_COUNT} L{layer} failures={run_state['failure_count']} invalid={run_state['invalid_count']} elapsed_min={elapsed/60:.1f}", flush=True)
            del inputs
    finally:
        if handle is not None:
            handle.remove()

    atomic_json(OUTPUT, {
        "schema_version": "phase22e_steered_generations_v1", "status": "COMPLETE",
        "execution_protocol_sha256": sha256_file(EXECUTION), "records": records,
    })
    output_hash = sha256_file(OUTPUT)
    final = {
        "schema_version": "phase22e_steered_generation_manifest_v1", "status": "COMPLETE",
        "execution_protocol_sha256": sha256_file(EXECUTION), "causal_protocol_sha256": sha256_file(PROTOCOL),
        "paired_denominator_sha256": sha256_file(DENOMINATOR), "runner_sha256": sha256_file(RUNNER),
        "record_count": len(records), "success_count": sum(x["generation_status"] == "SUCCESS" for x in records),
        "failure_count": sum(x["generation_status"] != "SUCCESS" for x in records),
        "invalid_count": sum(not x["validity"]["is_valid"] for x in records),
        "output_path": "revision/model3/phase22/outputs/phase22e_steered_generations.json",
        "output_sha256": output_hash, "record_ids_canonical_sha256": canonical_sha256([x["record_id"] for x in records]),
        "causal_steering_run": True, "scanner_run": False, "heldout_used": False,
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
        require(not OUTPUT.exists() and not FINAL_MANIFEST.exists(), "immutable steered output already exists")
        run(args.resume)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"PHASE22E_STEERING_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
