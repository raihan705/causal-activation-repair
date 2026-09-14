#!/usr/bin/env python3
"""Run or preflight the frozen Phase 14 B1-CWE development condition.

`--preflight-only` uses only the Python standard library and never imports or
loads the model.  Generation is intentionally not invoked during implementation
freeze.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from phase14_common import (
    BASELINE_SUBSET, EXPECTED_BASELINE_SUBSET_SHA256, EXPECTED_RECORD_COUNT,
    EXPECTED_SOURCE_RECOVERY_SHA256, MODEL_CACHE_SNAPSHOT, MODEL_ID, OUTPUTS,
    REQUIRED_SEED, SUBMITTED_B1_PREFIX, atomic_json, capture_rng_state,
    read_json, record_source_prompt, relative, require, restore_rng_state,
    sha256_path, sha256_text, validate_preparation,
)


SPEC = OUTPUTS / "b1cwe_method_spec.json"
GUIDANCE = OUTPUTS / "b1cwe_guidance_map.json"
DEFAULT_OUTPUT = OUTPUTS / "b1cwe_dev_outputs.json"
DEFAULT_CHECKPOINT = OUTPUTS / "b1cwe_dev_checkpoint.json"
DEFAULT_RUN_MANIFEST = OUTPUTS / "b1cwe_dev_run_manifest.json"
RUNNER = Path(__file__).resolve()


def route(record: dict[str, Any], guidance: dict[str, Any]) -> dict[str, Any]:
    source = record_source_prompt(record)
    cwe = source["cwe_identifier"]
    route_cfg = guidance["routes"].get(cwe)
    if route_cfg is None:
        policy = guidance["unmapped_or_unsupported_cwe_policy"]
        require(policy["action"] == "submitted B1" and policy["guidance_available"] is False,
                f"unsupported CWE policy is not frozen for {cwe}")
        return {
            "target_cwe": cwe,
            "guidance_available": False,
            "guidance_route_status": policy["route_status"],
            "fallback_status": "SUBMITTED_B1",
        }
    require(route_cfg["guidance_available"] is False, "active route unexpectedly contains guidance")
    require(route_cfg["fallback"] == "submitted B1", "active route fallback differs")
    require(route_cfg["guidance_text"] is None, "guidance text is outside the frozen protocol")
    return {
        "target_cwe": cwe,
        "guidance_available": False,
        "guidance_route_status": "ROUTE_GUIDANCE_UNAVAILABLE",
        "fallback_status": "SUBMITTED_B1",
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    state = validate_preparation()
    require(args.seed == REQUIRED_SEED, "seed must be exactly 42")
    require(args.input.resolve() == BASELINE_SUBSET.resolve(), "only the frozen baseline subset is allowed")
    require(args.spec.resolve() == SPEC.resolve(), "only the frozen B1-CWE spec is allowed")
    require(args.guidance.resolve() == GUIDANCE.resolve(), "only the frozen guidance map is allowed")
    require(args.output.resolve() == DEFAULT_OUTPUT.resolve(), "output path differs from frozen policy")
    require(args.checkpoint.resolve() == DEFAULT_CHECKPOINT.resolve(), "checkpoint path differs from frozen policy")
    require(args.run_manifest.resolve() == DEFAULT_RUN_MANIFEST.resolve(), "run-manifest path differs from frozen policy")
    require(args.spec.is_file() and args.guidance.is_file(), "B1-CWE spec/guidance map missing")

    spec = read_json(args.spec)
    guidance = read_json(args.guidance)
    require(spec.get("schema_version") == "phase14_b1cwe_method_spec_v1", "spec schema mismatch")
    require(spec.get("information_tier") == "ORACLE_CWE", "information tier mismatch")
    require(spec.get("seed") == 42 and spec.get("generation_stage_count") == 1, "B1-CWE frozen settings mismatch")
    require(spec["input"]["sha256"] == EXPECTED_BASELINE_SUBSET_SHA256, "spec subset hash mismatch")
    require(spec["source_identity"]["source_recovery_sha256"] == EXPECTED_SOURCE_RECOVERY_SHA256,
            "spec source-recovery hash mismatch")
    require(guidance.get("source_recovery_sha256") == EXPECTED_SOURCE_RECOVERY_SHA256,
            "guidance source-recovery hash mismatch")
    require(guidance.get("fabricated_guidance_count") == 0, "fabricated guidance detected")

    decisions = [route(record, guidance) for record in state["records"]]
    require(len(decisions) == EXPECTED_RECORD_COUNT, "routing decision count mismatch")
    active_fallback = sum(d["guidance_route_status"] == "ROUTE_GUIDANCE_UNAVAILABLE" for d in decisions)
    unsupported_fallback = sum(d["guidance_route_status"] == "UNSUPPORTED_ROUTE_FALLBACK_SUBMITTED_B1" for d in decisions)
    require(active_fallback + unsupported_fallback == 120, "not every prompt resolves to the frozen fallback")
    require(all(not d["guidance_available"] and d["fallback_status"] == "SUBMITTED_B1" for d in decisions),
            "non-fallback decision in the current 120-prompt subset")

    if args.checkpoint.exists():
        require(args.resume, "checkpoint exists; --resume is required")
        checkpoint = read_json(args.checkpoint)
        require(checkpoint.get("condition", {}).get("spec_sha256") == sha256_path(args.spec),
                "checkpoint spec hash mismatch")
        completed = checkpoint.get("records", [])
        expected_ids = [int(r["prompt_id"]) for r in state["records"]]
        require([int(r["prompt_id"]) for r in completed] == expected_ids[:len(completed)],
                "checkpoint is not a strict source-order prefix")
        require(isinstance(checkpoint.get("rng_state"), dict), "checkpoint RNG state missing")
    else:
        require(not args.resume, "--resume supplied without checkpoint")

    return {
        "schema_version": "phase14_b1cwe_preflight_v1",
        "status": "PASS",
        "generation_executed": False,
        "model_loaded": False,
        "held_out_accessed": False,
        "scanner_executed": False,
        "seed": 42,
        "record_count": 120,
        "source_order_preserved": True,
        "guidance_available_count": 0,
        "submitted_b1_fallback_count": 120,
        "route_guidance_unavailable_count": active_fallback,
        "unsupported_route_fallback_count": unsupported_fallback,
        "route_counts": {
            key: sum(d["target_cwe"] == key for d in decisions)
            for key in sorted({d["target_cwe"] for d in decisions})
        },
        "information_tier": "ORACLE_CWE",
        "input": {"path": relative(args.input), "sha256": sha256_path(args.input)},
        "spec": {"path": relative(args.spec), "sha256": sha256_path(args.spec)},
        "guidance_map": {"path": relative(args.guidance), "sha256": sha256_path(args.guidance)},
        "source_recovery_sha256": EXPECTED_SOURCE_RECOVERY_SHA256,
        "runner": {"path": relative(RUNNER), "sha256": sha256_path(RUNNER)},
        "output_path": relative(args.output),
        "checkpoint_path": relative(args.checkpoint),
        "run_manifest_path": relative(args.run_manifest),
    }


def validate_checkpoint(checkpoint: dict[str, Any], condition: dict[str, Any], ids: list[int]) -> None:
    require(checkpoint.get("schema_version") == "phase14_b1cwe_checkpoint_v1", "checkpoint schema mismatch")
    require(checkpoint.get("condition") == condition, "checkpoint condition mismatch")
    records = checkpoint.get("records")
    require(isinstance(records, list), "checkpoint records missing")
    require([int(row["prompt_id"]) for row in records] == ids[:len(records)], "checkpoint order mismatch")
    require(isinstance(checkpoint.get("rng_state"), dict), "checkpoint RNG missing")


def run_generation(args: argparse.Namespace, pf: dict[str, Any]) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    state = validate_preparation()
    spec = read_json(args.spec)
    guidance = read_json(args.guidance)
    ids = [int(record["prompt_id"]) for record in state["records"]]
    condition = {
        "method": "B1-CWE", "seed": 42,
        "input_sha256": pf["input"]["sha256"], "spec_sha256": pf["spec"]["sha256"],
        "guidance_map_sha256": pf["guidance_map"]["sha256"],
        "source_recovery_sha256": EXPECTED_SOURCE_RECOVERY_SHA256,
        "runner_sha256": pf["runner"]["sha256"],
    }
    records: list[dict[str, Any]] = []
    if args.checkpoint.exists():
        checkpoint = read_json(args.checkpoint)
        validate_checkpoint(checkpoint, condition, ids)
        records = checkpoint["records"]
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_CACHE_SNAPSHOT, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_CACHE_SNAPSHOT, torch_dtype=torch.float16,
        device_map="auto", local_files_only=True,
    ).eval()
    if args.checkpoint.exists():
        restore_rng_state(torch, read_json(args.checkpoint)["rng_state"])

    start = len(records)
    run_manifest = {"schema_version": "phase14_b1cwe_run_manifest_v1", "condition": condition,
                    "status": "IN_PROGRESS", "completed_records": start}
    atomic_json(args.run_manifest, run_manifest)
    for record in state["records"][start:]:
        source = record_source_prompt(record)
        decision = route(record, guidance)
        # Every configured route resolves to the B1 fallback. Fail closed
        # if a future map attempts to activate absent guidance text.
        require(decision["guidance_available"] is False, "renderable CWE guidance is not frozen")
        rendered = SUBMITTED_B1_PREFIX + source["test_case_prompt"]
        generated_code = ""
        generated_count = 0
        status = "FAILED"
        failure_reason = None
        try:
            inputs = tokenizer(rendered, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, max_new_tokens=512, temperature=0.2, top_p=0.95,
                    do_sample=True, pad_token_id=tokenizer.pad_token_id,
                )
            new_tokens = generated[0, inputs["input_ids"].shape[1]:]
            generated_count = int(new_tokens.shape[0])
            generated_code = tokenizer.decode(new_tokens, skip_special_tokens=True)
            status = "COMPLETED"
        except Exception as exc:
            failure_reason = f"{type(exc).__name__}: {exc}"
        output_record = {
            "schema_version": "phase14_b1cwe_record_v1",
            "prompt_id": int(record["prompt_id"]), "source_index": int(record["source_index"]),
            "population_type": record["population_type"], "prompt_text": source["test_case_prompt"],
            "prompt_text_sha256": record["prompt_text_sha256"], "language": source.get("language", ""),
            "target_cwe": decision["target_cwe"], "method": "B1-CWE",
            "information_tier": "ORACLE_CWE", "guidance_available": False,
            "guidance_route_status": decision["guidance_route_status"],
            "fallback_status": decision["fallback_status"], "rendered_prompt_sha256": sha256_text(rendered),
            "generated_code": generated_code, "generated_token_count": generated_count,
            "generation_status": status, "failure_reason": failure_reason,
            "empty_status": "EMPTY" if not generated_code.strip() else "NONEMPTY", "seed": 42,
            "model_id": MODEL_ID, "model_cache_snapshot": MODEL_CACHE_SNAPSHOT,
            "input_manifest_sha256": pf["input"]["sha256"], "method_spec_sha256": pf["spec"]["sha256"],
            "guidance_map_sha256": pf["guidance_map"]["sha256"],
            "source_recovery_sha256": EXPECTED_SOURCE_RECOVERY_SHA256,
            "runner_sha256": pf["runner"]["sha256"],
        }
        records.append(output_record)
        checkpoint = {"schema_version": "phase14_b1cwe_checkpoint_v1", "condition": condition,
                      "records": records, "rng_state": capture_rng_state(torch)}
        atomic_json(args.checkpoint, checkpoint)
        run_manifest["completed_records"] = len(records)
        atomic_json(args.run_manifest, run_manifest)
    require(len(records) == EXPECTED_RECORD_COUNT, "generation did not preserve 120 records")
    atomic_json(args.output, records)
    run_manifest["status"] = "COMPLETE"
    run_manifest["output_sha256"] = sha256_path(args.output)
    atomic_json(args.run_manifest, run_manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=BASELINE_SUBSET)
    parser.add_argument("--spec", type=Path, default=SPEC)
    parser.add_argument("--guidance", type=Path, default=GUIDANCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--run-manifest", type=Path, default=DEFAULT_RUN_MANIFEST)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-output", type=Path)
    args = parser.parse_args()
    for name in ("input", "spec", "guidance", "output", "checkpoint", "run_manifest"):
        value = getattr(args, name)
        setattr(args, name, value.resolve())
    if args.preflight_output:
        args.preflight_output = args.preflight_output.resolve()
    return args


def main() -> int:
    args = parse_args()
    pf = preflight(args)
    if args.preflight_output:
        atomic_json(args.preflight_output, pf)
    print(json.dumps(pf, indent=2, sort_keys=True))
    if args.preflight_only:
        return 0
    require(not args.output.exists(), "final output already exists; immutable review required")
    run_generation(args, pf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
