#!/usr/bin/env python3
"""Run the frozen Phase 17 AlwaysOn held-out condition after explicit approval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from phase17_common import (
    ALPHA, CHECKPOINT_INTERVAL, COMMON_HELPER, EXECUTION_APPROVAL, FEATURES, INPUT_MANIFEST,
    LAYER, OUTPUTS, SEED, atomic_json, capture_rng_state, empty_status,
    execution_approval, generate_with_alwayson, load_model_and_sae,
    load_prepared_manifest, read_json, relative, require,
    require_metadata_free_runner, restore_rng_state, sha256_path, sha256_text,
)


RUNNER = Path(__file__).resolve()
OUTPUT = OUTPUTS / "alwayson_security_seed42_outputs.json"
CHECKPOINT = OUTPUTS / "alwayson_security_seed42_checkpoint.json"
RUN_MANIFEST = OUTPUTS / "alwayson_security_seed42_run_manifest.json"
PREFLIGHT = OUTPUTS / "alwayson_security_preflight.json"


def condition(preparation: dict[str, Any]) -> dict[str, Any]:
    return {"method": "B*-AlwaysOn-L19", "information_tier": "METADATA_FREE",
            "layer": LAYER, "features": list(FEATURES),
            "alpha_by_feature": {str(feature): ALPHA for feature in FEATURES},
            "simultaneous": True, "seed": SEED,
            "routing_policy": "CONSTANT_NO_CONDITIONAL_ROUTE", "routing_fields_consumed": [],
            "input_manifest_sha256": preparation["input_manifest_sha256"],
            "runner_sha256": sha256_path(RUNNER),
            "shared_helper_sha256": sha256_path(COMMON_HELPER)}


def preflight(resume: bool) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    manifest, preparation = load_prepared_manifest()
    records = manifest["populations"]["heldout"]["generation_records"]
    require(len(records) == 575, "Security population count mismatch")
    ids = [int(row["prompt_id"]) for row in records]
    require(len(set(ids)) == 575, "Security prompt IDs are not unique")
    require(all(sha256_text(row["prompt_text"]) == row["prompt_text_sha256"] for row in records),
            "Security prompt hash mismatch")
    routing = require_metadata_free_runner(RUNNER)
    completed = 0
    resume_required = CHECKPOINT.exists()
    if CHECKPOINT.exists():
        require(resume, "Security checkpoint exists; --resume is required")
        checkpoint = read_json(CHECKPOINT)
        prior = checkpoint.get("records", [])
        require([int(row["prompt_id"]) for row in prior] == ids[:len(prior)],
                "Security checkpoint is not an exact frozen-order prefix")
        require(checkpoint.get("condition") == condition(preparation), "Security checkpoint condition mismatch")
        require(isinstance(checkpoint.get("rng_state"), dict), "Security checkpoint RNG state missing")
        completed = len(prior)
    else:
        require(not resume, "--resume supplied without a security checkpoint")
    require(not OUTPUT.exists(), "Final security output already exists and is immutable")
    approval_status = "PASS_AWAITING_EXECUTION_APPROVAL"
    if EXECUTION_APPROVAL.is_file():
        execution_approval(RUNNER)
        approval_status = "PASS_APPROVED"
    result = {"schema_version": "phase17_alwayson_security_preflight_v1",
              "status": approval_status,
              "model_loaded": False, "generation_run": False, "scanner_run": False,
              "prompt_count": 575, "completed_checkpoint_records": completed,
              "resume_required": resume_required, "condition": condition(preparation),
              "routing_audit": routing, "input_manifest_path": relative(INPUT_MANIFEST),
              "output_path": relative(OUTPUT), "checkpoint_path": relative(CHECKPOINT),
              "run_manifest_path": relative(RUN_MANIFEST)}
    return result, records, preparation


def run(records: list[dict[str, Any]], preparation: dict[str, Any]) -> None:
    execution_approval(RUNNER)
    approved_condition = condition(preparation)
    prior: list[dict[str, Any]] = []
    saved_rng = None
    if CHECKPOINT.exists():
        checkpoint = read_json(CHECKPOINT)
        prior = checkpoint["records"]
        saved_rng = checkpoint["rng_state"]
    torch, tokenizer, model, sae = load_model_and_sae()
    if saved_rng is not None:
        restore_rng_state(torch, saved_rng)
    results = list(prior)
    run_manifest = {"schema_version": "phase17_alwayson_security_run_manifest_v1",
                    "status": "IN_PROGRESS", "condition": approved_condition,
                    "completed_records": len(results), "routing_fields_consumed": []}
    atomic_json(RUN_MANIFEST, run_manifest)
    for source in records[len(results):]:
        generated, generated_tokens, input_tokens, tracker = "", 0, 0, {"hook_call_count": 0}
        failure = None
        try:
            generated, generated_tokens, input_tokens, tracker = generate_with_alwayson(
                torch, tokenizer, model, sae, source["prompt_text"], 1024, 512,
            )
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
        results.append({"schema_version": "phase17_alwayson_security_record_v1",
                        "prompt_id": int(source["prompt_id"]), "stable_id": source["stable_id"],
                        "source_index": int(source["source_index"]), "language": source["language"],
                        "prompt_text": source["prompt_text"],
                        "prompt_text_sha256": source["prompt_text_sha256"],
                        "method": "B*-AlwaysOn-L19", "information_tier": "METADATA_FREE",
                        "routing_policy": "CONSTANT_NO_CONDITIONAL_ROUTE",
                        "routing_fields_consumed": [], "layer": LAYER, "features": list(FEATURES),
                        "alpha_by_feature": {str(feature): ALPHA for feature in FEATURES},
                        "all_features_simultaneous": True, "seed": SEED,
                        "generated_code": generated, "input_token_count": input_tokens,
                        "generated_token_count": generated_tokens, "hook_evidence": tracker,
                        "generation_status": "COMPLETED" if failure is None else "FAILED",
                        "failure_reason": failure, "empty_status": empty_status(generated),
                        "input_manifest_sha256": preparation["input_manifest_sha256"],
                        "runner_sha256": sha256_path(RUNNER),
                        "shared_helper_sha256": sha256_path(COMMON_HELPER)})
        if len(results) % CHECKPOINT_INTERVAL == 0 or len(results) == len(records):
            atomic_json(CHECKPOINT, {"schema_version": "phase17_alwayson_security_checkpoint_v1",
                                    "condition": approved_condition, "records": results,
                                    "rng_state": capture_rng_state(torch)})
            run_manifest["completed_records"] = len(results)
            atomic_json(RUN_MANIFEST, run_manifest)
    require(len(results) == 575, "Security generation coverage incomplete")
    atomic_json(OUTPUT, results)
    run_manifest.update({"status": "COMPLETE", "output_sha256": sha256_path(OUTPUT)})
    atomic_json(RUN_MANIFEST, run_manifest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result, records, preparation = preflight(args.resume)
    atomic_json(PREFLIGHT, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.preflight_only:
        return 0
    run(records, preparation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
