#!/usr/bin/env python3
"""Run the frozen Phase 17 AlwaysOn utility suite after explicit approval."""

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
OUTPUT = OUTPUTS / "alwayson_utility_seed42_outputs.json"
CHECKPOINT = OUTPUTS / "alwayson_utility_seed42_checkpoint.json"
RUN_MANIFEST = OUTPUTS / "alwayson_utility_seed42_run_manifest.json"
PREFLIGHT = OUTPUTS / "alwayson_utility_preflight.json"
EXPECTED_COUNTS = {"humaneval": 164, "bigcodebench": 1140, "mmlu": 412}


def condition(preparation: dict[str, Any]) -> dict[str, Any]:
    return {"method": "B*-AlwaysOn-L19", "information_tier": "METADATA_FREE",
            "layer": LAYER, "features": list(FEATURES),
            "alpha_by_feature": {str(feature): ALPHA for feature in FEATURES},
            "simultaneous": True, "seed": SEED,
            "routing_policy": "CONSTANT_NO_CONDITIONAL_ROUTE", "routing_fields_consumed": [],
            "benchmark_order": ["humaneval", "bigcodebench", "mmlu"],
            "continuous_rng_stream": True,
            "input_manifest_sha256": preparation["input_manifest_sha256"],
            "runner_sha256": sha256_path(RUNNER),
            "shared_helper_sha256": sha256_path(COMMON_HELPER)}


def tasks(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for benchmark in ("humaneval", "bigcodebench", "mmlu"):
        rows = manifest["populations"][benchmark]["records"]
        require(len(rows) == EXPECTED_COUNTS[benchmark], f"{benchmark} population count mismatch")
        for row in rows:
            result.append({"benchmark": benchmark, **row})
    require(len(result) == 1716 and len({row["stable_id"] for row in result}) == 1716,
            "Utility global population/ID mismatch")
    return result


def preflight(resume: bool) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    manifest, preparation = load_prepared_manifest()
    ordered = tasks(manifest)
    routing = require_metadata_free_runner(RUNNER)
    stable_ids = [row["stable_id"] for row in ordered]
    completed = 0
    if CHECKPOINT.exists():
        require(resume, "Utility checkpoint exists; --resume is required")
        checkpoint = read_json(CHECKPOINT)
        prior = checkpoint.get("records", [])
        require([row["stable_id"] for row in prior] == stable_ids[:len(prior)],
                "Utility checkpoint is not an exact global-order prefix")
        require(checkpoint.get("condition") == condition(preparation), "Utility checkpoint condition mismatch")
        require(isinstance(checkpoint.get("rng_state"), dict), "Utility checkpoint RNG state missing")
        completed = len(prior)
    else:
        require(not resume, "--resume supplied without a utility checkpoint")
    require(not OUTPUT.exists(), "Final utility output already exists and is immutable")
    approval_status = "PASS_AWAITING_EXECUTION_APPROVAL"
    if EXECUTION_APPROVAL.is_file():
        execution_approval(RUNNER)
        approval_status = "PASS_APPROVED"
    result = {"schema_version": "phase17_alwayson_utility_preflight_v1",
              "status": approval_status,
              "model_loaded": False, "generation_run": False, "evaluator_run": False,
              "task_count": 1716, "benchmark_counts": EXPECTED_COUNTS,
              "benchmark_order": ["humaneval", "bigcodebench", "mmlu"],
              "completed_checkpoint_records": completed,
              "resume_required": CHECKPOINT.exists(), "condition": condition(preparation),
              "routing_audit": routing, "input_manifest_path": relative(INPUT_MANIFEST),
              "output_path": relative(OUTPUT), "checkpoint_path": relative(CHECKPOINT),
              "run_manifest_path": relative(RUN_MANIFEST)}
    return result, ordered, preparation


def render(tokenizer: Any, task: dict[str, Any]) -> tuple[str, int, int, int | list[int] | None]:
    benchmark = task["benchmark"]
    if benchmark == "humaneval":
        stop_strings = ["\ndef ", "\nclass ", "\n#", "\nif __name__"]
        stop_ids = [ids[0] for value in stop_strings
                    if (ids := tokenizer.encode(value, add_special_tokens=False))]
        stop_ids = list(set(stop_ids + [tokenizer.eos_token_id]))
        return task["prompt"], 2048, 256, stop_ids
    if benchmark == "bigcodebench":
        messages = [{"role": "system", "content": "You are an expert programmer. Complete the following coding task. Return only the code, no explanation."},
                    {"role": "user", "content": task["prompt"]}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), 2048, 512, None
    labels = "ABCD"
    options = "\n".join(f"{labels[index]}. {choice}" for index, choice in enumerate(task["choices"]))
    user = f"Question: {task['question']}\n\n{options}\n\nAnswer with only the letter (A, B, C, or D)."
    messages = [{"role": "system", "content": "You are a knowledgeable assistant."},
                {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), 2048, 8, None


def mmlu_prediction(response: str) -> str:
    for char in response.strip().upper():
        if char in "ABCD":
            return char
    return "X"


def run(ordered: list[dict[str, Any]], preparation: dict[str, Any]) -> None:
    execution_approval(RUNNER)
    approved_condition = condition(preparation)
    prior: list[dict[str, Any]] = []
    saved_rng = None
    if CHECKPOINT.exists():
        checkpoint = read_json(CHECKPOINT)
        prior, saved_rng = checkpoint["records"], checkpoint["rng_state"]
    torch, tokenizer, model, sae = load_model_and_sae()
    if saved_rng is not None:
        restore_rng_state(torch, saved_rng)
    results = list(prior)
    run_manifest = {"schema_version": "phase17_alwayson_utility_run_manifest_v1",
                    "status": "IN_PROGRESS", "condition": approved_condition,
                    "completed_records": len(results), "routing_fields_consumed": []}
    atomic_json(RUN_MANIFEST, run_manifest)
    for task in ordered[len(results):]:
        rendered, input_limit, output_limit, eos_token_ids = render(tokenizer, task)
        generated, generated_tokens, input_tokens, tracker = "", 0, 0, {"hook_call_count": 0}
        failure = None
        try:
            generated, generated_tokens, input_tokens, tracker = generate_with_alwayson(
                torch, tokenizer, model, sae, rendered, input_limit, output_limit,
                eos_token_id=eos_token_ids,
            )
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
        record = {"schema_version": "phase17_alwayson_utility_record_v1",
                  "stable_id": task["stable_id"], "benchmark": task["benchmark"],
                  "source_index": int(task["source_index"]),
                  "method": "B*-AlwaysOn-L19", "information_tier": "METADATA_FREE",
                  "routing_policy": "CONSTANT_NO_CONDITIONAL_ROUTE", "routing_fields_consumed": [],
                  "layer": LAYER, "features": list(FEATURES),
                  "alpha_by_feature": {str(feature): ALPHA for feature in FEATURES},
                  "all_features_simultaneous": True, "seed": SEED,
                  "source_prompt_sha256": sha256_text(task.get("prompt", task.get("question", ""))),
                  "rendered_prompt_sha256": sha256_text(rendered), "generated_text": generated,
                  "eos_token_ids": eos_token_ids,
                  "input_token_count": input_tokens, "generated_token_count": generated_tokens,
                  "hook_evidence": tracker,
                  "generation_status": "COMPLETED" if failure is None else "FAILED",
                  "failure_reason": failure, "empty_status": empty_status(generated),
                  "input_manifest_sha256": preparation["input_manifest_sha256"],
                  "runner_sha256": sha256_path(RUNNER),
                  "shared_helper_sha256": sha256_path(COMMON_HELPER)}
        if task["benchmark"] == "mmlu":
            prediction = mmlu_prediction(generated)
            record.update({"slice": task["slice"], "answer": task["answer"],
                           "predicted": prediction, "correct": prediction == task["answer"]})
        results.append(record)
        if len(results) % CHECKPOINT_INTERVAL == 0 or len(results) in (164, 1304, 1716):
            atomic_json(CHECKPOINT, {"schema_version": "phase17_alwayson_utility_checkpoint_v1",
                                    "condition": approved_condition, "records": results,
                                    "rng_state": capture_rng_state(torch)})
            run_manifest["completed_records"] = len(results)
            atomic_json(RUN_MANIFEST, run_manifest)
    require(len(results) == 1716, "Utility generation coverage incomplete")
    atomic_json(OUTPUT, results)
    run_manifest.update({"status": "COMPLETE", "output_sha256": sha256_path(OUTPUT)})
    atomic_json(RUN_MANIFEST, run_manifest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result, ordered, preparation = preflight(args.resume)
    atomic_json(PREFLIGHT, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.preflight_only:
        return 0
    run(ordered, preparation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
