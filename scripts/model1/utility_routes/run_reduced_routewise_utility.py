#!/usr/bin/env python3
"""Run the approved 726-task, nine-route active B* utility experiment."""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from routewise_utility_common import (
    APPROVAL as FULL_APPROVAL, EXPECTED_ROUTES, OUTPUTS, PHASE17_INPUT,
    atomic_json, baseline_output, baseline_run_manifest, canonical_sha256,
    ordered_tasks, read_json, relative, render_task, require, restore_rng_state,
    route_slug, sha256_path, sha256_text, validate_cache, validate_sources,
)
from run_routewise_utility import (
    COMMON as SHARED_COMMON, RUNNER as FULL_RUNNER, base_record, environment,
    generate, load_model, load_sae, write_progress,
)


RUNNER = Path(__file__).resolve()
PROTOCOL = OUTPUTS / "reduced_routewise_utility_protocol.json"
APPROVAL = OUTPUTS / "reduced_routewise_utility_execution_approval.json"
EXPECTED_B0_SHA256 = "889b20226cbf74badf30d30ec221ac4a028aeebc58756ecf3bce3a5e28dcf001"
EXPECTED_TASKS = 726
CHECKPOINT_INTERVAL = 25


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def output_path(cwe_id: str, route: dict[str, Any]) -> Path:
    return OUTPUTS / f"reduced_route_{route_slug(cwe_id, route)}_outputs.json"


def checkpoint_path(cwe_id: str, route: dict[str, Any]) -> Path:
    return OUTPUTS / f"reduced_route_{route_slug(cwe_id, route)}_checkpoint.json"


def run_manifest_path(cwe_id: str, route: dict[str, Any]) -> Path:
    return OUTPUTS / f"reduced_route_{route_slug(cwe_id, route)}_run_manifest.json"


def load_reduced_protocol() -> dict[str, Any]:
    validate_sources()
    require(PROTOCOL.is_file(), "Reduced protocol is missing")
    protocol = read_json(PROTOCOL)
    require(protocol.get("schema_version") == "reduced_routewise_bstar_utility_protocol_v1",
            "Reduced protocol schema mismatch")
    require(protocol.get("status") == "FROZEN", "Reduced protocol is not frozen")
    require(protocol.get("routes") == EXPECTED_ROUTES, "Reduced route map mismatch")
    population = protocol.get("population", {})
    ids = population.get("selected_task_ids")
    require(isinstance(ids, list) and len(ids) == EXPECTED_TASKS, "Reduced task-ID count mismatch")
    require(len(set(ids)) == EXPECTED_TASKS, "Reduced task IDs are not unique")
    require(canonical_sha256(ids) == population.get("selected_task_ids_sha256"),
            "Reduced task-ID hash mismatch")
    expected_indices = sorted(random.Random(42).sample(range(1140), 150))
    require(population.get("bigcodebench_selected_source_indices") == expected_indices,
            "BigCodeBench sample does not reproduce from random.Random(42)")
    return protocol


def validate_execution_approval(protocol: dict[str, Any]) -> dict[str, Any]:
    require(APPROVAL.is_file(), "Reduced execution approval is missing")
    approval = read_json(APPROVAL)
    require(approval.get("status") == "APPROVED_FOR_EXECUTION", "Reduced execution is not approved")
    require(approval.get("protocol_sha256") == sha256_path(PROTOCOL), "Reduced approval protocol mismatch")
    hashes = approval.get("execution_hashes", {})
    expected = {RUNNER, FULL_RUNNER, SHARED_COMMON}
    require(set(hashes) == {relative(path) for path in expected}, "Reduced approval executable set mismatch")
    for path in expected:
        require(hashes[relative(path)] == sha256_path(path), f"Approved hash mismatch: {relative(path)}")
    return approval


def selected_tasks(protocol: dict[str, Any], source: dict[str, Any]) -> list[dict[str, Any]]:
    full = ordered_tasks(source)
    mapping = {str(task["stable_id"]): task for task in full}
    ids = protocol["population"]["selected_task_ids"]
    require(set(ids) <= set(mapping), "Reduced protocol references an unknown task")
    tasks = [mapping[stable_id] for stable_id in ids]
    counts = {benchmark: sum(task["benchmark"] == benchmark for task in tasks)
              for benchmark in ("humaneval", "bigcodebench", "mmlu")}
    require(counts == protocol["population"]["counts"], "Reduced benchmark counts mismatch")
    require([task["source_index"] for task in tasks if task["benchmark"] == "bigcodebench"] ==
            protocol["population"]["bigcodebench_selected_source_indices"],
            "BigCodeBench source order mismatch")
    return tasks


def load_paired_b0(protocol: dict[str, Any], source: dict[str, Any], tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = baseline_output()
    manifest_path = baseline_run_manifest()
    require(output.is_file() and manifest_path.is_file(), "Completed paired B0 is missing")
    require(sha256_path(output) == EXPECTED_B0_SHA256 == protocol["paired_b0"]["sha256"],
            "Paired B0 hash mismatch")
    manifest = read_json(manifest_path)
    require(manifest.get("status") == "COMPLETE" and manifest.get("output_sha256") == EXPECTED_B0_SHA256,
            "Paired B0 run manifest mismatch")
    rows = read_json(output)
    full_tasks = ordered_tasks(source)
    require(isinstance(rows, list) and len(rows) == len(full_tasks) == 1716, "Paired B0 full count mismatch")
    require([row["stable_id"] for row in rows] == [str(task["stable_id"]) for task in full_tasks],
            "Paired B0 full order mismatch")
    mapping = {row["stable_id"]: row for row in rows}
    selected = [mapping[str(task["stable_id"])] for task in tasks]
    require(all(isinstance(row.get("rng_state_before"), dict) for row in selected),
            "Selected B0 record lacks RNG state")
    return selected


def condition(protocol: dict[str, Any], cwe_id: str) -> dict[str, Any]:
    route = protocol["routes"][cwe_id]
    return {
        "method": "B*_SINGLE_ROUTE_ACTIVE_REDUCED_UTILITY",
        "scientific_role": protocol["scientific_role"],
        "route_cwe_id": cwe_id,
        "layer": int(route["layer"]),
        "feature": int(route["feature"]),
        "alpha": 40.0,
        "seed": 42,
        "task_count": EXPECTED_TASKS,
        "selected_task_ids_sha256": protocol["population"]["selected_task_ids_sha256"],
        "paired_b0_sha256": EXPECTED_B0_SHA256,
        "protocol_sha256": sha256_path(PROTOCOL),
        "runner_sha256": sha256_path(RUNNER),
        "full_runner_sha256": sha256_path(FULL_RUNNER),
        "shared_common_sha256": sha256_path(SHARED_COMMON),
        "generation": protocol["generation"],
        "route_selection": "EXPERIMENT_CONDITION_ONLY",
        "task_cwe_fields_consumed": [],
        "simultaneous_features": False,
    }


def validate_prefix(records: list[dict[str, Any]], tasks: list[dict[str, Any]], label: str) -> None:
    require(len(records) <= len(tasks), f"{label} contains too many records")
    require([row.get("stable_id") for row in records] ==
            [str(task["stable_id"]) for task in tasks[:len(records)]],
            f"{label} is not an exact selected-task prefix")


def route_state(protocol: dict[str, Any], tasks: list[dict[str, Any]], cwe_id: str,
                resume: bool) -> dict[str, Any]:
    route = protocol["routes"][cwe_id]
    output = output_path(cwe_id, route)
    checkpoint = checkpoint_path(cwe_id, route)
    run_manifest = run_manifest_path(cwe_id, route)
    route_condition = condition(protocol, cwe_id)
    if output.exists():
        require(run_manifest.is_file(), f"{output.name} exists without its run manifest")
        run = read_json(run_manifest)
        require(run.get("status") == "COMPLETE" and run.get("condition") == route_condition,
                f"{run_manifest.name} completion/condition mismatch")
        require(run.get("output_sha256") == sha256_path(output), f"{output.name} hash mismatch")
        records = read_json(output)
        require(isinstance(records, list) and len(records) == EXPECTED_TASKS,
                f"{output.name} final count mismatch")
        validate_prefix(records, tasks, output.name)
        return {"status": "COMPLETE", "records": records, "output": output,
                "checkpoint": checkpoint, "run_manifest": run_manifest, "condition": route_condition}
    records: list[dict[str, Any]] = []
    if checkpoint.exists():
        require(resume, f"{checkpoint.name} exists; use --resume")
        saved = read_json(checkpoint)
        require(saved.get("condition") == route_condition, f"{checkpoint.name} condition mismatch")
        records = saved.get("records", [])
        require(isinstance(records, list), f"{checkpoint.name} records malformed")
        validate_prefix(records, tasks, checkpoint.name)
    else:
        require(not run_manifest.exists(), f"{run_manifest.name} exists without checkpoint/output")
    return {"status": "PENDING", "records": records, "output": output,
            "checkpoint": checkpoint, "run_manifest": run_manifest, "condition": route_condition}


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    source, _ = validate_sources()
    protocol = load_reduced_protocol()
    approval = validate_execution_approval(protocol)
    tasks = selected_tasks(protocol, source)
    b0 = load_paired_b0(protocol, source, tasks)
    cache = validate_cache(require_sae=True)
    cwes = list(protocol["routes"]) if args.all else [args.route]
    states = {cwe_id: route_state(protocol, tasks, cwe_id, args.resume) for cwe_id in cwes}
    summary = {
        "schema_version": "reduced_routewise_bstar_utility_preflight_v1",
        "status": "PASS",
        "model_loaded": False,
        "generation_run": False,
        "task_count_per_route": len(tasks),
        "benchmark_counts": protocol["population"]["counts"],
        "route_count": len(states),
        "active_generation_count": len(tasks) * len(states),
        "paired_b0_sha256": sha256_path(baseline_output()),
        "protocol_sha256": sha256_path(PROTOCOL),
        "approval_sha256": sha256_path(APPROVAL),
        "approval_status": approval["status"],
        "cache": cache,
        "targets": {cwe: {"status": state["status"],
                          "completed_records": len(state["records"]),
                          "output": relative(state["output"])} for cwe, state in states.items()},
    }
    atomic_json(OUTPUTS / "reduced_routewise_utility_preflight.json", summary)
    return {"summary": summary, "protocol": protocol, "tasks": tasks,
            "b0": b0, "states": states}


def execute(state: dict[str, Any]) -> None:
    tasks = state["tasks"]
    b0 = state["b0"]
    protocol = state["protocol"]
    torch, tokenizer, model = load_model()
    for cwe_id, target in state["states"].items():
        if target["status"] == "COMPLETE":
            print(f"{cwe_id} already COMPLETE; skipping", flush=True)
            continue
        route = protocol["routes"][cwe_id]
        layer, feature = int(route["layer"]), int(route["feature"])
        sae = load_sae(layer)
        records = list(target["records"])
        run = {
            "schema_version": "reduced_routewise_bstar_utility_run_manifest_v1",
            "status": "IN_PROGRESS",
            "condition": target["condition"],
            "environment": environment(torch),
            "started_at_utc": utc_now(),
            "completed_records": len(records),
            "output_sha256": None,
        }
        atomic_json(target["run_manifest"], run)
        started = time.perf_counter()
        try:
            for index, task in enumerate(tasks[len(records):], start=len(records)):
                paired = b0[index]
                require(paired["stable_id"] == str(task["stable_id"]), "Paired B0 task mismatch")
                rendered, input_limit, output_limit, eos_token_ids = render_task(tokenizer, task)
                require(paired["rendered_prompt_sha256"] == sha256_text(rendered),
                        "Paired B0 rendered prompt mismatch")
                restore_rng_state(torch, paired["rng_state_before"])
                text, generated_tokens, input_tokens, tracker = generate(
                    torch, tokenizer, model, rendered, input_limit, output_limit, eos_token_ids,
                    sae=sae, layer=layer, feature=feature,
                )
                require(tracker.get("hook_call_count", 0) > 0, f"{cwe_id} hook was not entered")
                record = base_record(task, rendered, text, generated_tokens, input_tokens,
                                     target["condition"])
                record.update({
                    "paired_b0_output_sha256": sha256_text(paired["generated_text"]),
                    "paired_rng_state_sha256": canonical_sha256(paired["rng_state_before"]),
                    "hook_evidence": tracker,
                })
                records.append(record)
                if len(records) % CHECKPOINT_INTERVAL == 0 or len(records) == len(tasks):
                    write_progress(target["checkpoint"], target["condition"], records)
                    run["completed_records"] = len(records)
                    atomic_json(target["run_manifest"], run)
                    print(f"{cwe_id} checkpoint={len(records)}/{len(tasks)}", flush=True)
        except Exception as exc:
            run.update({"status": "INTERRUPTED", "completed_records": len(records),
                        "error": f"{type(exc).__name__}: {exc}", "updated_at_utc": utc_now()})
            atomic_json(target["run_manifest"], run)
            raise
        require(len(records) == EXPECTED_TASKS, f"{cwe_id} final count mismatch")
        validate_prefix(records, tasks, cwe_id)
        atomic_json(target["output"], records)
        run.update({"status": "COMPLETE", "completed_records": len(records),
                    "output_sha256": sha256_path(target["output"]),
                    "elapsed_seconds": time.perf_counter() - started,
                    "peak_cuda_memory_bytes": max(
                        (torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())),
                        default=0),
                    "completed_at_utc": utc_now()})
        atomic_json(target["run_manifest"], run)
        print(json.dumps({"status": "COMPLETE", "route": cwe_id,
                          "records": len(records), "output": relative(target["output"]),
                          "sha256": run["output_sha256"]}, indent=2), flush=True)
        del sae
        gc.collect()
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--route", choices=list(EXPECTED_ROUTES))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state = preflight(args)
    print(json.dumps(state["summary"], indent=2, sort_keys=True))
    if not args.preflight_only:
        execute(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

