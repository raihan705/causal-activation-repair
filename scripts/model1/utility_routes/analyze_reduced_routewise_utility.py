#!/usr/bin/env python3
"""Finalize the approved reduced route-wise B* utility experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any

from routewise_utility_common import (
    OUTPUTS, atomic_json, baseline_output, baseline_run_manifest,
    canonical_sha256, read_json, require, route_slug, sha256_path,
)
from run_reduced_routewise_utility import (
    PROTOCOL, load_reduced_protocol, output_path, run_manifest_path,
)
from summarize_reduced_generation import ast_valid


RETURN_MANIFEST = OUTPUTS / "reduced_routewise_humaneval_return_manifest.json"
HUMAN_INPUT = OUTPUTS / "reduced_humaneval_evaluation_input.json"
RESULTS = OUTPUTS / "reduced_routewise_utility_results.csv"
INTERVALS = OUTPUTS / "reduced_routewise_utility_intervals.csv"
SUMMARY = OUTPUTS / "reduced_routewise_utility_summary.json"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 42
EXPECTED_COUNTS = {"humaneval": 164, "bigcodebench": 150, "mmlu": 412}


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    require(bool(rows), f"No rows for {path.name}")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validated_generation(path: Path, manifest_path: Path, expected: int) -> list[dict[str, Any]]:
    require(path.is_file() and manifest_path.is_file(), f"Missing {path.name} or manifest")
    manifest = read_json(manifest_path)
    require(manifest.get("status") == "COMPLETE", f"{manifest_path.name} is not COMPLETE")
    require(manifest.get("completed_records") == expected, f"{manifest_path.name} count mismatch")
    require(manifest.get("output_sha256") == sha256_path(path), f"{path.name} hash mismatch")
    rows = read_json(path)
    require(isinstance(rows, list) and len(rows) == expected, f"{path.name} count mismatch")
    require(len({str(row["stable_id"]) for row in rows}) == expected, f"{path.name} duplicate IDs")
    return rows


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def human_result_path(slug: str) -> Path:
    return OUTPUTS / f"humaneval_reduced_{slug}_task_results.json"


def validate_human_results(
    path: Path,
    expected_ids: list[str],
    generation_rows: list[dict[str, Any]],
    expected_file_hash: str,
) -> list[bool]:
    require(path.is_file(), f"Missing {path.name}")
    require(sha256_path(path) == expected_file_hash, f"{path.name} return-manifest hash mismatch")
    rows = read_json(path)
    require(isinstance(rows, list) and len(rows) == 164, f"{path.name} count mismatch")
    require([str(row["task_id"]) for row in rows] == expected_ids, f"{path.name} order mismatch")
    require(all(isinstance(row.get("passed"), bool) for row in rows), f"{path.name} malformed passed field")
    generated = {
        str(row["stable_id"]): sha256_text(str(row["generated_text"]))
        for row in generation_rows if row["benchmark"] == "humaneval"
    }
    require(len(generated) == 164, f"{path.name} source HumanEval count mismatch")
    for row in rows:
        task_id = str(row["task_id"])
        require(row.get("completion_sha256") == generated[task_id],
                f"{path.name} completion hash mismatch for {task_id}")
    return [bool(row["passed"]) for row in rows]


def type7(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def bootstrap_delta(reference: list[bool], active: list[bool]) -> tuple[float, float, float]:
    require(len(reference) == len(active), "Paired length mismatch")
    differences = [int(right) - int(left) for left, right in zip(reference, active)]
    point = sum(differences) / len(differences)
    rng = random.Random(BOOTSTRAP_SEED)
    estimates: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        estimates.append(sum(differences[rng.randrange(len(differences))]
                             for _ in differences) / len(differences))
    return point, type7(estimates, 0.025), type7(estimates, 0.975)


def nonhuman_outcomes(rows: list[dict[str, Any]], benchmark: str) -> tuple[list[str], list[bool]]:
    selected = [row for row in rows if row["benchmark"] == benchmark]
    require(len(selected) == EXPECTED_COUNTS[benchmark], f"{benchmark} count mismatch")
    ids = [str(row["stable_id"]) for row in selected]
    if benchmark == "mmlu":
        values = [bool(row["correct"]) for row in selected]
    else:
        values = [ast_valid(str(row["generated_text"])) for row in selected]
    return ids, values


def main() -> int:
    protocol = load_reduced_protocol()
    require(RETURN_MANIFEST.is_file() and HUMAN_INPUT.is_file(), "HumanEval return is incomplete")
    returned = read_json(RETURN_MANIFEST)
    require(returned.get("schema_version") == "reduced_routewise_humaneval_return_v1",
            "HumanEval return schema mismatch")
    require(returned.get("status") == "COMPLETE" and returned.get("official_evaluator") is True,
            "HumanEval return is not an official completed evaluation")
    require(returned.get("package") == "human-eval" and returned.get("package_version") == "1.0.3",
            "HumanEval package mismatch")
    require(returned.get("protocol_sha256") == sha256_path(PROTOCOL), "Return protocol mismatch")
    require(returned.get("input_sha256") == sha256_path(HUMAN_INPUT), "Return input mismatch")
    require(returned.get("task_count_per_condition") == 164, "Return task count mismatch")

    selected_ids = [str(value) for value in protocol["population"]["selected_task_ids"]]
    selected_set = set(selected_ids)
    all_b0 = validated_generation(baseline_output(), baseline_run_manifest(), 1716)
    b0 = [row for row in all_b0 if str(row["stable_id"]) in selected_set]
    require([str(row["stable_id"]) for row in b0] == selected_ids, "Selected B0 order mismatch")

    b0_human_rows = [row for row in b0 if row["benchmark"] == "humaneval"]
    b0_human_ids = [str(row["stable_id"]) for row in b0_human_rows]
    conditions = returned.get("conditions", {})
    require(set(conditions) == {"B0", *protocol["routes"]}, "Returned conditions mismatch")
    b0_return = conditions["B0"]
    require(b0_return.get("slug") == "b0", "B0 slug mismatch")
    require(b0_return.get("source_output_sha256") == sha256_path(baseline_output()),
            "B0 returned source hash mismatch")
    b0_human = validate_human_results(
        human_result_path("b0"), b0_human_ids, b0,
        str(b0_return["task_results_sha256"]),
    )
    require(sum(b0_human) == int(b0_return["pass_count"]), "B0 pass count mismatch")

    labels = {"humaneval": "HumanEval", "bigcodebench": "BigCodeBench", "mmlu": "MMLU"}
    metrics = {"humaneval": "pass@1", "bigcodebench": "AST syntax-pass proxy", "mmlu": "accuracy"}
    b0_outcomes: dict[str, tuple[list[str], list[bool]]] = {
        "humaneval": (b0_human_ids, b0_human),
        "bigcodebench": nonhuman_outcomes(b0, "bigcodebench"),
        "mmlu": nonhuman_outcomes(b0, "mmlu"),
    }
    result_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    summary_routes: dict[str, Any] = {}
    for benchmark, display in labels.items():
        ids, values = b0_outcomes[benchmark]
        result_rows.append({
            "benchmark": display, "condition": "B0", "route_cwe_id": "",
            "layer": "", "feature": "", "alpha": "", "task_count": len(ids),
            "metric": metrics[benchmark], "success_count": sum(values),
            "score": f"{sum(values) / len(values):.6f}", "scientific_role": "REFERENCE",
        })

    for cwe_id, route in protocol["routes"].items():
        path = output_path(cwe_id, route)
        active = validated_generation(path, run_manifest_path(cwe_id, route), 726)
        returned_condition = conditions[cwe_id]
        slug = route_slug(cwe_id, route)
        require(returned_condition.get("slug") == slug, f"{cwe_id} slug mismatch")
        require(returned_condition.get("layer") == int(route["layer"]), f"{cwe_id} layer mismatch")
        require(returned_condition.get("feature") == int(route["feature"]), f"{cwe_id} feature mismatch")
        require(returned_condition.get("source_output_sha256") == sha256_path(path),
                f"{cwe_id} returned source hash mismatch")
        human = validate_human_results(
            human_result_path(slug), b0_human_ids, active,
            str(returned_condition["task_results_sha256"]),
        )
        require(sum(human) == int(returned_condition["pass_count"]), f"{cwe_id} pass count mismatch")
        active_outcomes = {
            "humaneval": (b0_human_ids, human),
            "bigcodebench": nonhuman_outcomes(active, "bigcodebench"),
            "mmlu": nonhuman_outcomes(active, "mmlu"),
        }
        route_summary: dict[str, Any] = {
            "layer": int(route["layer"]), "feature": int(route["feature"]), "benchmarks": {}
        }
        for benchmark, display in labels.items():
            b0_ids, b0_values = b0_outcomes[benchmark]
            active_ids, active_values = active_outcomes[benchmark]
            require(active_ids == b0_ids, f"{cwe_id}/{benchmark} pairing mismatch")
            point, lower, upper = bootstrap_delta(b0_values, active_values)
            score = sum(active_values) / len(active_values)
            improved = sum((not left) and right for left, right in zip(b0_values, active_values))
            regressed = sum(left and (not right) for left, right in zip(b0_values, active_values))
            result_rows.append({
                "benchmark": display, "condition": "B*_SINGLE_ROUTE_ACTIVE",
                "route_cwe_id": cwe_id, "layer": int(route["layer"]),
                "feature": int(route["feature"]), "alpha": 40,
                "task_count": len(active_ids), "metric": metrics[benchmark],
                "success_count": sum(active_values), "score": f"{score:.6f}",
                "scientific_role": "ACTIVE_COORDINATE_SENSITIVITY",
            })
            interval_rows.append({
                "benchmark": display, "contrast": f"{cwe_id} active route - paired B0",
                "route_cwe_id": cwe_id, "layer": int(route["layer"]),
                "feature": int(route["feature"]), "alpha": 40,
                "task_count": len(active_ids), "metric": metrics[benchmark],
                "point_estimate": f"{point:.6f}", "ci95_lower": f"{lower:.6f}",
                "ci95_upper": f"{upper:.6f}", "improved_count": improved,
                "regressed_count": regressed, "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_seed": BOOTSTRAP_SEED,
            })
            route_summary["benchmarks"][benchmark] = {
                "success_count": sum(active_values), "task_count": len(active_values),
                "score": score, "delta": point, "ci95": [lower, upper],
                "improved_count": improved, "regressed_count": regressed,
            }
        summary_routes[cwe_id] = route_summary

    best_by_benchmark: dict[str, Any] = {}
    worst_by_benchmark: dict[str, Any] = {}
    for benchmark in labels:
        candidates = [(cwe_id, details["benchmarks"][benchmark])
                      for cwe_id, details in summary_routes.items()]
        best_cwe, best = max(candidates, key=lambda item: item[1]["delta"])
        worst_cwe, worst = min(candidates, key=lambda item: item[1]["delta"])
        best_by_benchmark[benchmark] = {"route_cwe_id": best_cwe, **best}
        worst_by_benchmark[benchmark] = {"route_cwe_id": worst_cwe, **worst}

    atomic_csv(RESULTS, result_rows)
    atomic_csv(INTERVALS, interval_rows)
    summary = {
        "schema_version": "reduced_routewise_bstar_utility_summary_v1",
        "status": "COMPLETE",
        "interpretation_boundary": (
            "Per-route active-coordinate sensitivity on general utility tasks without target-CWE labels. "
            "The nine conditions are not pooled into a deployable B* estimate."
        ),
        "protocol_sha256": sha256_path(PROTOCOL),
        "paired_b0_sha256": sha256_path(baseline_output()),
        "humaneval_return_manifest_sha256": sha256_path(RETURN_MANIFEST),
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
            "unit": "paired task", "interval": "equal-tailed Type-7 percentile 95%",
        },
        "routes": summary_routes,
        "best_observed_route_by_benchmark": best_by_benchmark,
        "worst_observed_route_by_benchmark": worst_by_benchmark,
        "artifacts": {
            "results_sha256": sha256_path(RESULTS),
            "intervals_sha256": sha256_path(INTERVALS),
        },
    }
    atomic_json(SUMMARY, summary)
    print(json.dumps({
        "status": "COMPLETE", "result_rows": len(result_rows),
        "interval_rows": len(interval_rows),
        "best_observed_route_by_benchmark": best_by_benchmark,
        "worst_observed_route_by_benchmark": worst_by_benchmark,
        "results_sha256": sha256_path(RESULTS),
        "intervals_sha256": sha256_path(INTERVALS),
        "summary_sha256": sha256_path(SUMMARY),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
