#!/usr/bin/env python3
"""Compute route-wise utility estimates and paired bootstrap intervals."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import random
from pathlib import Path
from typing import Any

from routewise_utility_common import (
    EXPECTED_COUNTS, OUTPUTS, PROTOCOL, atomic_json, baseline_output,
    baseline_run_manifest, load_protocol, read_json, require, route_output,
    route_run_manifest, route_slug, sha256_path,
)


RESULTS = OUTPUTS / "routewise_utility_results.csv"
INTERVALS = OUTPUTS / "routewise_utility_intervals.csv"
SUMMARY = OUTPUTS / "routewise_utility_summary.json"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 42


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    require(bool(rows), f"No rows available for {path.name}")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validated_output(path: Path, manifest_path: Path, expected: int) -> list[dict[str, Any]]:
    require(path.is_file() and manifest_path.is_file(), f"Missing generation artifact: {path.name}")
    manifest = read_json(manifest_path)
    require(manifest.get("status") == "COMPLETE", f"{manifest_path.name} is not COMPLETE")
    require(manifest.get("output_sha256") == sha256_path(path), f"{path.name} hash mismatch")
    rows = read_json(path)
    require(isinstance(rows, list) and len(rows) == expected, f"{path.name} count mismatch")
    return rows


def ast_valid(text: str) -> bool:
    code = text.strip()
    if not code:
        return False
    if "```" in code:
        inside = False
        extracted: list[str] = []
        for line in code.splitlines():
            if line.startswith("```"):
                inside = not inside
            elif inside:
                extracted.append(line)
        if extracted:
            code = "\n".join(extracted)
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def task_results(path: Path, expected_ids: list[str], output_rows: list[dict[str, Any]]) -> list[bool]:
    require(path.is_file(), f"Missing official HumanEval results: {path.name}")
    rows = read_json(path)
    require(isinstance(rows, list) and len(rows) == len(expected_ids), f"{path.name} count mismatch")
    require([str(row["task_id"]) for row in rows] == expected_ids, f"{path.name} order mismatch")
    require(all(isinstance(row.get("passed"), bool) for row in rows), f"{path.name} passed field malformed")
    expected_hashes = {
        row["stable_id"]: __import__("hashlib").sha256(row["generated_text"].encode("utf-8")).hexdigest()
        for row in output_rows if row["benchmark"] == "humaneval"
    }
    require(all(row["completion_sha256"] == expected_hashes[str(row["task_id"])] for row in rows),
            f"{path.name} completion hash mismatch")
    return [bool(row["passed"]) for row in rows]


def type7(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def bootstrap_delta(b0: list[bool], active: list[bool]) -> tuple[float, float, float]:
    require(len(b0) == len(active), "Paired outcome length mismatch")
    point = sum(active) / len(active) - sum(b0) / len(b0)
    rng = random.Random(BOOTSTRAP_SEED)
    estimates: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        indices = [rng.randrange(len(b0)) for _ in b0]
        estimates.append(sum(int(active[i]) - int(b0[i]) for i in indices) / len(indices))
    return point, type7(estimates, 0.025), type7(estimates, 0.975)


def benchmark_outcomes(rows: list[dict[str, Any]], benchmark: str,
                       human_path: Path | None = None) -> tuple[list[str], list[bool], str]:
    selected = [row for row in rows if row["benchmark"] == benchmark]
    ids = [str(row["stable_id"]) for row in selected]
    require(len(selected) == EXPECTED_COUNTS[benchmark], f"{benchmark} count mismatch")
    if benchmark == "humaneval":
        require(human_path is not None, "HumanEval task result path is absent")
        return ids, task_results(human_path, ids, rows), "pass@1"
    if benchmark == "bigcodebench":
        return ids, [ast_valid(row["generated_text"]) for row in selected], "syntax_pass_proxy"
    return ids, [bool(row["correct"]) for row in selected], "accuracy"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    protocol = load_protocol()
    total = int(protocol["population"]["total_tasks"])
    required = [baseline_output(), baseline_run_manifest(),
                OUTPUTS / "humaneval_b0_task_results.json"]
    for cwe_id, route in protocol["routes"].items():
        required.extend([
            route_output(cwe_id, route), route_run_manifest(cwe_id, route),
            OUTPUTS / f"humaneval_{route_slug(cwe_id, route)}_task_results.json",
        ])
    if args.preflight_only:
        result = {
            "status": "READY_FOR_ANALYSIS" if all(path.is_file() for path in required)
                      else "AWAITING_GENERATION_OR_HUMANEVAL_RESULTS",
            "calculation_run": False,
            "required_file_count": len(required),
            "present_file_count": sum(path.is_file() for path in required),
            "missing": [path.name for path in required if not path.is_file()],
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    b0 = validated_output(baseline_output(), baseline_run_manifest(), total)
    b0_human = OUTPUTS / "humaneval_b0_task_results.json"
    tolerances = protocol["evaluation"]["tolerances"]
    labels = {"humaneval": "HumanEval", "bigcodebench": "BigCodeBench", "mmlu": "MMLU"}
    b0_outcomes: dict[str, tuple[list[str], list[bool], str]] = {}
    for benchmark in labels:
        b0_outcomes[benchmark] = benchmark_outcomes(
            b0, benchmark, b0_human if benchmark == "humaneval" else None
        )

    result_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    route_summaries: dict[str, Any] = {}
    for benchmark, display in labels.items():
        ids, outcomes, metric = b0_outcomes[benchmark]
        result_rows.append({
            "benchmark": display, "condition": "B0", "route_cwe_id": "",
            "layer": "", "feature": "", "alpha": "", "task_count": len(ids),
            "metric": metric, "success_count": sum(outcomes),
            "score": f"{sum(outcomes) / len(outcomes):.6f}", "scientific_role": "REFERENCE",
        })

    for cwe_id, route in protocol["routes"].items():
        active = validated_output(route_output(cwe_id, route), route_run_manifest(cwe_id, route), total)
        route_summary: dict[str, Any] = {"layer": int(route["layer"]),
                                         "feature": int(route["feature"]), "benchmarks": {}}
        for benchmark, display in labels.items():
            b0_ids, b0_values, metric = b0_outcomes[benchmark]
            human_path = (OUTPUTS / f"humaneval_{route_slug(cwe_id, route)}_task_results.json"
                          if benchmark == "humaneval" else None)
            active_ids, active_values, active_metric = benchmark_outcomes(active, benchmark, human_path)
            require(active_ids == b0_ids and active_metric == metric, f"{cwe_id}/{benchmark} pairing mismatch")
            point, lower, upper = bootstrap_delta(b0_values, active_values)
            tolerance = float(tolerances[benchmark])
            if lower >= -tolerance:
                gate = "PRESERVED_CONSERVATIVE_CI"
            elif point >= -tolerance:
                gate = "POINT_WITHIN_TOLERANCE_CI_CROSSES"
            else:
                gate = "DEGRADATION_EXCEEDS_TOLERANCE"
            improved = sum((not left) and right for left, right in zip(b0_values, active_values))
            regressed = sum(left and (not right) for left, right in zip(b0_values, active_values))
            score = sum(active_values) / len(active_values)
            result_rows.append({
                "benchmark": display, "condition": "B*_SINGLE_ROUTE_ACTIVE",
                "route_cwe_id": cwe_id, "layer": int(route["layer"]),
                "feature": int(route["feature"]), "alpha": 40,
                "task_count": len(active_ids), "metric": metric,
                "success_count": sum(active_values), "score": f"{score:.6f}",
                "scientific_role": "ACTIVE_COORDINATE_SENSITIVITY",
            })
            interval_rows.append({
                "benchmark": display, "contrast": f"{cwe_id} active route - paired B0",
                "route_cwe_id": cwe_id, "layer": int(route["layer"]),
                "feature": int(route["feature"]), "alpha": 40,
                "task_count": len(active_ids), "metric": metric,
                "point_estimate": f"{point:.6f}", "ci95_lower": f"{lower:.6f}",
                "ci95_upper": f"{upper:.6f}", "tolerance": f"{tolerance:.6f}",
                "improved_count": improved, "regressed_count": regressed,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_seed": BOOTSTRAP_SEED, "gate": gate,
            })
            route_summary["benchmarks"][benchmark] = {
                "score": score, "delta": point, "ci95": [lower, upper],
                "tolerance": tolerance, "gate": gate,
                "improved_count": improved, "regressed_count": regressed,
            }
        route_summaries[cwe_id] = route_summary

    worst_by_benchmark: dict[str, Any] = {}
    for benchmark in labels:
        entries = [(cwe, details["benchmarks"][benchmark]) for cwe, details in route_summaries.items()]
        cwe_id, detail = min(entries, key=lambda item: item[1]["delta"])
        worst_by_benchmark[benchmark] = {"route_cwe_id": cwe_id, **detail}

    atomic_csv(RESULTS, result_rows)
    atomic_csv(INTERVALS, interval_rows)
    summary = {
        "schema_version": "routewise_bstar_utility_summary_v1",
        "status": "COMPLETE",
        "interpretation": (
            "Per-route active utility sensitivity for the nine frozen B* coordinates. "
            "Results are not pooled as a deployable B* routing effect."
        ),
        "protocol_sha256": sha256_path(PROTOCOL),
        "paired_b0_sha256": sha256_path(baseline_output()),
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
                      "unit": "paired task", "interval": "equal-tailed Type-7 percentile 95%"},
        "routes": route_summaries,
        "worst_observed_route_by_benchmark": worst_by_benchmark,
        "artifacts": {"results_sha256": sha256_path(RESULTS),
                      "intervals_sha256": sha256_path(INTERVALS)},
    }
    atomic_json(SUMMARY, summary)
    print(json.dumps({"status": "COMPLETE", "result_rows": len(result_rows),
                      "interval_rows": len(interval_rows),
                      "worst_observed_route_by_benchmark": worst_by_benchmark},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

