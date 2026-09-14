#!/usr/bin/env python3
"""Compute Phase 17 utility scores and paired task-level bootstrap intervals."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

from phase17_common import OUTPUTS, ROOT, atomic_csv, atomic_json, load_prepared_manifest, read_json, require, sha256_path


ACTIVE = OUTPUTS / "alwayson_utility_seed42_outputs.json"
HE_B0 = OUTPUTS / "humaneval_b0_task_results.json"
HE_ACTIVE = OUTPUTS / "humaneval_alwayson_task_results.json"
RESULTS = OUTPUTS / "phase17_utility_results.csv"
INTERVALS = OUTPUTS / "phase17_utility_intervals.csv"
BOOTSTRAP = OUTPUTS / "phase17_utility_bootstrap_manifest.json"
REPLICATES = 10_000
BOOTSTRAP_SEED = 42


def type7(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def ast_valid(text: str) -> bool:
    code = text.strip()
    if not code:
        return False
    if "```" in code:
        lines = code.split("\n")
        code_lines = []
        in_block = False
        for line in lines:
            if line.startswith("```"):
                in_block = not in_block
                continue
            if in_block:
                code_lines.append(line)
        code = "\n".join(code_lines) if code_lines else code
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def human_outcomes(path: Path, expected: list[str]) -> list[bool]:
    require(path.is_file(), f"Missing HumanEval task-level result: {path}")
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        rows = read_json(path)
    require(isinstance(rows, list), f"Malformed HumanEval task-level result: {path}")
    require(all(isinstance(row.get("passed"), bool) for row in rows),
            f"HumanEval passed field is not Boolean: {path}")
    mapping = {str(row.get("task_id", row.get("stable_id"))): row["passed"] for row in rows}
    require(set(mapping) == set(expected), f"HumanEval task population mismatch: {path}")
    return [mapping[task_id] for task_id in expected]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    manifest, preparation = load_prepared_manifest()
    required = [ACTIVE, HE_B0, HE_ACTIVE]
    if args.preflight_only:
        result = {"schema_version": "phase17_utility_analysis_preflight_v1",
                  "status": ("READY_FOR_ANALYSIS" if all(path.is_file() for path in required)
                             else "PASS_AWAITING_GENERATION_AND_HUMANEVAL_EVALUATION"),
                  "calculation_run": False, "required_inputs": [str(path.relative_to(ROOT)) for path in required],
                  "present": {path.name: path.is_file() for path in required},
                  "bootstrap": {"replicates": REPLICATES, "seed": BOOTSTRAP_SEED,
                                "sampling_unit": "paired task", "interval": "equal-tailed Type-7 percentile"}}
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    active = read_json(ACTIVE)
    require(isinstance(active, list) and len(active) == 1716, "Active utility output count mismatch")
    active_map = {row["stable_id"]: row for row in active}
    require(len(active_map) == 1716, "Active utility IDs are not unique")
    b0_he = read_json(ROOT / "outputs/phase10/utility_baseline_humaneval.json")
    b0_bcb = read_json(ROOT / "outputs/phase10/utility_baseline_bigcodebench.json")
    b0_mmlu = read_json(ROOT / "outputs/phase10/utility_baseline_mmlu.json")

    he_ids = [row["task_id"] for row in b0_he]
    outcomes: dict[str, tuple[list[str], list[bool], list[bool], str]] = {
        "HumanEval": (he_ids, human_outcomes(HE_B0, he_ids), human_outcomes(HE_ACTIVE, he_ids), "pass_fail"),
        "BigCodeBench": ([row["task_id"] for row in b0_bcb],
                         [ast_valid(row["completion"]) for row in b0_bcb],
                         [ast_valid(active_map[row["task_id"]]["generated_text"]) for row in b0_bcb],
                         "syntax_pass_proxy"),
    }
    mmlu_records = manifest["populations"]["mmlu"]["records"]
    mmlu_ids = [row["stable_id"] for row in mmlu_records]
    outcomes["MMLU"] = (mmlu_ids, [bool(row["correct"]) for row in b0_mmlu],
                         [bool(active_map[stable_id]["correct"]) for stable_id in mmlu_ids], "accuracy")

    result_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    rng = random.Random(BOOTSTRAP_SEED)
    bootstrap_summary: dict[str, Any] = {}
    for benchmark, (ids, b0, always, metric) in outcomes.items():
        require(len(ids) == len(b0) == len(always), f"{benchmark} outcome alignment mismatch")
        b0_score = sum(b0) / len(b0)
        active_score = sum(always) / len(always)
        for condition, tier, score, role in (
            ("B0", "NO_INTERVENTION_BASELINE", b0_score, "REFERENCE"),
            ("GATED_BSTAR_INACTIVE_PAIRED", "ORACLE_CWE_INACTIVE", b0_score, "CORRECTED_EXACT_B0_REFERENCE"),
            ("B*-AlwaysOn-L19", "METADATA_FREE", active_score, "ACTIVE_STRESS"),
        ):
            result_rows.append({"benchmark": benchmark, "condition": condition,
                                "information_tier": tier, "scientific_role": role,
                                "task_count": len(ids), "metric": metric,
                                "success_count": int(round(score * len(ids))),
                                "score": f"{score:.6f}"})
        interval_rows.append({"benchmark": benchmark, "contrast": "GATED_BSTAR_INACTIVE_PAIRED - B0",
                              "metric": metric, "task_count": len(ids), "point_estimate": "0.000000",
                              "ci95_lower": "0.000000", "ci95_upper": "0.000000",
                              "bootstrap_replicates": REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED,
                              "status": "EXACT_IDENTITY_BY_CONSTRUCTION"})
        replicates = []
        for _ in range(REPLICATES):
            sampled = [rng.randrange(len(ids)) for _ in ids]
            replicates.append(sum(int(always[i]) - int(b0[i]) for i in sampled) / len(sampled))
        point = active_score - b0_score
        lower, upper = type7(replicates, 0.025), type7(replicates, 0.975)
        interval_rows.append({"benchmark": benchmark, "contrast": "B*-AlwaysOn-L19 - B0",
                              "metric": metric, "task_count": len(ids), "point_estimate": f"{point:.6f}",
                              "ci95_lower": f"{lower:.6f}", "ci95_upper": f"{upper:.6f}",
                              "bootstrap_replicates": REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED,
                              "status": "COMPUTED_PAIRED_TASK_BOOTSTRAP"})
        bootstrap_summary[benchmark] = {"task_ids_sha256": __import__("hashlib").sha256(
            json.dumps(ids, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest(),
            "point_estimate": point, "ci95": [lower, upper]}

    atomic_csv(RESULTS, list(result_rows[0]), result_rows)
    atomic_csv(INTERVALS, list(interval_rows[0]), interval_rows)
    atomic_json(BOOTSTRAP, {"schema_version": "phase17_utility_bootstrap_manifest_v1",
                            "status": "COMPLETE", "replicates": REPLICATES,
                            "seed": BOOTSTRAP_SEED, "sampling_unit": "paired task",
                            "interval": "equal-tailed 95% percentile Type-7",
                            "input_manifest_sha256": preparation["input_manifest_sha256"],
                            "benchmarks": bootstrap_summary,
                            "utility_results_sha256": sha256_path(RESULTS),
                            "utility_intervals_sha256": sha256_path(INTERVALS)})
    print(json.dumps({"status": "COMPLETE", "result_rows": len(result_rows),
                      "interval_rows": len(interval_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
