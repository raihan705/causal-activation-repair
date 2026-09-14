#!/usr/bin/env python3
"""Run the frozen 10,000-replicate paired Phase 16 CorrVRR bootstrap."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from phase16_analysis_common import (
    CONDITIONS, OUT, PRIMARY, baseline_sets, load_all, metric_values,
    percentile_type7, rounded, sha256_path, valid, write_csv,
)

REPLICATES = 10_000
BOOTSTRAP_SEED = 42


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    loaded = load_all()
    universe = loaded["universe"]
    condition_keys = [(method, seed) for method, seed, *_ in CONDITIONS]

    baseline_masks: dict[int, list[int]] = {}
    repair_masks: dict[tuple[str, int], list[int]] = {}
    for seed in (42, 43, 44):
        vulnerable, _ = baseline_sets(loaded, seed)
        baseline_masks[seed] = [int(prompt_id in vulnerable) for prompt_id in universe]
    for method, seed in condition_keys:
        item = loaded["data"][(method, seed)]
        generation = item["generation_map"]
        scan = item["scan_map"]
        vulnerable, _ = baseline_sets(loaded, seed)
        repair_masks[(method, seed)] = [int(prompt_id in vulnerable and valid(generation[prompt_id])
                                            and scan[prompt_id]["skipped"] is False
                                            and scan[prompt_id]["is_vulnerable"] is False)
                                         for prompt_id in universe]

    def sampled_rate(key: tuple[str, int], sampled: list[int]) -> float:
        seed = key[1]
        denominator = sum(baseline_masks[seed][index] for index in sampled)
        if denominator == 0:
            raise RuntimeError(f"Zero bootstrap denominator for seed {seed}")
        return sum(repair_masks[key][index] for index in sampled) / denominator

    point = {(method, seed): metric_values(loaded, method, seed)["corrvrr"] for method, seed in condition_keys}
    method_replicates: dict[str, list[float]] = {method: [] for method in PRIMARY}
    for method in ("B3-ungated", "B1-CWE", "RCI-1", "CAA-CWE", "B3-PG"):
        method_replicates[method] = []
    per_seed_replicates: dict[tuple[str, int], list[float]] = {key: [] for key in condition_keys}
    contrast_defs = [
        ("B* - B1", "MULTI_SEED_MEAN", "B1"),
        ("B* - B2-alpha20", "MULTI_SEED_MEAN", "B2-alpha20"),
        ("B* - B3-ungated", "COMMON_SEED_42", "B3-ungated"),
        ("B* - B1-CWE", "COMMON_SEED_42", "B1-CWE"),
        ("B* - RCI-1", "COMMON_SEED_42", "RCI-1"),
        ("B* - CAA-CWE", "COMMON_SEED_42", "CAA-CWE"),
    ]
    contrast_values = {name: [] for name, _, _ in contrast_defs}

    rng = random.Random(BOOTSTRAP_SEED)
    for _ in range(REPLICATES):
        sampled = [rng.randrange(len(universe)) for _ in universe]
        rates = {key: sampled_rate(key, sampled) for key in condition_keys}
        for key, value in rates.items():
            per_seed_replicates[key].append(value)
        means = {method: sum(rates[(method, seed)] for seed in (42, 43, 44)) / 3 for method in PRIMARY}
        for method in PRIMARY:
            method_replicates[method].append(means[method])
        for method in ("B3-ungated", "B1-CWE", "RCI-1", "CAA-CWE", "B3-PG"):
            method_replicates[method].append(rates[(method, 42)])
        contrast_values["B* - B1"].append(means["B*"] - means["B1"])
        contrast_values["B* - B2-alpha20"].append(means["B*"] - means["B2-alpha20"])
        for name, scope, comparator in contrast_defs[2:]:
            contrast_values[name].append(rates[("B*", 42)] - rates[(comparator, 42)])

    pairwise_rows = []
    for name, scope, comparator in contrast_defs:
        if scope == "MULTI_SEED_MEAN":
            point_estimate = sum(point[("B*", seed)] for seed in (42, 43, 44)) / 3 - sum(
                point[(comparator, seed)] for seed in (42, 43, 44)) / 3
        else:
            point_estimate = point[("B*", 42)] - point[(comparator, 42)]
        values = contrast_values[name]
        pairwise_rows.append({
            "contrast": name, "metric": "CorrVRR", "pairing_scope": scope,
            "status": "DESCRIPTIVE" if comparator == "B2-alpha20" else "COMPUTED",
            "point_estimate": rounded(point_estimate),
            "ci95_lower": rounded(percentile_type7(values, 0.025)),
            "ci95_upper": rounded(percentile_type7(values, 0.975)),
            "bootstrap_replicates": REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED,
            "resampling_universe_count": len(universe),
            "interpretation": "POSITIVE_FAVORS_BSTAR_NEGATIVE_FAVORS_COMPARATOR",
        })
    write_csv(OUT / "phase16_pairwise_intervals.csv", list(pairwise_rows[0]), pairwise_rows)

    method_intervals: list[dict[str, Any]] = []
    for method, values in method_replicates.items():
        seeds = (42, 43, 44) if method in PRIMARY else (42,)
        estimate = sum(point[(method, seed)] for seed in seeds) / len(seeds)
        method_intervals.append({
            "method": method, "seed_scope": "MEAN_42_43_44" if method in PRIMARY else "SINGLE_SEED_42",
            "provenance_status": "DESCRIPTIVE_ONLY" if method == "B2-alpha20" else "PRIMARY_OR_COMPLEMENTARY",
            "point_estimate": float(rounded(estimate)),
            "ci95_lower": float(rounded(percentile_type7(values, 0.025))),
            "ci95_upper": float(rounded(percentile_type7(values, 0.975))),
        })
    per_seed_intervals = []
    for key, values in per_seed_replicates.items():
        per_seed_intervals.append({
            "method": key[0], "seed": key[1], "point_estimate": float(rounded(point[key])),
            "ci95_lower": float(rounded(percentile_type7(values, 0.025))),
            "ci95_upper": float(rounded(percentile_type7(values, 0.975))),
        })
    manifest = {
        "schema_version": "phase16_bootstrap_manifest_public_v1", "status": "COMPLETE",
        "metric": "CorrVRR", "replicates": REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED,
        "rng": "Python random.Random MT19937", "sampling_unit": "paired prompt-level cluster record",
        "resampling_universe_source": "phase2_denominator_audit.json heldout.scanner_eligible_prompt_ids",
        "resampling_universe_count": len(universe),
        "interval": "equal-tailed 95% percentile, Type-7 linear interpolation",
        "seed_identifiers_resampled": False, "method_outcomes_resampled_independently": False,
        "method_intervals": method_intervals, "per_seed_intervals": per_seed_intervals,
        "pairwise_output": "revision/model1/phase16/outputs/phase16_pairwise_intervals.csv",
        "pairwise_output_sha256": sha256_path(OUT / "phase16_pairwise_intervals.csv"),
    }
    (OUT / "phase16_bootstrap_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "pairwise_rows": len(pairwise_rows),
                      "replicates": REPLICATES}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
