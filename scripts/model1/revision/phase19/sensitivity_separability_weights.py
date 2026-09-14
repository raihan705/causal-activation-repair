#!/usr/bin/env python3
"""Recompute the frozen W1-W5 separability rankings from saved Phase 4 latents."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ks_2samp

from phase19_common import FROZEN_BSTAR, OUT, ROOT, atomic_csv, atomic_json, hash_record, read_json, require


LAYERS = (16, 19, 23)
CWES = ("CWE-120", "CWE-327", "CWE-89", "CWE-338", "CWE-79", "CWE-125", "CWE-787", "CWE-190", "CWE-476")
WEIGHTS = {
    "W1": (0.4, 0.4, 0.2),
    "W2": (1 / 3, 1 / 3, 1 / 3),
    "W3": (0.5, 0.3, 0.2),
    "W4": (0.3, 0.5, 0.2),
    "W5": (0.4, 0.3, 0.3),
}
TOP_K = 200
KS_SUBSAMPLE = 500


def load_latents(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu")
    if isinstance(value, dict):
        value = value["latents"]
    return value.float()


def histogram_distance(a: torch.Tensor, b: torch.Tensor, bins: int = 20) -> np.ndarray:
    aa = a.numpy()
    bb = b.numpy()
    result = np.zeros(aa.shape[1], dtype=np.float32)
    for feature in range(aa.shape[1]):
        values = np.concatenate([aa[:, feature], bb[:, feature]])
        lo, hi = values.min(), values.max()
        if hi - lo < 1e-8:
            continue
        edges = np.linspace(lo, hi, bins + 1)
        ha, _ = np.histogram(aa[:, feature], bins=edges, density=True)
        hb, _ = np.histogram(bb[:, feature], bins=edges, density=True)
        result[feature] = 1.0 - np.sum(np.minimum(ha, hb)) * (edges[1] - edges[0])
    return result


def component_arrays(layer: int, cwe: str):
    number = cwe.split("-")[1]
    unsafe_path = ROOT / f"outputs/phase4/unsafe_latents_layer_{layer}_cwe{number}.pt"
    safe_path = ROOT / f"outputs/phase4/safe_latents_layer_{layer}_cwe{number}.pt"
    unsafe = load_latents(unsafe_path)
    safe = load_latents(safe_path)
    require(unsafe.shape[0] > 0, f"no unsafe latents for {cwe} layer {layer}")
    original_safe_count = int(safe.shape[0])
    if safe.shape[0] > 5 * unsafe.shape[0]:
        indices = torch.randperm(safe.shape[0], generator=torch.Generator().manual_seed(42))[: 5 * unsafe.shape[0]]
        safe = safe[indices]
    hd = histogram_distance(unsafe, safe)
    dmu = (unsafe.mean(dim=0) - safe.mean(dim=0)).abs().numpy()
    return unsafe.numpy(), safe.numpy(), hd, dmu, unsafe_path, safe_path, original_safe_count


def jaccard(a, b) -> float:
    aa, bb = set(a), set(b)
    return len(aa & bb) / len(aa | bb)


def main() -> None:
    bstar_config_path = ROOT / "configs/bstar_config.json"
    bstar = read_json(bstar_config_path)
    require(float(bstar["alpha"]) == 40.0, "frozen B* alpha changed")
    require({cwe: (int(value["layer"]), int(value["feature"])) for cwe, value in bstar["feature_map"].items()} == FROZEN_BSTAR, "frozen B* map changed")

    script_path = ROOT / "phases/phase4/rank_features_statistical_multilayer.py"
    script_text = script_path.read_text(encoding="utf-8")
    require("ALPHA       = 0.4" in script_text and "BETA        = 0.4" in script_text and "GAMMA       = 0.2" in script_text, "submitted weights not found in ranking script")
    require("TOP_K       = 200" in script_text and "KS_SUBSAMPLE = 500" in script_text, "submitted ranking procedure changed")

    sensitivity_rows = []
    overlap_rows = []
    input_records = [hash_record(script_path), hash_record(bstar_config_path)]
    w1_match_count = 0
    for layer in LAYERS:
        submitted_rankings_path = ROOT / f"outputs/phase4/statistical_feature_rankings_layer_{layer}.json"
        candidate_path = ROOT / f"outputs/phase4/pruned_feature_candidates_layer_{layer}.json"
        submitted = read_json(submitted_rankings_path)
        candidates = read_json(candidate_path)
        input_records.extend([hash_record(submitted_rankings_path), hash_record(candidate_path)])
        for cwe in CWES:
            unsafe, safe, hd, dmu, unsafe_path, safe_path, original_safe_count = component_arrays(layer, cwe)
            input_records.extend([hash_record(unsafe_path), hash_record(safe_path)])

            partial_top = {}
            union = set()
            for name, (wh, wm, _wk) in WEIGHTS.items():
                indices = np.argsort(wh * hd + wm * dmu)[::-1][:KS_SUBSAMPLE]
                partial_top[name] = indices
                union.update(int(value) for value in indices)
            ks_values = {feature: float(ks_2samp(unsafe[:, feature], safe[:, feature]).statistic) for feature in sorted(union)}

            rankings = {}
            scores = {}
            for name, (wh, wm, wk) in WEIGHTS.items():
                ks = np.zeros(unsafe.shape[1], dtype=np.float32)
                indices = partial_top[name]
                ks[indices] = np.asarray([ks_values[int(feature)] for feature in indices], dtype=np.float32)
                score = wh * hd + wm * dmu + wk * ks
                order = np.argsort(score)[::-1]
                rankings[name] = [int(value) for value in order[:TOP_K]]
                scores[name] = score

            submitted_ids = [int(row["feature_id"]) for row in submitted[cwe]]
            require(rankings["W1"] == submitted_ids, f"W1 does not reproduce submitted ranking for {cwe} layer {layer}")
            w1_match_count += 1
            candidate_ids = {int(row["feature_id"]) for row in candidates[cwe]}
            frozen = FROZEN_BSTAR.get(cwe)
            frozen_feature = frozen[1] if frozen and frozen[0] == layer else None
            w1_frozen_rank = rankings["W1"].index(frozen_feature) + 1 if frozen_feature in rankings["W1"] else None

            for name, weights in WEIGHTS.items():
                top10 = rankings[name][:10]
                top200 = rankings[name]
                frozen_rank = top200.index(frozen_feature) + 1 if frozen_feature in top200 else None
                sensitivity_rows.append({
                    "weight_id": name,
                    "histogram_weight": format(weights[0], ".12g"),
                    "mean_difference_weight": format(weights[1], ".12g"),
                    "ks_weight": format(weights[2], ".12g"),
                    "cwe": cwe,
                    "layer": layer,
                    "n_unsafe": unsafe.shape[0],
                    "n_safe_used": safe.shape[0],
                    "n_safe_before_balance": original_safe_count,
                    "top1_feature": top200[0],
                    "top1_score": format(float(scores[name][top200[0]]), ".12g"),
                    "top10_features": json.dumps(top10, separators=(",", ":")),
                    "top200_features": json.dumps(top200, separators=(",", ":")),
                    "frozen_bstar_feature": "" if frozen_feature is None else frozen_feature,
                    "frozen_bstar_rank": "" if frozen_rank is None else frozen_rank,
                    "frozen_bstar_rank_change_from_w1": "" if frozen_rank is None or w1_frozen_rank is None else frozen_rank - w1_frozen_rank,
                    "frozen_bstar_top10": "" if frozen_feature is None else str(frozen_feature in top10).lower(),
                    "frozen_bstar_top200": "" if frozen_feature is None else str(frozen_feature in top200).lower(),
                    "frozen_bstar_in_saved_candidate_set": "" if frozen_feature is None else str(frozen_feature in candidate_ids).lower(),
                    "diagnostic_only": "true",
                })
                overlap_rows.append({
                    "weight_id": name,
                    "reference_weight_id": "W1",
                    "cwe": cwe,
                    "layer": layer,
                    "top10_jaccard": format(jaccard(top10, rankings["W1"][:10]), ".12g"),
                    "top200_jaccard": format(jaccard(top200, rankings["W1"]), ".12g"),
                    "top10_intersection_count": len(set(top10) & set(rankings["W1"][:10])),
                    "top200_intersection_count": len(set(top200) & set(rankings["W1"])),
                })

    sensitivity_path = OUT / "separability_weight_sensitivity.csv"
    overlap_path = OUT / "separability_rank_overlap.csv"
    atomic_csv(sensitivity_path, list(sensitivity_rows[0]), sensitivity_rows)
    atomic_csv(overlap_path, list(overlap_rows[0]), overlap_rows)

    # Deduplicate identical source paths while retaining deterministic order.
    unique_inputs = {}
    for record in input_records:
        unique_inputs[record["path"]] = record
    provenance = {
        "schema_version": "phase19_separability_weight_provenance_v1",
        "status": "PASS",
        "reference_components": ["histogram_distance", "absolute_mean_difference", "ks_statistic"],
        "reference_weights": {"histogram": 0.4, "absolute_mean_difference": 0.4, "ks": 0.2},
        "provenance_classification": "FIXED_BEFORE_DOWNSTREAM_SELECTION",
        "explicit_tuning_evidence": "NOT_FOUND",
        "evidence": {
            "ranking_script": hash_record(script_path),
            "chronology": "Ranking outputs precede pruning, causal validation, and intervention-library construction in preserved repository timestamps.",
            "procedure": "The reference script fixes W1 constants, top-200, and top-500 KS candidate computation before executing rankings.",
        },
        "weight_grid": {name: {"histogram": values[0], "absolute_mean_difference": values[1], "ks": values[2]} for name, values in WEIGHTS.items()},
        "w1_reference_ranking_matches": w1_match_count,
        "w1_expected_ranking_matches": len(LAYERS) * len(CWES),
        "input_artifacts": unique_inputs,
        "outputs": {"sensitivity": hash_record(sensitivity_path), "overlap": hash_record(overlap_path)},
        "diagnostic_only": True,
        "bstar_modified": False,
        "activations_regenerated": False,
        "causal_validation_rerun": False,
        "scanner_run": False,
    }
    atomic_json(OUT / "separability_weight_provenance.json", provenance)
    print(json.dumps({
        "status": "COMPLETE",
        "w1_exact_matches": w1_match_count,
        "sensitivity_rows": len(sensitivity_rows),
        "overlap_rows": len(overlap_rows),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
