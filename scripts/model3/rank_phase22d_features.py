#!/usr/bin/env python
"""Run the prospectively frozen Model3 statistical feature ranking only."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "revision/model3/phase22/outputs"
ACT = OUT / "activations"
PARTITION = OUT / "phase22d_feature_partition_manifest.json"
PROTOCOL = OUT / "phase22d_activation_ranking_protocol.json"
RANKINGS = OUT / "phase22d_statistical_feature_rankings.json"
SUMMARY = OUT / "phase22d_statistical_feature_rankings_summary.csv"
CANDIDATES = OUT / "phase22d_causal_candidate_manifest.json"
CHECKPOINT = OUT / "phase22d_feature_discovery_checkpoint.json"
TARGETS = ("CWE-120", "CWE-327", "CWE-89")
LAYERS = (7, 15, 23)
D_SAE = 131072
TOP_K = 200
KS_SUBSAMPLE = 500
RUNNER = Path(__file__).resolve()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def histogram_distance(unsafe: torch.Tensor, safe: torch.Tensor, bins: int = 20) -> np.ndarray:
    a, b = unsafe.numpy(), safe.numpy()
    distance = np.zeros(a.shape[1], dtype=np.float32)
    for feature in range(a.shape[1]):
        values = np.concatenate([a[:, feature], b[:, feature]])
        low, high = values.min(), values.max()
        if high - low < 1e-8:
            continue
        edges = np.linspace(low, high, bins + 1)
        unsafe_hist, _ = np.histogram(a[:, feature], bins=edges, density=True)
        safe_hist, _ = np.histogram(b[:, feature], bins=edges, density=True)
        distance[feature] = 1.0 - np.sum(np.minimum(unsafe_hist, safe_hist)) * (edges[1] - edges[0])
    return distance


def rank_cell(target: str, layer: int, unsafe: torch.Tensor, safe: torch.Tensor) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    require(tuple(unsafe.shape)[1:] == (D_SAE,) and tuple(safe.shape)[1:] == (D_SAE,), "ranking tensor width mismatch")
    histogram = histogram_distance(unsafe, safe)
    unsafe_mean_t, safe_mean_t = unsafe.mean(0), safe.mean(0)
    unsafe_mean, safe_mean = unsafe_mean_t.numpy(), safe_mean_t.numpy()
    difference = (unsafe_mean_t - safe_mean_t).abs().numpy()
    preliminary = 0.4 * histogram + 0.4 * difference
    top_ks = np.argsort(preliminary)[::-1][:KS_SUBSAMPLE]
    ks_full = np.zeros(D_SAE, dtype=np.float32)
    unsafe_np, safe_np = unsafe.numpy(), safe.numpy()
    for feature in top_ks:
        ks_full[feature] = ks_2samp(unsafe_np[:, feature], safe_np[:, feature]).statistic
    scores = 0.4 * histogram + 0.4 * difference + 0.2 * ks_full
    top_features = np.argsort(scores)[::-1][:TOP_K]
    unsafe_frequency = (unsafe > 0).float().mean(0).numpy()
    safe_frequency = (safe > 0).float().mean(0).numpy()
    rankings, eligible = [], []
    for rank, feature in enumerate(top_features, 1):
        values = [histogram[feature], unsafe_mean[feature], safe_mean[feature], difference[feature], ks_full[feature], scores[feature], unsafe_frequency[feature], safe_frequency[feature]]
        reasons = []
        if not all(math.isfinite(float(x)) for x in values):
            reasons.append("NON_FINITE_STATISTIC")
        if unsafe_frequency[feature] <= 0:
            reasons.append("NEVER_ACTIVE_IN_UNSAFE_ARM")
        record = {
            "cwe_id": target, "layer": layer, "feature_id": int(feature), "rank": rank,
            "histogram_distance": float(histogram[feature]), "unsafe_mean": float(unsafe_mean[feature]),
            "safe_mean": float(safe_mean[feature]), "absolute_mean_difference": float(difference[feature]),
            "ks_statistic": float(ks_full[feature]), "composite_score": float(scores[feature]),
            "unsafe_activation_frequency": float(unsafe_frequency[feature]),
            "safe_activation_frequency": float(safe_frequency[feature]),
            "unsafe_n": int(unsafe.shape[0]), "safe_n": int(safe.shape[0]),
            "candidate_eligibility": "ELIGIBLE" if not reasons else "INELIGIBLE",
            "ineligibility_reasons": reasons,
        }
        rankings.append(record)
        if not reasons:
            eligible.append(record)
    return rankings, eligible


def execute(partition: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    rankings: dict[str, Any] = {target: {} for target in TARGETS}
    candidate_cells: dict[str, Any] = {target: {} for target in TARGETS}
    summary = []
    for layer in LAYERS:
        manifest_path = OUT / f"phase22d_layer_{layer}_latent_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        latent_path = ACT / f"phase22d_layer_{layer}_latents.pt"
        require(manifest["status"] == "COMPLETE" and manifest["latent_sha256"] == file_hash(latent_path), f"latent manifest/hash failure L{layer}")
        payload = torch.load(latent_path, map_location="cpu", weights_only=False)
        latents = payload["latents"].float().contiguous()
        ids = [int(x["prompt_id"]) for x in payload["metadata"]]
        require(tuple(latents.shape) == (144, D_SAE), f"latent shape mismatch L{layer}")
        require(ids == partition["ranking_prompt_union_ids_source_order"], f"latent ID order mismatch L{layer}")
        index = {pid: i for i, pid in enumerate(ids)}
        for target in TARGETS:
            spec = partition["partitions"][target]
            unsafe_ids = spec["unsafe_ids_source_order"]
            safe_ids = spec["selected_safe_ids_sampling_order"]
            unsafe = latents[[index[pid] for pid in unsafe_ids]].contiguous()
            safe = latents[[index[pid] for pid in safe_ids]].contiguous()
            cell, eligible = rank_cell(target, layer, unsafe, safe)
            rankings[target][f"L{layer}"] = cell
            chosen = eligible[:10]
            candidate_cells[target][f"L{layer}"] = {
                "status": "CAUSAL_VALIDATION_CANDIDATES_FROZEN" if chosen else "NO_ELIGIBLE_CANDIDATE",
                "eligible_top200_count": len(eligible), "candidate_count": len(chosen), "candidates": chosen,
            }
            summary.append({
                "cwe_id": target, "layer": layer, "unsafe_n": len(unsafe_ids),
                "selected_safe_n": len(safe_ids), "features_considered": D_SAE,
                "ks_candidate_count": KS_SUBSAMPLE, "top200_count": len(cell),
                "eligible_top200_count": len(eligible), "causal_candidate_count": len(chosen),
                "top_feature_id": cell[0]["feature_id"], "top_feature_score": cell[0]["composite_score"],
            })
    return rankings, summary, candidate_cells


def main() -> None:
    for path in (RANKINGS, SUMMARY, CANDIDATES, CHECKPOINT):
        require(not path.exists(), f"feature discovery output already exists: {path.name}")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    partition = json.loads(PARTITION.read_text(encoding="utf-8"))
    require(protocol["scripts"]["ranking_sha256"] == file_hash(RUNNER), "ranking runner drift")
    require(protocol["partition_sha256"] == file_hash(PARTITION), "partition hash drift")
    first = execute(partition)
    replay = execute(partition)
    require(canonical_hash(first) == canonical_hash(replay), "deterministic ranking replay mismatch")
    rankings, summary, cells = first
    atomic_json(RANKINGS, {
        "schema_version": "phase22d_model3_statistical_feature_rankings_v1", "status": "COMPLETE",
        "protocol_sha256": file_hash(PROTOCOL), "partition_sha256": file_hash(PARTITION),
        "features_considered_per_cell": D_SAE, "histogram_bins": 20,
        "weights": {"histogram": 0.4, "absolute_mean_difference": 0.4, "ks": 0.2},
        "ks_preselection_count": KS_SUBSAMPLE, "non_preselected_ks_default": 0.0,
        "ordering": "numpy.argsort(score)[::-1]", "top_k": TOP_K,
        "absolute_cross_model_activation_thresholds_used": False, "cells": rankings,
        "causal_intervention_run": False, "alpha_selected": False, "scanner_run": False, "heldout_used": False,
    })
    atomic_csv(SUMMARY, summary)
    atomic_json(CANDIDATES, {
        "schema_version": "phase22d_model3_causal_candidate_manifest_v1", "status": "FROZEN_BEFORE_CAUSAL_INTERVENTION",
        "protocol_sha256": file_hash(PROTOCOL), "partition_sha256": file_hash(PARTITION),
        "ranking_sha256": file_hash(RANKINGS),
        "selection_rule": "first 10 finite top-200 features active in at least one target-unsafe record, in frozen statistical rank order",
        "candidate_meaning": "CAUSAL_VALIDATION_CANDIDATES_ONLY_NOT_VALIDATED_SECURITY_FEATURES",
        "cells": cells, "causal_intervention_run": False, "alpha_selected": False,
        "steered_generation_run": False, "scanner_run": False, "heldout_used": False,
    })
    checkpoint = {
        "schema_version": "phase22d_model3_feature_discovery_checkpoint_v1", "status": "PASS",
        "state": "FEATURE_RANKING_COMPLETE", "protocol_sha256": file_hash(PROTOCOL),
        "partition_sha256": file_hash(PARTITION), "ranking_sha256": file_hash(RANKINGS),
        "ranking_summary_sha256": file_hash(SUMMARY), "causal_candidate_manifest_sha256": file_hash(CANDIDATES),
        "determinism_validation": "PASS", "cells_completed": 9, "features_considered_per_cell": D_SAE,
        "top200_count_per_cell": TOP_K,
        "no_candidate_cells": [{"cwe_id": target, "layer": layer} for target in TARGETS for layer in LAYERS if cells[target][f"L{layer}"]["candidate_count"] == 0],
        "next_execution_authorized": False,
        "next_stage": "causal validation under a separately frozen protocol",
        "causal_intervention_run": False, "alpha_selected_or_transferred": False,
        "steered_generation_run": False, "scanner_run": False, "heldout_used": False,
    }
    atomic_json(CHECKPOINT, checkpoint)
    print(json.dumps({"status": "PASS", "summary": summary, "hashes": {"rankings": file_hash(RANKINGS), "candidates": file_hash(CANDIDATES), "checkpoint": file_hash(CHECKPOINT)}}, indent=2))


if __name__ == "__main__":
    main()
