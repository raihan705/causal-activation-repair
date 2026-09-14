#!/usr/bin/env python
"""Run frozen Model2 statistical feature ranking and candidate pruning only."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "revision/model2/phase21/outputs"
ACT = OUT / "activations"
CELLS = ACT / "feature_cells"
PARTITION = OUT / "model2_feature_partition_manifest.json"
LATENT_MANIFEST = OUT / "model2_feature_latent_manifest.csv"
LATENT_RUN = OUT / "model2_feature_latent_extraction_run.json"
RANKINGS = OUT / "model2_statistical_feature_rankings.json"
RANKING_SUMMARY = OUT / "model2_statistical_feature_rankings_summary.csv"
PRUNED = OUT / "model2_pruned_candidates.json"
CAUSAL = OUT / "model2_causal_candidate_manifest.json"
SAFE_UNSAFE_SUMMARY = OUT / "model2_safe_unsafe_summary.csv"
CHECKPOINT = OUT / "model2_feature_discovery_checkpoint.json"
TARGETS = ("CWE-120", "CWE-327", "CWE-89")
LAYERS = (9, 20, 31)
PARTITION_HASH = "4aefb5aa35ee7845c556bac03a4e64a29d3b41422824a18767e36a5800e710c2"
TOP_K = 200
KS_SUBSAMPLE = 500
ACTIVE_THRESHOLD = 0.01
FREQ_MIN = 0.02
FREQ_MAX = 0.95
SCORE_MIN = 0.15
MAG_MAX = 1e4


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def atomic_json(path: Path, value: object) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def atomic_torch(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def histogram_distance(unsafe: torch.Tensor, safe: torch.Tensor, bins: int = 20) -> np.ndarray:
    """Byte-semantic copy of the executed Model1 pooled min/max histogram implementation."""
    a = unsafe.numpy()
    b = safe.numpy()
    dimensions = a.shape[1]
    distance = np.zeros(dimensions, dtype=np.float32)
    for feature in range(dimensions):
        all_values = np.concatenate([a[:, feature], b[:, feature]])
        low, high = all_values.min(), all_values.max()
        if high - low < 1e-8:
            distance[feature] = 0.0
            continue
        edges = np.linspace(low, high, bins + 1)
        unsafe_histogram, _ = np.histogram(a[:, feature], bins=edges, density=True)
        safe_histogram, _ = np.histogram(b[:, feature], bins=edges, density=True)
        bin_width = edges[1] - edges[0]
        overlap = np.sum(np.minimum(unsafe_histogram, safe_histogram)) * bin_width
        distance[feature] = 1.0 - overlap
    return distance


def rank_cell(target: str, layer: int, unsafe: torch.Tensor, safe: torch.Tensor) -> tuple[list[dict], list[dict]]:
    require(tuple(unsafe.shape)[1:] == (16384,) and tuple(safe.shape)[1:] == (16384,), "ranking tensor width mismatch")
    histogram = histogram_distance(unsafe, safe)
    unsafe_mean_tensor = unsafe.mean(dim=0)
    safe_mean_tensor = safe.mean(dim=0)
    unsafe_mean = unsafe_mean_tensor.numpy()
    safe_mean = safe_mean_tensor.numpy()
    mean_difference = (unsafe_mean_tensor - safe_mean_tensor).abs().numpy()
    preliminary = 0.4 * histogram + 0.4 * mean_difference
    top_ks = np.argsort(preliminary)[::-1][:KS_SUBSAMPLE]
    ks_full = np.zeros(unsafe.shape[1], dtype=np.float32)
    unsafe_numpy, safe_numpy = unsafe.numpy(), safe.numpy()
    for feature in top_ks:
        statistic, _ = ks_2samp(unsafe_numpy[:, feature], safe_numpy[:, feature])
        ks_full[feature] = statistic
    scores = 0.4 * histogram + 0.4 * mean_difference + 0.2 * ks_full
    top_features = np.argsort(scores)[::-1][:TOP_K]
    unsafe_frequency = (unsafe > ACTIVE_THRESHOLD).float().mean(dim=0).numpy()
    safe_frequency = (safe > ACTIVE_THRESHOLD).float().mean(dim=0).numpy()
    unsafe_mean_magnitude = unsafe_mean_tensor.abs().numpy()
    ranking = []
    kept = []
    for rank, feature in enumerate(top_features, 1):
        values = [histogram[feature], unsafe_mean[feature], safe_mean[feature], mean_difference[feature], ks_full[feature], scores[feature], unsafe_frequency[feature], safe_frequency[feature], unsafe_mean_magnitude[feature]]
        finite = all(np.isfinite(value) for value in values)
        reasons = []
        if not finite:
            reasons.append("NON_FINITE_STATISTIC")
        if unsafe_frequency[feature] < FREQ_MIN:
            reasons.append("UNSAFE_FREQUENCY_BELOW_0.02")
        # Reject a feature when either arm exceeds the prespecified 0.95 ceiling.
        if unsafe_frequency[feature] > FREQ_MAX:
            reasons.append("UNSAFE_FREQUENCY_ABOVE_0.95")
        if safe_frequency[feature] > FREQ_MAX:
            reasons.append("SAFE_FREQUENCY_ABOVE_0.95")
        if scores[feature] < SCORE_MIN:
            reasons.append("COMPOSITE_SCORE_BELOW_0.15")
        if not np.isfinite(unsafe_mean_magnitude[feature]) or unsafe_mean_magnitude[feature] > MAG_MAX:
            reasons.append("EXTREME_UNSAFE_MEAN_MAGNITUDE")
        record = {
            "cwe_id": target, "layer": layer, "feature_id": int(feature), "rank": rank,
            "histogram_distance": float(histogram[feature]),
            "unsafe_mean": float(unsafe_mean[feature]), "safe_mean": float(safe_mean[feature]),
            "absolute_mean_difference": float(mean_difference[feature]),
            "ks_statistic": float(ks_full[feature]), "composite_score": float(scores[feature]),
            "unsafe_activation_frequency": float(unsafe_frequency[feature]),
            "safe_activation_frequency": float(safe_frequency[feature]),
            "unsafe_mean_magnitude": float(unsafe_mean_magnitude[feature]),
            "unsafe_n": int(unsafe.shape[0]), "safe_n": int(safe.shape[0]),
            "finite": finite, "pruning_status": "KEPT" if not reasons else "REJECTED",
            "pruning_reasons": reasons,
        }
        ranking.append(record)
        if not reasons:
            kept.append(record)
    return ranking, kept


def load_inputs():
    require(file_hash(PARTITION) == PARTITION_HASH, "partition hash drift")
    partition = json.loads(PARTITION.read_text(encoding="utf-8"))
    latent_run = json.loads(LATENT_RUN.read_text(encoding="utf-8"))
    require(latent_run["status"] == "COMPLETE" and latent_run["partition_sha256"] == PARTITION_HASH, "feature latent extraction incomplete")
    require(file_hash(LATENT_MANIFEST) == latent_run["latent_manifest_sha256"], "feature latent manifest hash drift")
    union_data = {}
    for layer in LAYERS:
        artifact = latent_run["union_artifacts"][str(layer)]
        path = ROOT / artifact["path"]
        require(file_hash(path) == artifact["sha256"], f"union latent artifact hash drift L{layer}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        require(tuple(payload["latents"].shape) == (302, 16384), f"union tensor shape mismatch L{layer}")
        require(bool(torch.isfinite(payload["latents"]).all()), f"union tensor nonfinite L{layer}")
        ids = [int(row["prompt_id"]) for row in payload["metadata"]]
        require(ids == partition["ranking_prompt_union_ids_source_order"], f"union tensor ID order mismatch L{layer}")
        union_data[layer] = payload
    return partition, latent_run, union_data


def execute_once(partition: dict, union_data: dict, save_cells: bool):
    all_rankings = {}
    all_pruned = {}
    summary = []
    safe_unsafe = []
    for target in TARGETS:
        spec = partition["partitions"][target]
        unsafe_ids = spec["unsafe_ids_source_order"]
        safe_ids = spec["selected_safe_ids_sampling_order"]
        all_rankings[target] = {}
        all_pruned[target] = {}
        for layer in LAYERS:
            ids = [int(row["prompt_id"]) for row in union_data[layer]["metadata"]]
            index = {pid: position for position, pid in enumerate(ids)}
            unsafe = union_data[layer]["latents"][[index[pid] for pid in unsafe_ids]].float().contiguous()
            safe = union_data[layer]["latents"][[index[pid] for pid in safe_ids]].float().contiguous()
            require(unsafe.shape[0] == spec["unsafe_count"] and safe.shape[0] == spec["selected_safe_count"], "cell sample count mismatch")
            cell_key = f"L{layer}"
            cell_artifacts = {}
            if save_cells:
                CELLS.mkdir(parents=True, exist_ok=True)
                stem = target.lower().replace("-", "")
                unsafe_path = CELLS / f"model2_{stem}_layer_{layer}_unsafe_latents.pt"
                safe_path = CELLS / f"model2_{stem}_layer_{layer}_selected_safe_latents.pt"
                atomic_torch(unsafe_path, {"schema_version": "phase21_model2_ranking_cell_v1", "class": "UNSAFE", "cwe_id": target, "layer": layer, "partition_sha256": PARTITION_HASH, "prompt_ids": unsafe_ids, "latents": unsafe})
                atomic_torch(safe_path, {"schema_version": "phase21_model2_ranking_cell_v1", "class": "SELECTED_SAFE", "cwe_id": target, "layer": layer, "partition_sha256": PARTITION_HASH, "prompt_ids": safe_ids, "latents": safe})
                cell_artifacts = {"unsafe_path": str(unsafe_path.relative_to(ROOT)).replace("\\", "/"), "unsafe_sha256": file_hash(unsafe_path), "safe_path": str(safe_path.relative_to(ROOT)).replace("\\", "/"), "safe_sha256": file_hash(safe_path)}
            ranking, kept = rank_cell(target, layer, unsafe, safe)
            all_rankings[target][cell_key] = ranking
            all_pruned[target][cell_key] = {
                "status": "PRUNED_CANDIDATES_AVAILABLE" if kept else "NO_PRUNED_CANDIDATE",
                "input_top200_count": len(ranking), "surviving_count": len(kept), "candidates": kept,
            }
            summary.append({
                "cwe_id": target, "layer": layer, "unsafe_n": unsafe.shape[0], "selected_safe_n": safe.shape[0],
                "features_considered": 16384, "ks_candidate_count": 500, "top200_count": len(ranking),
                "pruned_candidate_count": len(kept), "cell_status": all_pruned[target][cell_key]["status"],
                "top_feature_id": ranking[0]["feature_id"], "top_feature_score": ranking[0]["composite_score"],
            })
            safe_unsafe.append({
                "cwe_id": target, "layer": layer, "unsafe_n": unsafe.shape[0], "selected_safe_n": safe.shape[0],
                "unsafe_ids_sha256": canonical_hash(unsafe_ids), "selected_safe_ids_sampling_order_sha256": canonical_hash(safe_ids),
                **cell_artifacts,
            })
    return all_rankings, all_pruned, summary, safe_unsafe


def main() -> None:
    for output in (RANKINGS, RANKING_SUMMARY, PRUNED, CAUSAL, SAFE_UNSAFE_SUMMARY, CHECKPOINT):
        require(not output.exists(), f"feature-discovery output already exists: {output.name}")
    partition, latent_run, union_data = load_inputs()
    rankings, pruned, summary, safe_unsafe = execute_once(partition, union_data, True)
    replay_rankings, replay_pruned, replay_summary, _ = execute_once(partition, union_data, False)
    determinism = canonical_hash({"rankings": rankings, "pruned": pruned, "summary": summary}) == canonical_hash({"rankings": replay_rankings, "pruned": replay_pruned, "summary": replay_summary})
    require(determinism, "ranking/pruning deterministic replay mismatch")
    ranking_payload = {
        "schema_version": "phase21_model2_statistical_feature_rankings_v1", "status": "COMPLETE",
        "partition_sha256": PARTITION_HASH, "features_considered_per_cell": 16384,
        "histogram_bins": 20, "weights": {"histogram": 0.4, "absolute_mean_difference": 0.4, "ks": 0.2},
        "ks_preselection_count": 500, "non_preselected_ks_default": 0.0,
        "ordering": "numpy.argsort(score)[::-1]", "top_k": 200, "cells": rankings,
        "feature_ranking_only": True, "causal_intervention_run": False, "alpha_selected": False,
        "scanner_run": False, "heldout_used": False,
    }
    atomic_json(RANKINGS, ranking_payload)
    atomic_csv(RANKING_SUMMARY, summary)
    atomic_csv(SAFE_UNSAFE_SUMMARY, safe_unsafe)
    pruned_payload = {
        "schema_version": "phase21_model2_pruned_candidates_v1", "status": "COMPLETE",
        "partition_sha256": PARTITION_HASH,
        "rules": {"active_threshold_strictly_greater": 0.01, "unsafe_frequency_minimum": 0.02, "unsafe_frequency_maximum": 0.95, "safe_frequency_maximum": 0.95, "high_frequency_boolean": "reject if unsafe > 0.95 OR safe > 0.95", "composite_score_minimum": 0.15, "extreme_unsafe_mean_magnitude_maximum": 10000.0},
        "frequency_rule_note": "The unsafe and safe activation-frequency ceilings are applied independently.",
        "cells": pruned, "causal_intervention_run": False, "alpha_selected": False,
    }
    atomic_json(PRUNED, pruned_payload)
    causal_cells = {}
    no_candidate = []
    for target in TARGETS:
        causal_cells[target] = {}
        for layer in LAYERS:
            key = f"L{layer}"
            source = pruned[target][key]
            chosen = source["candidates"][:10]
            status = "CAUSAL_VALIDATION_CANDIDATES_FROZEN" if chosen else "NO_PRUNED_CANDIDATE"
            if not chosen:
                no_candidate.append({"cwe_id": target, "layer": layer})
            causal_cells[target][key] = {"status": status, "candidate_count": len(chosen), "candidates": chosen}
    causal_payload = {
        "schema_version": "phase21_model2_causal_candidate_manifest_v1", "status": "FROZEN_BEFORE_CAUSAL_INTERVENTION",
        "partition_sha256": PARTITION_HASH, "ranking_sha256": file_hash(RANKINGS), "pruned_sha256": file_hash(PRUNED),
        "selection_rule": "first 10 surviving pruned candidates in frozen statistical rank order, or all survivors if fewer",
        "candidate_meaning": "CAUSAL_VALIDATION_CANDIDATES_ONLY_NOT_VALIDATED_SECURITY_FEATURES",
        "cells": causal_cells, "no_pruned_candidate_cells": no_candidate,
        "causal_intervention_run": False, "alpha_selected": False, "steered_generation_run": False,
        "scanner_run": False, "heldout_used": False,
    }
    atomic_json(CAUSAL, causal_payload)
    checkpoint = {
        "schema_version": "phase21_model2_feature_discovery_checkpoint_v1", "status": "PASS",
        "state": "FEATURE_DISCOVERY_COMPLETE", "partition_sha256": PARTITION_HASH,
        "latent_manifest_sha256": file_hash(LATENT_MANIFEST), "safe_unsafe_summary_sha256": file_hash(SAFE_UNSAFE_SUMMARY),
        "ranking_sha256": file_hash(RANKINGS), "ranking_summary_sha256": file_hash(RANKING_SUMMARY),
        "pruned_sha256": file_hash(PRUNED), "causal_candidate_manifest_sha256": file_hash(CAUSAL),
        "determinism_validation": "PASS", "cells_completed": 9, "features_considered_per_cell": 16384,
        "top200_count_per_cell": 200, "no_pruned_candidate_cells": no_candidate,
        "reused_latent_records": latent_run["reused_records"], "newly_extracted_latent_records": latent_run["newly_extracted_records"],
        "next_stage": "causal intervention under a separately frozen protocol",
        "causal_intervention_run": False, "alpha_selected_or_transferred": False,
        "steered_generation_run": False, "scanner_run": False, "heldout_used": False, "model1_modified": False,
    }
    atomic_json(CHECKPOINT, checkpoint)
    print(json.dumps({"status": "PASS", "state": checkpoint["state"], "determinism": "PASS", "summary": summary, "no_pruned_candidate_cells": no_candidate, "hashes": {"rankings": checkpoint["ranking_sha256"], "pruned": checkpoint["pruned_sha256"], "causal": checkpoint["causal_candidate_manifest_sha256"], "checkpoint": file_hash(CHECKPOINT)}}, indent=2))


if __name__ == "__main__":
    main()
