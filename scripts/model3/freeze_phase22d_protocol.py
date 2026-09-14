#!/usr/bin/env python
"""Freeze Model3 Stage 22D partitions and activation/ranking protocol."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[4]
PHASE = ROOT / "revision/model3/phase22"
OUT = PHASE / "outputs"
B0 = OUT / "phase22c_b0_dev_outputs.json"
FROZEN_B0 = OUT / "phase22c_b0_population_manifest.json"
VERIFY = OUT / "phase22c_b0_scan_verification.json"
PARTITION = OUT / "phase22d_feature_partition_manifest.json"
PROTOCOL = OUT / "phase22d_activation_ranking_protocol.json"
RAW_RUNNER = PHASE / "scripts/extract_phase22d_raw_activations.py"
LATENT_RUNNER = PHASE / "scripts/encode_phase22d_latents.py"
RANK_RUNNER = PHASE / "scripts/rank_phase22d_features.py"
SAE_HELPER = PHASE / "scripts/phase22d_sae.py"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
TARGETS = ("CWE-120", "CWE-327", "CWE-89")
LAYERS = (7, 15, 23)
SEED = 42
EXPECTED_SOURCE = {
    "CWE-120": {"population": 110, "unsafe": 14, "safe": 96, "selected_safe": 70},
    "CWE-327": {"population": 46, "unsafe": 6, "safe": 40, "selected_safe": 30},
    "CWE-89": {"population": 24, "unsafe": 5, "safe": 19, "selected_safe": 19},
}
EXPECTED_B0_SHA256 = "7390fe102e391baa77af5316e8f315acef99057f2d9f8283ad85fc089731b6b0"
EXPECTED_FROZEN_SHA256 = "042eda6d74db12ad2f3ccf8b6c32ff419f2e5fb41cbb841287e1d4bcf1df8c5d"
EXPECTED_VERIFY_SHA256 = "bb91c62e2a29e2672c2ed59fc404df0534107831f2b418469d1c1fe41bd3d300"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def model_snapshot_path() -> Path:
    path = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots" / MODEL_REVISION
    require(path.is_dir() and (path / "tokenizer_config.json").is_file(), f"pinned model snapshot missing: {path}")
    return path


def build_partition(frozen: dict[str, Any]) -> dict[str, Any]:
    records = frozen["records"]
    partitions = {}
    union = set()
    for target in TARGETS:
        matched = [x for x in records if x["cwe_identifier"] == target]
        unsafe = [int(x["prompt_id"]) for x in matched if x["scanner_eligible"] and x["target_vulnerable"]]
        safe_candidates = [int(x["prompt_id"]) for x in matched if x["scanner_eligible"] and x["target_safe"]]
        sample_n = min(len(safe_candidates), 5 * len(unsafe))
        indices = torch.randperm(len(safe_candidates), generator=torch.Generator().manual_seed(SEED))[:sample_n].tolist()
        selected_sampling = [safe_candidates[index] for index in indices]
        selected_set = set(selected_sampling)
        selected_source = [int(x["prompt_id"]) for x in matched if int(x["prompt_id"]) in selected_set]
        expected = EXPECTED_SOURCE[target]
        require((len(matched), len(unsafe), len(safe_candidates), sample_n) == (expected["population"], expected["unsafe"], expected["safe"], expected["selected_safe"]), f"frozen denominator mismatch: {target}")
        require(not (set(unsafe) & selected_set), f"unsafe/safe overlap: {target}")
        union.update(unsafe)
        union.update(selected_sampling)
        partitions[target] = {
            "population_definition": "scanner-eligible Model3 B0 development records whose source cwe_identifier equals target CWE",
            "unsafe_definition": "source-matched record with frozen target-CWE scanner finding",
            "safe_candidate_definition": "source-matched record with no frozen target-CWE scanner finding",
            "population_count": len(matched), "unsafe_count": len(unsafe),
            "safe_candidate_count": len(safe_candidates), "selected_safe_count": sample_n,
            "safe_cap_multiplier": 5, "selection_seed": SEED,
            "selection_rng": "torch.randperm with a fresh torch.Generator().manual_seed(42) per target CWE",
            "unsafe_ids_source_order": unsafe, "safe_candidate_ids_source_order": safe_candidates,
            "selected_safe_ids_sampling_order": selected_sampling,
            "selected_safe_ids_source_order": selected_source,
            "unsafe_ids_sha256": canonical_hash(unsafe),
            "safe_candidate_ids_sha256": canonical_hash(safe_candidates),
            "selected_safe_ids_sampling_order_sha256": canonical_hash(selected_sampling),
            "selected_safe_ids_source_order_sha256": canonical_hash(selected_source),
        }
    union_source = [int(x["prompt_id"]) for x in records if int(x["prompt_id"]) in union]
    require(len(union_source) == 144 and len(set(union_source)) == 144, "ranking union must contain 144 unique IDs")
    return {
        "schema_version": "phase22d_model3_feature_partition_manifest_v1",
        "status": "FROZEN_BEFORE_ACTIVATION_EXTRACTION", "split": "DEVELOPMENT",
        "scanner_label_source": "verified Model3 source-matched B0 scan only",
        "source_b0_path": "revision/model3/phase22/outputs/phase22c_b0_dev_outputs.json",
        "source_b0_sha256": sha256_file(B0),
        "frozen_b0_population_path": "revision/model3/phase22/outputs/phase22c_b0_population_manifest.json",
        "frozen_b0_population_sha256": sha256_file(FROZEN_B0),
        "targets": list(TARGETS), "layers": list(LAYERS), "d_sae": 131072,
        "scanner_skipped_excluded_from_both_classes": True,
        "source_cwe_matching_required": True, "model1_or_model2_labels_used": False,
        "partitions": partitions, "ranking_prompt_union_count": len(union_source),
        "ranking_prompt_union_ids_source_order": union_source,
        "ranking_prompt_union_ids_sha256": canonical_hash(union_source),
        "heldout_used": False, "new_scan_run": False, "causal_intervention_run": False,
    }


def main() -> None:
    require(not PARTITION.exists() and not PROTOCOL.exists(), "Stage 22D protocol already frozen")
    for path in (B0, FROZEN_B0, VERIFY, RAW_RUNNER, LATENT_RUNNER, RANK_RUNNER, SAE_HELPER):
        require(path.is_file(), f"missing required file: {path}")
    require(sha256_file(B0) == EXPECTED_B0_SHA256, "accepted B0 hash drift")
    require(sha256_file(FROZEN_B0) == EXPECTED_FROZEN_SHA256, "accepted frozen population hash drift")
    require(sha256_file(VERIFY) == EXPECTED_VERIFY_SHA256, "accepted verification hash drift")
    verify = json.loads(VERIFY.read_text(encoding="utf-8"))
    require(verify["verification_status"] == "PASS" and verify["eligible_count"] == 180 and verify["skipped_count"] == 0, "Stage 22C verification not accepted")
    frozen = json.loads(FROZEN_B0.read_text(encoding="utf-8"))
    first = build_partition(frozen)
    second = build_partition(frozen)
    require(first == second, "partition deterministic rebuild mismatch")
    atomic_json(PARTITION, first)
    b0_by_id = {int(x["prompt_id"]): x for x in json.loads(B0.read_text(encoding="utf-8"))}
    tokenizer = AutoTokenizer.from_pretrained(model_snapshot_path(), local_files_only=True)
    token_records = []
    for order, pid in enumerate(first["ranking_prompt_union_ids_source_order"]):
        row = b0_by_id[pid]
        encoded = tokenizer(row["generated_code"], add_special_tokens=True, truncation=True, max_length=512)
        ids = encoded["input_ids"]
        require(len(ids) > 8, f"record outside Qwen SAE post-offset support: {pid}")
        token_records.append({
            "ranking_order": order, "prompt_id": pid,
            "generated_code_sha256": hashlib.sha256(row["generated_code"].encode()).hexdigest(),
            "input_token_count": len(ids), "selected_token_index": len(ids) - 1,
            "selected_token_id": int(ids[-1]), "selected_token_is_special": int(ids[-1]) in set(tokenizer.all_special_ids),
        })
    protocol = {
        "schema_version": "phase22d_model3_activation_ranking_protocol_v1",
        "status": "FROZEN_BEFORE_ACTIVATION_EXTRACTION", "split": "DEVELOPMENT",
        "partition_path": "revision/model3/phase22/outputs/phase22d_feature_partition_manifest.json",
        "partition_sha256": sha256_file(PARTITION),
        "ranking_prompt_union_count": 144,
        "ranking_prompt_union_ids_source_order": first["ranking_prompt_union_ids_source_order"],
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "dtype": "bfloat16", "device": "cuda:0", "quantization": False, "offload": False},
        "sae": {
            "repository": "andyrdt/saes-qwen2.5-7b-instruct", "revision": "c37e53c4bb07127ad17ab88f28b93d4e87142e59",
            "loader_revision": "de9138b02fffdf9919c53cee828beb4e05049741", "layers": list(LAYERS),
            "trainer": "trainer_0", "type": "BatchTopK", "d_in": 3584, "d_sae": 131072,
            "k": 32, "encoding": "encode(use_threshold=True)", "sae_batch_size": 1,
        },
        "tokenization": {
            "input": "raw generated_code text", "chat_template": False, "add_special_tokens": True,
            "truncation": True, "max_length": 512, "model_batch_size": 1,
            "selection": "last index whose attention_mask value is one",
            "minimum_nonpadding_tokens": 9,
            "qwen_sae_first_eight_token_exclusion_respected": True,
        },
        "ranking": {
            "cells": 9, "histogram_bins": 20,
            "weights": {"histogram_distance": 0.4, "absolute_raw_mean_difference": 0.4, "ks_statistic": 0.2},
            "ks_preselection_count": 500, "non_preselected_ks_default": 0.0,
            "top_ranked_features_per_cell": 200,
            "candidate_rule": "first 10 finite top-200 features active (>0) in at least one unsafe-arm record",
            "candidates_per_cell_maximum": 10,
            "absolute_model1_or_model2_frequency_score_or_magnitude_thresholds_reused": False,
            "reason": "BatchTopK activation scale is Model3-specific; fixed cross-model activation-scale pruning thresholds are prohibited",
            "deterministic_full_replay_required": True,
        },
        "scripts": {
            "raw_extraction_path": "revision/model3/phase22/scripts/extract_phase22d_raw_activations.py",
            "raw_extraction_sha256": sha256_file(RAW_RUNNER),
            "latent_encoding_path": "revision/model3/phase22/scripts/encode_phase22d_latents.py",
            "latent_encoding_sha256": sha256_file(LATENT_RUNNER),
            "sae_helper_path": "revision/model3/phase22/scripts/phase22d_sae.py",
            "sae_helper_sha256": sha256_file(SAE_HELPER),
            "ranking_path": "revision/model3/phase22/scripts/rank_phase22d_features.py",
            "ranking_sha256": sha256_file(RANK_RUNNER),
        },
        "records": token_records, "scanner_run": False, "causal_intervention_run": False,
        "alpha_selected": False, "heldout_used": False,
    }
    atomic_json(PROTOCOL, protocol)
    print(json.dumps({
        "status": "PASS", "partition_sha256": sha256_file(PARTITION),
        "protocol_sha256": sha256_file(PROTOCOL), "union_count": 144,
        "class_counts": {target: {"unsafe": first["partitions"][target]["unsafe_count"], "selected_safe": first["partitions"][target]["selected_safe_count"]} for target in TARGETS},
        "token_count_min": min(x["input_token_count"] for x in token_records),
        "token_count_max": max(x["input_token_count"] for x in token_records),
        "selected_special_token_count": sum(x["selected_token_is_special"] for x in token_records),
        "script_hashes": protocol["scripts"],
    }, indent=2))


if __name__ == "__main__":
    main()
