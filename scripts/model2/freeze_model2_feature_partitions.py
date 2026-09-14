#!/usr/bin/env python
"""Freeze deterministic Model2 B0 safe/unsafe feature-ranking partitions."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "revision/model2/phase21/outputs"
SCAN = OUT / "model2_b0_dev_scan.json"
B0 = OUT / "model2_b0_dev_outputs.json"
SOURCE_MANIFEST = OUT / "model2_activation_source_manifest.json"
VIABILITY = OUT / "model2_layer_viability.json"
DEST = OUT / "model2_feature_partition_manifest.json"
TARGETS = ("CWE-120", "CWE-327", "CWE-89")
LAYERS = (9, 20, 31)
SEED = 42

EXPECTED = {
    "scan": "65d33037b09a6796d598c26747a01b5347ea15725a418eaadc8bbcd11350ba92",
    "activation_manifest": "0ecff36e346740e71f0ce43d7eff43d00117c891b9a6f6e05adea2990f4b1f47",
    "source_reconstruction": "bfeb71655fd17bc072527eca8ca92dbaa893337bd72a607b1a3eed0c478d68f9",
    "viability": "a5809cfe2450e21ad58195e8cfd31fd331c730841385db2d06862e552639020f",
    "viability_checkpoint": "988778b84ebcf82efe3f2328fea7ac583cfb9a06ebb834e4b9e7d750daf0cb3e",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def build(scan_rows: list[dict], b0_rows: list[dict], existing_ids: set[int]) -> dict:
    b0_ids = [int(row["prompt_id"]) for row in b0_rows]
    if len(b0_ids) != 1341 or len(set(b0_ids)) != 1341:
        raise RuntimeError("B0 IDs are not the accepted unique 1,341-record population")
    if [int(row["prompt_id"]) for row in scan_rows] != b0_ids:
        raise RuntimeError("verified scan order/IDs do not match frozen B0 output")
    partitions = {}
    union_ids = set()
    for target in TARGETS:
        unsafe = [int(row["prompt_id"]) for row in scan_rows if not row["skipped"] and target in row["vulnerable_cwes"]]
        safe_candidates = [int(row["prompt_id"]) for row in scan_rows if not row["skipped"] and target not in row["vulnerable_cwes"]]
        sample_n = min(len(safe_candidates), 5 * len(unsafe))
        indices = torch.randperm(
            len(safe_candidates), generator=torch.Generator().manual_seed(SEED)
        )[:sample_n].tolist()
        selected_sampling = [safe_candidates[index] for index in indices]
        selected_set = set(selected_sampling)
        selected_source = [pid for pid in b0_ids if pid in selected_set]
        if len(set(unsafe)) != len(unsafe) or len(set(selected_sampling)) != len(selected_sampling):
            raise RuntimeError(f"duplicate partition ID for {target}")
        if set(unsafe) & selected_set:
            raise RuntimeError(f"unsafe/safe overlap for {target}")
        union_ids.update(unsafe)
        union_ids.update(selected_sampling)
        partitions[target] = {
            "unsafe_definition": "scanner-eligible accepted Model2 B0 development row containing target CWE",
            "safe_candidate_definition": "scanner-eligible accepted Model2 B0 development row not containing target CWE",
            "unsafe_count": len(unsafe), "safe_candidate_count": len(safe_candidates),
            "selected_safe_count": len(selected_sampling), "safe_cap_multiplier": 5,
            "selection_seed": SEED, "selection_rng": "torch.randperm with torch.Generator().manual_seed(42)",
            "unsafe_ids_source_order": unsafe,
            "safe_candidate_ids_source_order": safe_candidates,
            "selected_safe_ids_sampling_order": selected_sampling,
            "selected_safe_ids_source_order": selected_source,
            "unsafe_ids_sha256": canonical_sha(unsafe),
            "safe_candidate_ids_sha256": canonical_sha(safe_candidates),
            "selected_safe_ids_sampling_order_sha256": canonical_sha(selected_sampling),
            "selected_safe_ids_source_order_sha256": canonical_sha(selected_source),
        }
    expected_unsafe = {"CWE-120": 22, "CWE-327": 24, "CWE-89": 10}
    expected_safe = {"CWE-120": 110, "CWE-327": 120, "CWE-89": 50}
    if {key: value["unsafe_count"] for key, value in partitions.items()} != expected_unsafe:
        raise RuntimeError("accepted target-CWE unsafe counts differ from the approved protocol")
    if {key: value["selected_safe_count"] for key, value in partitions.items()} != expected_safe:
        raise RuntimeError("selected safe counts differ from the approved 5x cap")
    union_source = [pid for pid in b0_ids if pid in union_ids]
    return {
        "schema_version": "phase21_model2_feature_partition_manifest_v1",
        "status": "FROZEN_BEFORE_RANKING",
        "scanner_label_source": "accepted Model2 B0 development scan only",
        "source_scan_path": "revision/model2/phase21/outputs/model2_b0_dev_scan.json",
        "source_scan_sha256": sha256(SCAN),
        "source_b0_path": "revision/model2/phase21/outputs/model2_b0_dev_outputs.json",
        "source_b0_sha256": sha256(B0),
        "targets": list(TARGETS), "layers": list(LAYERS), "d_sae": 16384,
        "scanner_skipped_excluded_from_both_classes": True,
        "source_cwe_used_as_partition_criterion": False,
        "partitions": partitions,
        "ranking_prompt_union_count": len(union_source),
        "ranking_prompt_union_ids_source_order": union_source,
        "ranking_prompt_union_ids_sha256": canonical_sha(union_source),
        "existing_source_matched_latent_prompt_count": len(existing_ids),
        "reusable_union_prompt_count": len(union_ids & existing_ids),
        "missing_union_prompt_count": len(union_ids - existing_ids),
        "reusable_union_prompt_ids_source_order": [pid for pid in union_source if pid in existing_ids],
        "missing_union_prompt_ids_source_order": [pid for pid in union_source if pid not in existing_ids],
        "source_matched_support_preserved_separately": {
            "CWE-120": {"source_eligible": 110, "source_vulnerable": 12, "source_safe": 98, "global_scanner_unsafe": 22},
            "CWE-327": {"source_eligible": 46, "source_vulnerable": 3, "source_safe": 43, "global_scanner_unsafe": 24, "status": "DISCOVERY_USABLE_SOURCE_MATCHED_SUPPORT_LIMITED"},
            "CWE-89": {"source_eligible": 24, "source_vulnerable": 10, "source_safe": 14, "global_scanner_unsafe": 10},
        },
        "model1_ranking_source": {
            "path": "phases/phase4/rank_features_statistical_multilayer.py",
            "sha256": sha256(ROOT / "phases/phase4/rank_features_statistical_multilayer.py"),
        },
        "model1_pruning_source": {
            "path": "phases/phase4/prune_feature_candidates.py",
            "sha256": sha256(ROOT / "phases/phase4/prune_feature_candidates.py"),
        },
        "heldout_used": False, "new_scan_run": False, "causal_intervention_run": False,
    }


def main() -> None:
    paths = {
        "scan": SCAN,
        "activation_manifest": OUT / "model2_activation_manifest.csv",
        "source_reconstruction": OUT / "model2_source_reconstruction.csv",
        "viability": VIABILITY,
        "viability_checkpoint": OUT / "model2_layer_viability_checkpoint.json",
    }
    for key, path in paths.items():
        if sha256(path) != EXPECTED[key]:
            raise RuntimeError(f"accepted upstream hash drift: {key}")
    viability = json.loads(VIABILITY.read_text(encoding="utf-8"))
    if viability["viable_layers"] != list(LAYERS):
        raise RuntimeError("frozen viable-layer set drift")
    source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    existing_ids = {int(row["prompt_id"]) for row in source["records"] if row["source"] == "MODEL2_B0"}
    scan_rows = json.loads(SCAN.read_text(encoding="utf-8"))
    b0_rows = json.loads(B0.read_text(encoding="utf-8"))
    first = build(scan_rows, b0_rows, existing_ids)
    second = build(scan_rows, b0_rows, existing_ids)
    if first != second:
        raise RuntimeError("deterministic partition rebuild mismatch")
    if DEST.exists():
        if json.loads(DEST.read_text(encoding="utf-8")) != first:
            raise RuntimeError("existing frozen partition differs")
    else:
        tmp = DEST.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(first, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, DEST)
    print(json.dumps({
        "status": "PASS", "determinism_validation": "PASS",
        "unsafe_counts": {key: value["unsafe_count"] for key, value in first["partitions"].items()},
        "safe_candidate_counts": {key: value["safe_candidate_count"] for key, value in first["partitions"].items()},
        "selected_safe_counts": {key: value["selected_safe_count"] for key, value in first["partitions"].items()},
        "union": first["ranking_prompt_union_count"], "reused": first["reusable_union_prompt_count"],
        "missing": first["missing_union_prompt_count"], "sha256": sha256(DEST),
    }, indent=2))


if __name__ == "__main__":
    main()
