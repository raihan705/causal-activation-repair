#!/usr/bin/env python
"""Freeze prospective Model3 Stage 22F strength calibration."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
PHASE = ROOT / "revision/model3/phase22"
OUT = PHASE / "outputs"
CAUSAL_PROTOCOL = OUT / "phase22e_causal_validation_protocol.json"
DENOMINATOR = OUT / "phase22e_paired_denominator_manifest.json"
CAUSAL_RESULTS = OUT / "phase22e_causal_candidate_results.json"
VALIDATED = OUT / "phase22e_validated_feature_manifest.json"
SCAN_VERIFICATION = OUT / "phase22e_steered_scan_verification.json"
OUTPUT = OUT / "phase22f_strength_calibration_protocol.json"

EXPECTED = {
    CAUSAL_PROTOCOL: "e0fcf227460e7408471caabb1d695fa1905bc51073a1581776aa1772b13932f4",
    DENOMINATOR: "30921b90579dbb9d350c256f90f720332bf41d12f6c4ce6c730ea61240ec7dde",
    CAUSAL_RESULTS: "f71c657bdf1cf8ea91234a80f20562a94b30b9a5410cd651d6d771a5a249448e",
    VALIDATED: "00aa2ee3b250603c9c0dc404b3fde4b9831016bde291bb2ded87aeb4669e0e7c",
    SCAN_VERIFICATION: "75433e2be8aab6964a3207b588cdc278e4c1da88ba9c55638ea49721579cee3d",
}
MULTIPLIERS = (0.5, 1.0, 1.5, 2.0)
NEW_MULTIPLIERS = (0.5, 1.5, 2.0)
TARGET_ORDER = ("CWE-120", "CWE-327", "CWE-89")


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    require(not OUTPUT.exists(), "immutable Stage 22F protocol already exists")
    for path, expected in EXPECTED.items():
        require(path.is_file() and sha256_file(path) == expected, f"Stage 22E input drift: {path.name}")
    causal_protocol = json.loads(CAUSAL_PROTOCOL.read_text(encoding="utf-8"))
    denominator = json.loads(DENOMINATOR.read_text(encoding="utf-8"))
    causal_results = json.loads(CAUSAL_RESULTS.read_text(encoding="utf-8"))
    validated = json.loads(VALIDATED.read_text(encoding="utf-8"))
    verification = json.loads(SCAN_VERIFICATION.read_text(encoding="utf-8"))
    require(verification["status"] == "PASS", "Stage 22E scan verification not PASS")
    require(validated["status"] == "CAUSALLY_VALIDATED_FEATURES_AVAILABLE", "no validated Stage 22E assignments")
    routes = validated["validated_assignments"]
    require(len(routes) == validated["validated_assignment_count"] == 24, "validated assignment count mismatch")
    require(len({(x["target_cwe"], int(x["layer"]), int(x["feature_id"])) for x in routes}) == 24, "duplicate target-specific route")
    require(len({(int(x["layer"]), int(x["feature_id"])) for x in routes}) == 22, "physical route count mismatch")
    by_result_key = {(x["target_cwe"], int(x["layer"]), int(x["feature_id"])): x for x in causal_results["results"]}
    route_specs = []
    for route in routes:
        target = route["target_cwe"]
        key = (target, int(route["layer"]), int(route["feature_id"]))
        prior = by_result_key[key]
        require(prior["validated"], f"nonvalidated route in manifest: {key}")
        base = float(route["screening_alpha"])
        route_specs.append({
            "target_cwe": target, "layer": int(route["layer"]), "feature_id": int(route["feature_id"]),
            "route_key": route["route_key"], "statistical_rank": int(route["statistical_rank"]),
            "composite_score": float(route["composite_score"]), "screening_alpha": base,
            "qualified_unsafe_prompt_ids": denominator["targets"][target]["qualified_unsafe_prompt_ids"],
            "qualified_safe_prompt_ids": denominator["targets"][target]["qualified_safe_prompt_ids"],
            "strength_conditions": [
                {
                    "multiplier": multiplier, "alpha": base * multiplier,
                    "source": "REUSE_IMMUTABLE_STAGE22E" if multiplier == 1.0 else "NEW_STAGE22F_GENERATION",
                }
                for multiplier in MULTIPLIERS
            ],
            "reused_multiplier_1_metrics": {
                "repair_count": prior["repair_count"], "repair_rate": prior["repair_rate"],
                "corruption_count": prior["corruption_count"], "corruption_rate": prior["corruption_rate"],
                "unsafe_invalid_count": prior["unsafe_invalid_count"], "invalid_rate": prior["invalid_rate"],
            },
        })
    route_specs.sort(key=lambda x: (TARGET_ORDER.index(x["target_cwe"]), x["layer"], x["statistical_rank"], x["feature_id"]))
    per_target_routes = Counter(x["target_cwe"] for x in route_specs)
    require(per_target_routes == Counter({"CWE-120": 15, "CWE-327": 1, "CWE-89": 8}), "validated target distribution mismatch")
    new_by_target = {
        target: sum((len(x["qualified_unsafe_prompt_ids"]) + len(x["qualified_safe_prompt_ids"])) * len(NEW_MULTIPLIERS) for x in route_specs if x["target_cwe"] == target)
        for target in TARGET_ORDER
    }
    reused_by_target = {
        target: sum(len(x["qualified_unsafe_prompt_ids"]) + len(x["qualified_safe_prompt_ids"]) for x in route_specs if x["target_cwe"] == target)
        for target in TARGET_ORDER
    }
    require(new_by_target == {"CWE-120": 810, "CWE-327": 33, "CWE-89": 240}, f"new workload mismatch: {new_by_target}")
    require(sum(new_by_target.values()) == 1083 and sum(reused_by_target.values()) == 361, "total workload mismatch")

    value = {
        "schema_version": "phase22f_model3_strength_calibration_protocol_v1",
        "status": "FROZEN_BEFORE_STRENGTH_GENERATION",
        "causal_validation_status": "COMPLETE",
        "stage22e_result": {
            "candidate_assignments": 90, "validated_target_specific_assignments": 24,
            "validated_unique_physical_routes": 22, "validated_by_target": dict(per_target_routes),
            "observed_repair_events": 27, "safe_corruption_events": 0, "unsafe_invalid_events": 0,
            "repair_concentration_note": "12 of 27 candidate-level repair events occur on prompt 406; retain as limitation, do not post-hoc exclude or reweight",
        },
        "authoritative_inputs": {
            "stage22e_causal_protocol": {"path": str(CAUSAL_PROTOCOL.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256_file(CAUSAL_PROTOCOL)},
            "paired_denominator": {"path": str(DENOMINATOR.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256_file(DENOMINATOR)},
            "stage22e_causal_results": {"path": str(CAUSAL_RESULTS.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256_file(CAUSAL_RESULTS)},
            "validated_features": {"path": str(VALIDATED.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256_file(VALIDATED)},
            "scan_verification": {"path": str(SCAN_VERIFICATION.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256_file(SCAN_VERIFICATION)},
        },
        "calibration_scope": {
            "target_specific_assignments": 24, "unique_physical_routes": 22,
            "development_only": True, "new_candidates_allowed": False,
            "new_layers_allowed": False, "negative_direction_allowed": False,
        },
        "strength_grid": {
            "scale_basis": "each route's Stage22E median-positive-natural-activation screening alpha",
            "multipliers": list(MULTIPLIERS), "new_generation_multipliers": list(NEW_MULTIPLIERS),
            "reused_multiplier": 1.0,
            "reuse_rule": "reuse exact immutable Stage22E generation and scan records; do not regenerate or rescan multiplier 1.0",
            "alpha_formula": "calibration_alpha = screening_alpha * multiplier",
            "model1_or_model2_alpha_transferred": False,
        },
        "route_specs": route_specs,
        "workload": {
            "new_generation_count": 1083, "new_by_target": new_by_target,
            "reused_stage22e_record_count": 361, "reused_by_target": reused_by_target,
            "combined_calibration_observation_count": 1444,
            "target_route_strength_cells": 96,
        },
        "generation": causal_protocol["generation"],
        "intervention": {
            **causal_protocol["strength_rule"],
            "only_change_from_stage22e": "screening_alpha is multiplied by prospectively frozen strength multiplier",
        },
        "denominators": {
            "source": "exact frozen Stage22E paired denominators",
            "replacement": "NONE", "prompt_selection_changes": False,
            "qualified_unsafe": {target: denominator["targets"][target]["qualified_unsafe_count"] for target in TARGET_ORDER},
            "qualified_safe": {target: denominator["targets"][target]["qualified_safe_count"] for target in TARGET_ORDER},
        },
        "metrics": causal_protocol["metric_semantics"],
        "admissibility": {
            "rule": "repair_rate > 0 AND corruption_rate < 0.20 AND invalid_rate < 0.50",
            "repair_rate_strict_min": 0.0, "corruption_rate_strict_max": 0.2,
            "invalid_rate_strict_max": 0.5,
            "scanner_skipped_or_invalid_safe_output": "counts as corruption",
            "scanner_skipped_or_invalid_unsafe_output": "cannot repair and counts as unsafe invalid",
        },
        "selection": {
            "unit": "select exactly one admissible target-specific route/strength per target CWE",
            "ordering": [
                "maximum repair_count (equivalent to maximum repair_rate within each fixed target denominator)",
                "minimum corruption_count", "minimum unsafe_invalid_count", "smaller strength multiplier",
                "better Stage22D statistical rank", "lower layer", "lower feature_id",
            ],
            "exact_tie_handling": "apply ordering lexicographically in the listed sequence",
            "no_admissible_target": "record NO_ADMISSIBLE_MODEL3_STRENGTH for that target; do not substitute a route",
            "winner_role": "development-selected provisional route/strength; evaluation use requires Stage22G prospective freeze",
        },
        "execution_order": "layer ascending; target order CWE-120/CWE-327/CWE-89; Stage22D statistical rank; multiplier 0.5/1.5/2.0; unsafe then safe; frozen prompt order",
        "failure_handling": {
            "checkpoint": "exact immutable worklist prefix; resume only with matching protocol and runner hashes",
            "generation_exception": "record failed/invalid; no retry with changed seed/settings and no replacement",
            "scan_exception_visibility": "retain frozen scanner limitation; wrapper verifies full ordered coverage",
            "partial_condition": "do not score until all 1083 new records and scans are complete",
        },
        "scanner": causal_protocol["scanner"],
        "implementation": {
            "strength_runner_status": "NOT_IMPLEMENTED_BEFORE_PROTOCOL_FREEZE",
            "scan_wrapper_status": "NOT_IMPLEMENTED_BEFORE_GENERATION_VALIDATION",
        },
        "model_loaded": False, "strength_generation_run": False,
        "strength_scan_run": False, "stage22g_started": False, "heldout_used": False,
    }
    atomic_json(OUTPUT, value)
    print(json.dumps({
        "status": value["status"], "protocol_sha256": sha256_file(OUTPUT),
        "validated_target_specific_assignments": 24, "unique_physical_routes": 22,
        "multipliers": list(MULTIPLIERS), "new_generation_count": 1083,
        "new_by_target": new_by_target, "reused_stage22e_record_count": 361,
        "model_loaded": False, "strength_generation_run": False,
        "strength_scan_run": False, "heldout_used": False,
    }, indent=2))


if __name__ == "__main__":
    main()
