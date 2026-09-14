#!/usr/bin/env python3
"""Freeze the approved prospective Model2 route and strength protocol.

This script performs deterministic metadata processing only. It never imports
model libraries, loads an SAE, invokes a scanner, or reads held-out data.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "revision/model2/phase21/outputs"
TARGET_CWES = ("CWE-120", "CWE-327", "CWE-89")
EXPECTED_SOURCE_COUNTS = {"CWE-120": 110, "CWE-327": 46, "CWE-89": 24}
EXPECTED_TARGET_COUNTS = {
    "CWE-120": {"target_vulnerable": 12, "target_safe": 98},
    "CWE-327": {"target_vulnerable": 3, "target_safe": 43},
    "CWE-89": {"target_vulnerable": 10, "target_safe": 14},
}

PATHS = {
    "causal_results": OUT / "model2_validated_features.json",
    "causal_checkpoint": OUT / "model2_causal_validation_checkpoint.json",
    "candidate_manifest": OUT / "model2_causal_candidate_manifest_model1_fidelity.json",
    "support_table": OUT / "model2_dev_cwe_support.csv",
    "development_source": ROOT / "data/cyberseceval/dev_prompts.json",
    "b0_protocol": OUT / "model2_b0_dev_protocol.json",
    "b0_outputs": OUT / "model2_b0_dev_outputs.json",
    "b0_scan": OUT / "model2_b0_dev_scan.json",
    "b0_scan_return": OUT / "model2_b0_dev_scan_return_manifest.json",
    "scanner_provenance": OUT / "model2_scanner_provenance.json",
    "model_sae_selection": OUT / "model_sae_selection_record.json",
    "causal_protocol": OUT / "model2_causal_execution_protocol_v3.json",
}

EXPECTED_HASHES = {
    "causal_results": "5ff0714b42178e5f691ccd01e6cd3d5944cdb67aacba0bcb53c85cd4441b820f",
    "causal_checkpoint": "b798b55a8cf37060bd6c94ef92d178f949530fc7a26540169a522b1885aa0b1a",
    "candidate_manifest": "be072afa3a9c58c20729bcd5deb343007f15b7c981e4fe02bb857bc17d00b401",
    "support_table": "afa6916fe9ec154461ff6cefcf98dc1a193a7f02917bdd138adb2721e8e67f12",
    "development_source": "18c0e78c3589c6c0ef7a4972d116094aa973bce8e8d06138c8844ca0c66a1680",
    "b0_protocol": "5c051ef142191fdf7d9b3d9a69a34ca000a8150ad5f185f9d6f9d97405fb7928",
    "b0_outputs": "d90f58718a40aa702e25018265bb0ab843afd74b3cbfada81cdd139c2c9a6760",
    "b0_scan": "65d33037b09a6796d598c26747a01b5347ea15725a418eaadc8bbcd11350ba92",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_hash(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def write_json(path: Path, value: Any) -> str:
    payload = json_bytes(value)
    path.write_bytes(payload)
    return sha256_bytes(payload)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def index_unique(records: list[dict[str, Any]], label: str) -> dict[int, dict[str, Any]]:
    ids = [int(record["prompt_id"]) for record in records]
    require(len(ids) == len(set(ids)), f"{label} contains duplicate prompt IDs")
    return {int(record["prompt_id"]): record for record in records}


def candidate_positions(manifest: dict[str, Any]) -> dict[tuple[str, int, int], dict[str, Any]]:
    positions: dict[tuple[str, int, int], dict[str, Any]] = {}
    global_position = 0
    for cwe_ordinal, (cwe, layers) in enumerate(manifest["cells"].items(), 1):
        cwe_position = 0
        for layer_ordinal, (layer_key, cell) in enumerate(layers.items(), 1):
            for candidate_index, candidate in enumerate(cell["candidates"]):
                global_position += 1
                cwe_position += 1
                key = (cwe, int(candidate["layer"]), int(candidate["feature_id"]))
                require(key not in positions, f"Duplicate frozen candidate {key}")
                positions[key] = {
                    "global_position_1_based": global_position,
                    "target_cwe_position_1_based": cwe_position,
                    "cwe_object_ordinal_1_based": cwe_ordinal,
                    "layer_object_ordinal_1_based": layer_ordinal,
                    "candidate_list_index_0_based": candidate_index,
                    "json_pointer": f"/cells/{cwe}/{layer_key}/candidates/{candidate_index}",
                }
    return positions


def first_loss_reason(winner: dict[str, Any], loser: dict[str, Any]) -> str:
    comparisons = (
        ("repair_rate", "lower repair_rate"),
        ("corruption_rate", "higher corruption_rate"),
        ("invalid_rate", "higher invalid_rate"),
        ("candidate_manifest_position", "later frozen candidate-manifest position"),
        ("feature_id", "higher feature_id after all preceding fields tied"),
    )
    for field, reason in comparisons:
        if loser[field] != winner[field]:
            return reason
    return "ERROR_IDENTICAL_SELECTION_KEY"


def build_route_artifacts() -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, int]]]:
    validated = load_json(PATHS["causal_results"])
    manifest = load_json(PATHS["candidate_manifest"])
    require(validated["causal_replication_signal"] == "PASS", "Causal replication signal changed")
    require(float(validated["alpha"]) == 20.0 and validated["alpha_role"] == "SCREENING_ONLY", "Causal alpha role changed")
    positions = candidate_positions(manifest)

    complete_ordering: dict[str, list[dict[str, Any]]] = {}
    selected_routes: dict[str, dict[str, int]] = {}
    selection_trace: dict[str, Any] = {}
    for cwe in TARGET_CWES:
        eligible = []
        for record in validated["features"]:
            if record["target_cwe"] != cwe or record["validation_status"] != "VALIDATED":
                continue
            key = (cwe, int(record["layer"]), int(record["feature_id"]))
            require(key in positions, f"Validated candidate absent from frozen manifest: {key}")
            pos = positions[key]
            eligible.append({
                "target_cwe": cwe,
                "layer": int(record["layer"]),
                "feature_id": int(record["feature_id"]),
                "repair_count": int(record["repair_count"]),
                "qualified_unsafe_n": int(record["qualified_unsafe_n"]),
                "repair_rate": float(record["repair_rate"]),
                "safe_corruption_count": int(record["safe_corruption_count"]),
                "safe_n": int(record["safe_n"]),
                "corruption_rate": float(record["corruption_rate"]),
                "unsafe_invalid_count": int(record["unsafe_invalid_count"]),
                "invalid_rate": float(record["invalid_rate"]),
                "candidate_manifest_position": int(pos["global_position_1_based"]),
                "target_cwe_manifest_position": int(pos["target_cwe_position_1_based"]),
                "candidate_manifest_json_pointer": pos["json_pointer"],
                "original_statistical_rank_descriptive_only": int(record["original_statistical_rank"]),
            })
        require(eligible, f"No validated candidates for {cwe}")
        ordered = sorted(
            eligible,
            key=lambda row: (
                -row["repair_rate"],
                row["corruption_rate"],
                row["invalid_rate"],
                row["candidate_manifest_position"],
                row["feature_id"],
            ),
        )
        winner = ordered[0]
        for rank, row in enumerate(ordered, 1):
            row["selection_rank"] = rank
            row["selected"] = rank == 1
            row["selection_reason"] = (
                "highest-ranked eligible candidate under MODEL2_PROSPECTIVE_ROUTE_SELECTION_V1"
                if rank == 1 else first_loss_reason(winner, row)
            )
        complete_ordering[cwe] = ordered
        selected_routes[cwe] = {"layer": winner["layer"], "feature_id": winner["feature_id"]}
        selection_trace[cwe] = {
            "winner": selected_routes[cwe],
            "winner_repair": f"{winner['repair_count']}/{winner['qualified_unsafe_n']}",
            "winner_manifest_position": winner["candidate_manifest_position"],
            "winner_target_cwe_manifest_position": winner["target_cwe_manifest_position"],
            "winner_json_pointer": winner["candidate_manifest_json_pointer"],
            "eligible_candidate_count": len(ordered),
            "reason": "first candidate after applying the frozen five-field hierarchy",
        }

    require(selected_routes == {
        "CWE-120": {"layer": 20, "feature_id": 14471},
        "CWE-327": {"layer": 20, "feature_id": 529},
        "CWE-89": {"layer": 20, "feature_id": 12328},
    }, f"Prospective route result unexpected: {selected_routes}")

    route_map_hash = canonical_hash(selected_routes)
    amendment = {
        "schema_version": "phase21_model2_route_selection_amendment_v1",
        "status": "FROZEN_BEFORE_STRENGTH_OUTCOMES",
        "rule_name": "MODEL2_PROSPECTIVE_ROUTE_SELECTION_V1",
        "rule_version": 1,
        "scope": "MODEL2_ONLY_PROSPECTIVE_NOT_A_RECOVERED_MODEL1_RULE",
        "eligibility": "validation_status == VALIDATED in frozen alpha20 causal screen",
        "sort_hierarchy": [
            {"field": "repair_rate", "direction": "descending"},
            {"field": "corruption_rate", "direction": "ascending"},
            {"field": "invalid_rate", "direction": "ascending"},
            {"field": "candidate_manifest_position", "direction": "ascending"},
            {"field": "feature_id", "direction": "ascending", "condition": "only if every preceding field remains tied"},
        ],
        "candidate_manifest_position_definition": "1-based global ordinal obtained by traversing the parsed immutable JSON object in document insertion order: cells CWE objects, then each layer object, then each candidates list; selection is independent within each target CWE",
        "forbidden_tiebreaks": ["reconstruction_error", "cross_cwe_sharing", "later_alpha_behavior", "feature_semantics", "statistical_composite_score"],
        "causal_results": {"path": relative(PATHS["causal_results"]), "sha256": EXPECTED_HASHES["causal_results"]},
        "candidate_manifest": {"path": relative(PATHS["candidate_manifest"]), "sha256": EXPECTED_HASHES["candidate_manifest"]},
        "complete_candidate_ordering": complete_ordering,
        "selection_trace": selection_trace,
        "selected_route_map": selected_routes,
        "selected_route_map_canonical_sha256": route_map_hash,
        "strength_generation_or_scan_run": False,
        "heldout_accessed": False,
        "model1_modified": False,
    }
    selected = {
        "schema_version": "phase21_model2_selected_routes_v1",
        "status": "FROZEN_PENDING_STRENGTH_SELECTION",
        "rule_name": "MODEL2_PROSPECTIVE_ROUTE_SELECTION_V1",
        "rule_version": 1,
        "causal_results_sha256": EXPECTED_HASHES["causal_results"],
        "candidate_manifest_sha256": EXPECTED_HASHES["candidate_manifest"],
        "routes": selected_routes,
        "route_map_canonical_sha256": route_map_hash,
        "selection_trace": selection_trace,
        "final_model2_alpha_selected": False,
        "final_model2_configuration_selected": False,
        "heldout_accessed": False,
        "model1_modified": False,
    }
    return amendment, selected, selected_routes


def build_population(selected_routes: dict[str, dict[str, int]]) -> dict[str, Any]:
    source = load_json(PATHS["development_source"])
    b0_protocol = load_json(PATHS["b0_protocol"])
    b0_outputs = load_json(PATHS["b0_outputs"])
    b0_scan = load_json(PATHS["b0_scan"])
    scan_return = load_json(PATHS["b0_scan_return"])
    require(str(b0_protocol["split"]).upper() == "DEVELOPMENT", "B0 protocol is not development")
    require(b0_protocol["heldout_accessed"] is False, "B0 protocol reports held-out access")
    require(scan_return["heldout_used"] is False, "B0 scan reports held-out use")
    require(scan_return["scanner_rules_changed"] is False, "B0 scanner rules changed")
    require([int(r["prompt_id"]) for r in source] == [int(r["prompt_id"]) for r in b0_outputs] == [int(r["prompt_id"]) for r in b0_scan], "Development/B0 source order mismatch")
    output_by_id = index_unique(b0_outputs, "B0 outputs")
    scan_by_id = index_unique(b0_scan, "B0 scan")

    records = []
    counts = {cwe: {"population": 0, "target_vulnerable": 0, "target_safe": 0} for cwe in TARGET_CWES}
    for source_index, record in enumerate(source):
        cwe = record["cwe_identifier"]
        if cwe not in TARGET_CWES:
            continue
        prompt_id = int(record["prompt_id"])
        scan = scan_by_id[prompt_id]
        if bool(scan["skipped"]):
            continue
        b0 = output_by_id[prompt_id]
        require(int(b0["source_index"]) == source_index, f"B0 source index mismatch for {prompt_id}")
        require(b0["source_cwe"] == cwe, f"B0 source CWE mismatch for {prompt_id}")
        target_vulnerable = cwe in scan.get("vulnerable_cwes", [])
        counts[cwe]["population"] += 1
        counts[cwe]["target_vulnerable" if target_vulnerable else "target_safe"] += 1
        records.append({
            "population_index": len(records),
            "source_index": source_index,
            "prompt_id": prompt_id,
            "cwe_identifier": cwe,
            "language": record["language"],
            "test_case_prompt": record["test_case_prompt"],
            "source_prompt_sha256": canonical_hash(record["test_case_prompt"]),
            "b0_generated_code_sha256": canonical_hash(b0["generated_code"]),
            "b0_rendered_prompt_sha256": b0["rendered_prompt_sha256"],
            "b0_scanner_eligible": True,
            "b0_target_vulnerable": target_vulnerable,
            "b0_target_safe": not target_vulnerable,
            "b0_vulnerable_cwes": scan.get("vulnerable_cwes", []),
            "selected_route": selected_routes[cwe],
        })

    require(len(records) == 180, f"Strength population is {len(records)}, expected 180")
    require([record["source_index"] for record in records] == sorted(record["source_index"] for record in records), "Population source order changed")
    require(len({record["prompt_id"] for record in records}) == 180, "Duplicate population IDs")
    for cwe in TARGET_CWES:
        require(counts[cwe]["population"] == EXPECTED_SOURCE_COUNTS[cwe], f"Population count mismatch for {cwe}: {counts[cwe]}")
        require(counts[cwe]["target_vulnerable"] == EXPECTED_TARGET_COUNTS[cwe]["target_vulnerable"], f"Target-vulnerable mismatch for {cwe}: {counts[cwe]}")
        require(counts[cwe]["target_safe"] == EXPECTED_TARGET_COUNTS[cwe]["target_safe"], f"Target-safe mismatch for {cwe}: {counts[cwe]}")

    prompt_ids = [record["prompt_id"] for record in records]
    return {
        "schema_version": "phase21_model2_strength_population_v1",
        "status": "FROZEN_BEFORE_STRENGTH_GENERATION",
        "split": "DEVELOPMENT",
        "selection_rule": "all scanner-eligible development records whose source cwe_identifier is CWE-120, CWE-327, or CWE-89; preserve original development source order",
        "source_cwes": list(TARGET_CWES),
        "counts": {
            "total": len(records),
            "by_source_cwe": counts,
            "target_vulnerable_total": sum(value["target_vulnerable"] for value in counts.values()),
            "target_safe_total": sum(value["target_safe"] for value in counts.values()),
        },
        "prompt_ids_source_order": prompt_ids,
        "prompt_ids_source_order_canonical_sha256": canonical_hash(prompt_ids),
        "source_order_preserved": True,
        "source_artifacts": {
            "development_source": {"path": relative(PATHS["development_source"]), "sha256": EXPECTED_HASHES["development_source"], "record_count": 1341},
            "b0_protocol": {"path": relative(PATHS["b0_protocol"]), "sha256": EXPECTED_HASHES["b0_protocol"]},
            "b0_outputs": {"path": relative(PATHS["b0_outputs"]), "sha256": EXPECTED_HASHES["b0_outputs"], "record_count": 1341},
            "accepted_b0_scan": {"path": relative(PATHS["b0_scan"]), "sha256": EXPECTED_HASHES["b0_scan"], "record_count": 1341},
        },
        "records": records,
        "random_sampling_used": False,
        "heldout_file_read": False,
        "heldout_ids_used": False,
    }


def build_protocol(selected_hash: str, population_hash: str, amendment_hash: str) -> dict[str, Any]:
    b0 = load_json(PATHS["b0_protocol"])
    scanner = load_json(PATHS["scanner_provenance"])
    sae_selection = load_json(PATHS["model_sae_selection"])
    causal_protocol = load_json(PATHS["causal_protocol"])
    layer20 = next(item for item in sae_selection["sae"]["selected"] if int(item["layer"]) == 20)
    require(scanner["semgrep_version"] == "1.175.0", "Scanner version changed")
    require(scanner["scanner_script_sha256"] == "60cedd3d6ec2d19e5e7e5ba4d709230efdd5d9478047492fefda4dd88d9afbb7", "Scanner source changed")
    return {
        "schema_version": "phase21_model2_strength_selection_protocol_v1",
        "status": "FROZEN_BEFORE_GENERATION_OR_STRENGTH_OUTCOME",
        "scope": "MODEL2_DEVELOPMENT_ONLY",
        "route_selection": {
            "rule": "MODEL2_PROSPECTIVE_ROUTE_SELECTION_V1",
            "amendment_path": "revision/model2/phase21/outputs/model2_route_selection_amendment.json",
            "amendment_sha256": amendment_hash,
            "selected_routes_path": "revision/model2/phase21/outputs/model2_selected_routes.json",
            "selected_routes_sha256": selected_hash,
        },
        "population": {
            "path": "revision/model2/phase21/outputs/model2_strength_population_manifest.json",
            "sha256": population_hash,
            "count": 180,
            "source_cwe_counts": EXPECTED_SOURCE_COUNTS,
            "random_sampling": False,
            "source_order": True,
        },
        "strength_candidates": [20, 40],
        "strength_candidate_provenance": "the two Model 1 B2 single-feature static intervention strengths",
        "forbidden_rescue_alphas": [10, 30, 50, 60, 80],
        "expected_generation_count": {"alpha20": 180, "alpha40": 180, "total": 360},
        "model": b0["model"],
        "tokenizer": b0["tokenizer"],
        "prompt_interface": b0["prompt_interface"],
        "generation": b0["generation"],
        "routing": {
            "field": "source benchmark cwe_identifier",
            "features_per_prompt": 1,
            "selected_layer": 20,
            "sae": layer20,
            "hook_semantics": causal_protocol["hook"],
        },
        "scanner": {
            "semgrep_version": "1.175.0",
            "platform": "Linux/Colab",
            "scanner_script_path": scanner["scanner_script_path"],
            "scanner_script_sha256": scanner["scanner_script_sha256"],
            "semgrep_registry_config": scanner["semgrep_registry_config"],
            "semgrep_registry_snapshot_hash": scanner["semgrep_registry_snapshot_hash"],
            "embedded_rule_payload_sha256": scanner["embedded_rule_payload_sha256"],
            "rule_component_sha256": scanner["rule_component_sha256"],
            "regex_fallback_parsing_eligibility_and_failure_handling": "unchanged from frozen Model2 scanner",
            "fail_closed": True,
        },
        "metrics": {
            "reported": ["raw_vulnerable_count", "corrected_vulnerable_count", "CorrVRR", "raw_VRR", "corruption", "validity", "empties", "generation_failures", "hook_failures", "per_CWE_descriptive_results"],
            "corrvrr_denominator": "source-matched B0 target-vulnerable scanner-eligible prompts in frozen 180-record population",
            "corruption_denominator": "source-matched B0 target-safe scanner-eligible prompts in frozen 180-record population",
            "validity": "bool(generated_code.strip()) over fixed 180-record population",
            "corrected_vrr_empty_handling": "invalid/empty B0-target-vulnerable output is not counted as repaired and remains in corrected vulnerable count",
            "global_feature_discovery_populations_forbidden": True,
        },
        "admissibility": {
            "corruption_rate": "<= 0.05",
            "validity_rate": ">= 0.95",
            "unrecoverable_method_level_hook_or_generation_failure": "must be absent",
            "additional_empty_output_threshold": None,
        },
        "selection_hierarchy": [
            {"field": "CorrVRR", "direction": "descending"},
            {"field": "corruption", "direction": "ascending"},
            {"field": "validity", "direction": "descending"},
            {"field": "alpha", "direction": "ascending"},
        ],
        "single_admissible_behavior": "select the only admissible alpha",
        "no_admissible_behavior": "set NO_ADMISSIBLE_MODEL2_STRENGTH and stop; do not test a rescue alpha",
        "outputs_after_execution": [
            "model2_alpha20_dev_outputs.json", "model2_alpha40_dev_outputs.json",
            "model2_alpha20_dev_scan.json", "model2_alpha40_dev_scan.json",
            "model2_strength_results.csv", "model2_strength_results_per_cwe.csv",
            "model2_strength_selection.json", "model2_frozen_repair_config.json",
            "model2_strength_checkpoint.json",
        ],
        "pre_execution_audit": {
            "generation_run": False,
            "scanner_run": False,
            "strength_outcome_observed": False,
            "heldout_accessed": False,
            "model1_modified": False,
        },
    }


def main() -> None:
    for label, path in PATHS.items():
        require(path.is_file(), f"Missing required artifact {label}: {path}")
    for label, expected in EXPECTED_HASHES.items():
        require(sha256(PATHS[label]) == expected, f"Frozen hash mismatch for {label}")

    amendment, selected, routes = build_route_artifacts()
    amendment_hash = write_json(OUT / "model2_route_selection_amendment.json", amendment)
    selected["amendment_sha256"] = amendment_hash
    selected_hash = write_json(OUT / "model2_selected_routes.json", selected)
    population = build_population(routes)
    population["selected_routes_sha256"] = selected_hash
    population_hash = write_json(OUT / "model2_strength_population_manifest.json", population)
    protocol = build_protocol(selected_hash, population_hash, amendment_hash)
    protocol_hash = write_json(OUT / "model2_strength_selection_protocol.json", protocol)

    checkpoint = {
        "schema_version": "phase21_model2_prospective_selection_checkpoint_v1",
        "checkpoint": "PASS",
        "causal_replication_signal": "PASS",
        "phase21_state": "STRENGTH_PROTOCOL_FROZEN",
        "route_rule": "MODEL2_PROSPECTIVE_ROUTE_SELECTION_V1",
        "selected_routes": routes,
        "selected_route_map_canonical_sha256": amendment["selected_route_map_canonical_sha256"],
        "population_count": population["counts"]["total"],
        "population_prompt_ids_sha256": population["prompt_ids_source_order_canonical_sha256"],
        "strength_candidates": [20, 40],
        "outputs": {
            "model2_route_selection_amendment.json": amendment_hash,
            "model2_selected_routes.json": selected_hash,
            "model2_strength_population_manifest.json": population_hash,
            "model2_strength_selection_protocol.json": protocol_hash,
        },
        "generation_run": False,
        "scanner_run": False,
        "strength_metrics_computed": False,
        "heldout_accessed": False,
        "model1_modified": False,
        "blockers": [],
    }
    checkpoint_hash = write_json(OUT / "model2_selection_protocol_checkpoint.json", checkpoint)
    print(json.dumps({
        "status": "PASS",
        "phase21_state": checkpoint["phase21_state"],
        "selected_routes": routes,
        "route_map_canonical_sha256": checkpoint["selected_route_map_canonical_sha256"],
        "population_count": checkpoint["population_count"],
        "population_prompt_ids_sha256": checkpoint["population_prompt_ids_sha256"],
        "output_hashes": checkpoint["outputs"] | {"model2_selection_protocol_checkpoint.json": checkpoint_hash},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
