#!/usr/bin/env python3
"""Fail-closed immutable validation of the frozen Model2 causal generations."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "revision/model2/phase21/outputs"
OUTPUT = OUT / "model2_causal_steered_generation_validation.json"

EXPECTED_HASHES = {
    "model2_causal_candidate_manifest_model1_fidelity.json": "be072afa3a9c58c20729bcd5deb343007f15b7c981e4fe02bb857bc17d00b401",
    "model2_causal_validation_protocol.json": "b3aac3b21f52270c231e1c077adeb0cc1ef2a2ebce1f156c08a13df2695d414e",
    "model2_causal_metric_resolution.json": "8474d6d2eb67f2acf0b40ff0407b50cec2a4cbe1e4a1ee7a45426b70f3cf2868",
    "model2_causal_execution_protocol_v2.json": "ae790dd42538ac172dd7ca7b178840afa794dc371c1da015ba59a4cdecfd88f8",
    "model2_causal_execution_protocol_v3.json": "a2a0da4393a5826aa6dae9c7b2ec1c3dd08454768aa2af2ab78a730f5e8e3153",
    "model2_causal_baseline_generations.json": "3940c731ef49b68409fd168a0d1afadc100146240fba598ac80aa91939fe78cc",
    "model2_causal_baseline_scans_v2.json": "232b953815c6cfc3e17501add22b18f090be725541cf18b408c06b73d6da115a",
    "model2_causal_denominator_manifest_v2.json": "66f1ff6a0f0f927cd0adc779956cdb85a7bdcd250b5b927f128a5434b884165c",
    "model2_scanner_provenance.json": "e249982d929f13c7a0cda0211f73e4cd721331b98cf483e9b205f3c4936d64a9",
    "model2_causal_steered_generation_manifest.json": "12237b2dfb6a41af36dfd171ca97113f459720d49b20f8bded5e18c22d1080a8",
    "model2_causal_steered_generations.json": "2c5f4937b63c1e710dd2c949fee8a8179172eb9426e68880948f609b53365c6a",
}

EXPECTED_BY_TARGET = {"CWE-120": 570, "CWE-327": 600, "CWE-89": 450}
EXPECTED_UNSAFE = {"CWE-120": 14, "CWE-327": 15, "CWE-89": 10}
EXPECTED_SAFE = [1121, 144, 179, 1841, 1853]
EXPECTED_LAYERS = [9, 20, 31]
SCANNER_SHA256 = "60cedd3d6ec2d19e5e7e5ba4d709230efdd5d9478047492fefda4dd88d9afbb7"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    errors: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    provenance = []
    loaded: dict[str, Any] = {}
    for name, expected in EXPECTED_HASHES.items():
        path = OUT / name
        actual = sha256_file(path) if path.is_file() else None
        provenance.append({
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "match": actual == expected,
        })
        check(actual == expected, f"CAUSAL_PROVENANCE_HASH_MISMATCH:{name}")
        if path.is_file():
            try:
                loaded[name] = load_json(path)
            except Exception as exc:
                errors.append(f"JSON_PARSE_FAILURE:{name}:{type(exc).__name__}:{exc}")

    required = set(EXPECTED_HASHES)
    if not required.issubset(loaded):
        result = {
            "schema_version": "phase21_model2_causal_steered_generation_validation_v1",
            "status": "FAIL",
            "stop_reason": "CAUSAL_PROVENANCE_HASH_MISMATCH",
            "provenance": provenance,
            "errors": errors,
        }
        atomic_json(OUTPUT, result)
        print(json.dumps(result, indent=2))
        return 2

    candidate = loaded["model2_causal_candidate_manifest_model1_fidelity.json"]
    validation_protocol = loaded["model2_causal_validation_protocol.json"]
    metric = loaded["model2_causal_metric_resolution.json"]
    v2 = loaded["model2_causal_execution_protocol_v2.json"]
    v3 = loaded["model2_causal_execution_protocol_v3.json"]
    baseline_generation = loaded["model2_causal_baseline_generations.json"]
    baseline_scan = loaded["model2_causal_baseline_scans_v2.json"]
    denominator = loaded["model2_causal_denominator_manifest_v2.json"]
    scanner_provenance = loaded["model2_scanner_provenance.json"]
    manifest = loaded["model2_causal_steered_generation_manifest.json"]
    generation = loaded["model2_causal_steered_generations.json"]

    refs = {
        "original_causal_protocol": EXPECTED_HASHES["model2_causal_validation_protocol.json"],
        "candidate_manifest": EXPECTED_HASHES["model2_causal_candidate_manifest_model1_fidelity.json"],
        "paired_denominator_v2": EXPECTED_HASHES["model2_causal_denominator_manifest_v2.json"],
        "paired_baseline_generation": EXPECTED_HASHES["model2_causal_baseline_generations.json"],
        "paired_baseline_scan_v2": EXPECTED_HASHES["model2_causal_baseline_scans_v2.json"],
    }
    for version_name, protocol in (("v2", v2), ("v3", v3)):
        for key, expected in refs.items():
            actual = protocol.get("authoritative_inputs", {}).get(key, {}).get("sha256")
            check(actual == expected, f"{version_name.upper()}_AUTHORITATIVE_REFERENCE_MISMATCH:{key}")
        check(
            protocol.get("metric_resolution", {}).get("sha256")
            == EXPECTED_HASHES["model2_causal_metric_resolution.json"],
            f"{version_name.upper()}_METRIC_RESOLUTION_MISMATCH",
        )

    scientific_fields = [
        "authoritative_inputs", "candidate_conditions", "qualified_prompts",
        "generation", "hook", "alpha", "alpha_role", "metric_resolution",
        "metric_semantics", "retention", "scanner", "execution_order",
        "expected_workload", "execution_plan_sha256", "condition_key_sha256",
    ]
    scientific_equivalence = {key: v2.get(key) == v3.get(key) for key in scientific_fields}
    check(all(scientific_equivalence.values()), "V2_V3_SCIENTIFIC_PROTOCOL_DRIFT")
    check(v2.get("alpha") == v3.get("alpha") == 20.0, "ALPHA_PROTOCOL_MISMATCH")
    check(metric.get("resolution") == "HISTORICAL_COUPLED", "METRIC_RESOLUTION_NOT_HISTORICAL_COUPLED")
    check(scanner_provenance.get("semgrep_version") == "1.175.0", "SCANNER_VERSION_PROVENANCE_MISMATCH")
    check(scanner_provenance.get("scanner_script_sha256") == SCANNER_SHA256, "SCANNER_SOURCE_PROVENANCE_MISMATCH")
    check(scanner_provenance.get("scanner_rules_changed") is False, "SCANNER_RULE_CHANGE_RECORDED")
    check(baseline_scan.get("scanner_failure_count") == 0, "BASELINE_SCAN_FAILURE_PRESENT")
    check(baseline_scan.get("semgrep_version") == "1.175.0", "BASELINE_SEMGREP_VERSION_MISMATCH")

    qualified_unsafe = denominator.get("qualified_unsafe_prompt_ids", {})
    qualified_safe = denominator.get("qualified_safe_prompt_ids", [])
    for target, expected_n in EXPECTED_UNSAFE.items():
        check(len(qualified_unsafe.get(target, [])) == expected_n, f"DENOMINATOR_COUNT_MISMATCH:{target}")
    check(qualified_safe == EXPECTED_SAFE, "SAFE_DENOMINATOR_MISMATCH")
    check(122 not in qualified_unsafe.get("CWE-120", []), "PROMPT_122_NOT_EXCLUDED")

    candidates: dict[tuple[str, int, int], dict[str, Any]] = {}
    for target, layer_cells in candidate.get("cells", {}).items():
        for layer_name, cell in layer_cells.items():
            layer = int(str(layer_name).removeprefix("L"))
            check(cell.get("candidate_count") == 10, f"CANDIDATE_CELL_COUNT_MISMATCH:{target}:{layer}")
            for row in cell.get("candidates", []):
                key = (target, layer, int(row["feature_id"]))
                check(key not in candidates, f"DUPLICATE_CANDIDATE:{key}")
                candidates[key] = row
    check(len(candidates) == 90, "FROZEN_CANDIDATE_COUNT_NOT_90")

    baseline_by_prompt: dict[int, dict[str, Any]] = {}
    for row in baseline_generation.get("records", []):
        prompt_id = int(row["prompt_id"])
        check(prompt_id not in baseline_by_prompt, f"DUPLICATE_BASELINE_PROMPT:{prompt_id}")
        baseline_by_prompt[prompt_id] = row

    records = generation.get("records", [])
    check(generation.get("status") == "COMPLETE", "GENERATION_TOP_STATUS_NOT_COMPLETE")
    check(generation.get("record_count") == 1620, "GENERATION_DECLARED_COUNT_MISMATCH")
    check(len(records) == 1620, "GENERATION_LOGICAL_COUNT_NOT_1620")
    check(generation.get("condition_count") == 90, "GENERATION_DECLARED_CONDITION_COUNT_MISMATCH")
    check(generation.get("candidate_manifest_sha256") == EXPECTED_HASHES["model2_causal_candidate_manifest_model1_fidelity.json"], "GENERATION_CANDIDATE_BINDING_MISMATCH")
    check(generation.get("denominator_manifest_sha256") == EXPECTED_HASHES["model2_causal_denominator_manifest_v2.json"], "GENERATION_DENOMINATOR_BINDING_MISMATCH")
    check(generation.get("causal_baseline_generation_sha256") == EXPECTED_HASHES["model2_causal_baseline_generations.json"], "GENERATION_BASELINE_BINDING_MISMATCH")
    check(generation.get("execution_protocol_sha256") == EXPECTED_HASHES["model2_causal_execution_protocol_v3.json"], "GENERATION_EXECUTION_BINDING_MISMATCH")
    check(manifest.get("status") == "COMPLETE", "FINAL_GENERATION_MANIFEST_NOT_COMPLETE")
    check(manifest.get("actual_record_count") == 1620, "FINAL_MANIFEST_RECORD_COUNT_MISMATCH")
    check(manifest.get("condition_count") == 90, "FINAL_MANIFEST_CONDITION_COUNT_MISMATCH")
    check(manifest.get("generation_output", {}).get("sha256") == EXPECTED_HASHES["model2_causal_steered_generations.json"], "FINAL_MANIFEST_GENERATION_HASH_MISMATCH")
    check(manifest.get("execution_protocol", {}).get("sha256") == EXPECTED_HASHES["model2_causal_execution_protocol_v3.json"], "FINAL_MANIFEST_PROTOCOL_HASH_MISMATCH")

    logical_keys: list[str] = []
    condition_keys: list[tuple[str, int, int]] = []
    target_counts: Counter[str] = Counter()
    population_counts: Counter[str] = Counter()
    invalid_indices: list[int] = []
    unexpected_prompt_ids: list[int] = []
    malformed_indices: list[int] = []

    for index, row in enumerate(records):
        row_errors_before = len(errors)
        target = row.get("target_cwe")
        layer = row.get("layer")
        feature = row.get("feature_id")
        prompt_id = row.get("prompt_id")
        population = row.get("population")
        condition = (target, layer, feature)
        expected_candidate = candidates.get(condition)
        expected_population = None
        if target in qualified_unsafe and prompt_id in qualified_unsafe[target]:
            expected_population = "UNSAFE"
        elif prompt_id in qualified_safe:
            expected_population = "SAFE"
        else:
            unexpected_prompt_ids.append(prompt_id)

        check(row.get("record_index") == index, f"RECORD_INDEX_MISMATCH:{index}")
        check(expected_candidate is not None, f"NONFROZEN_CANDIDATE:{index}:{condition}")
        if expected_candidate is not None:
            check(row.get("original_rank") == expected_candidate.get("original_statistical_rank"), f"ORIGINAL_RANK_MISMATCH:{index}")
            check(row.get("candidate_score") == expected_candidate.get("composite_score"), f"CANDIDATE_SCORE_MISMATCH:{index}")
        check(expected_population == population, f"DENOMINATOR_ROLE_MISMATCH:{index}:{prompt_id}")
        check(prompt_id != 122, f"EXCLUDED_PROMPT_122_PRESENT:{index}")
        check(row.get("alpha") == 20.0, f"ALPHA_MISMATCH:{index}")
        check(row.get("seed") == 42, f"SEED_MISMATCH:{index}")
        check(row.get("intervention_applied") is True, f"INTERVENTION_NOT_APPLIED:{index}")
        check(row.get("generation_status") == "SUCCESS", f"GENERATION_FAILURE:{index}")
        check(row.get("hook_call_count", 0) > 0, f"NO_SUCCESSFUL_HOOK_CALL:{index}")
        check(row.get("hook_finite") is True, f"NONFINITE_HOOK:{index}")
        check((row.get("selected_feature_decoded_delta_l2") or 0) > 0, f"ZERO_FEATURE_DELTA:{index}")
        check((row.get("residual_replacement_delta_l2_max") or 0) > 0, f"ZERO_RESIDUAL_DELTA:{index}")
        check(row.get("exception") is None, f"GENERATION_EXCEPTION:{index}")
        check(row.get("fallback_status") == "NONE", f"GENERATION_FALLBACK:{index}")
        valid = row.get("validity", {}).get("is_valid") is True
        if not valid:
            invalid_indices.append(index)
        check(valid, f"INVALID_GENERATION:{index}")
        check(sha256_text(str(row.get("generated_text", ""))) == row.get("generated_text_sha256"), f"GENERATED_TEXT_HASH_MISMATCH:{index}")
        baseline = baseline_by_prompt.get(prompt_id)
        check(baseline is not None, f"BASELINE_PROMPT_MISSING:{index}:{prompt_id}")
        if baseline is not None:
            check(row.get("baseline_generated_text_sha256") == baseline.get("generated_text_sha256"), f"BASELINE_TEXT_BINDING_MISMATCH:{index}")
        check(row.get("candidate_manifest_sha256") == EXPECTED_HASHES["model2_causal_candidate_manifest_model1_fidelity.json"], f"RECORD_CANDIDATE_HASH_MISMATCH:{index}")
        check(row.get("denominator_manifest_sha256") == EXPECTED_HASHES["model2_causal_denominator_manifest_v2.json"], f"RECORD_DENOMINATOR_HASH_MISMATCH:{index}")
        check(row.get("causal_baseline_generation_sha256") == EXPECTED_HASHES["model2_causal_baseline_generations.json"], f"RECORD_BASELINE_HASH_MISMATCH:{index}")
        check(row.get("execution_protocol_sha256") == EXPECTED_HASHES["model2_causal_execution_protocol_v3.json"], f"RECORD_EXECUTION_HASH_MISMATCH:{index}")
        expected_key = f"{target}|L{layer}|F{feature}|P{prompt_id}|{population}|A20.0|S42"
        check(row.get("condition_key") == expected_key, f"LOGICAL_KEY_MISMATCH:{index}")
        logical_keys.append(row.get("condition_key"))
        condition_keys.append(condition)
        target_counts[target] += 1
        population_counts[population] += 1
        if len(errors) > row_errors_before:
            malformed_indices.append(index)

    unique_conditions = set(condition_keys)
    duplicate_keys = sorted(key for key, count in Counter(logical_keys).items() if count > 1)
    check(len(unique_conditions) == 90, "UNIQUE_CONDITION_COUNT_NOT_90")
    check(len(set(logical_keys)) == 1620, "UNIQUE_LOGICAL_KEY_COUNT_NOT_1620")
    check(not duplicate_keys, "DUPLICATE_LOGICAL_KEYS_PRESENT")
    check(dict(target_counts) == EXPECTED_BY_TARGET, f"TARGET_COUNTS_MISMATCH:{dict(target_counts)}")
    check(not unexpected_prompt_ids, "UNEXPECTED_OR_HELDOUT_PROMPT_IDS_PRESENT")
    check(not invalid_indices, "INVALID_GENERATED_OUTPUTS_PRESENT")

    per_condition_errors = []
    record_counter = Counter(condition_keys)
    for target, layer, feature in sorted(candidates):
        expected = EXPECTED_UNSAFE[target] + len(EXPECTED_SAFE)
        actual = record_counter[(target, layer, feature)]
        if actual != expected:
            per_condition_errors.append({"target_cwe": target, "layer": layer, "feature_id": feature, "expected": expected, "actual": actual})
    check(not per_condition_errors, "PER_CONDITION_CARDINALITY_MISMATCH")

    status = "PASS" if not errors else "FAIL"
    result = {
        "schema_version": "phase21_model2_causal_steered_generation_validation_v1",
        "status": status,
        "stop_reason": None if status == "PASS" else ("CAUSAL_PROVENANCE_HASH_MISMATCH" if any(x.startswith("CAUSAL_PROVENANCE_HASH_MISMATCH") for x in errors) else "GENERATION_VALIDATION_FAILURE"),
        "generation_path": "revision/model2/phase21/outputs/model2_causal_steered_generations.json",
        "generation_sha256": EXPECTED_HASHES["model2_causal_steered_generations.json"],
        "record_count_expected": 1620,
        "record_count_actual": len(records),
        "condition_count_expected": 90,
        "condition_count_actual": len(unique_conditions),
        "target_record_counts_expected": EXPECTED_BY_TARGET,
        "target_record_counts_actual": dict(target_counts),
        "population_record_counts": dict(population_counts),
        "frozen_candidate_count": len(candidates),
        "duplicate_logical_key_count": len(duplicate_keys),
        "duplicate_logical_keys": duplicate_keys,
        "invalid_generated_output_count": len(invalid_indices),
        "invalid_record_indices": invalid_indices,
        "malformed_record_count": len(set(malformed_indices)),
        "malformed_record_indices": sorted(set(malformed_indices)),
        "unexpected_or_heldout_prompt_ids": sorted(set(unexpected_prompt_ids)),
        "excluded_prompt_122_absent": 122 not in [row.get("prompt_id") for row in records],
        "alpha_values": sorted(set(row.get("alpha") for row in records)),
        "seed_values": sorted(set(row.get("seed") for row in records)),
        "intervention_applied_count": sum(row.get("intervention_applied") is True for row in records),
        "successful_hook_evidence_count": sum(row.get("hook_call_count", 0) > 0 and row.get("hook_finite") is True for row in records),
        "v2_execution_protocol_sha256": EXPECTED_HASHES["model2_causal_execution_protocol_v2.json"],
        "v3_execution_protocol_sha256": EXPECTED_HASHES["model2_causal_execution_protocol_v3.json"],
        "v2_v3_scientific_fields_equal": scientific_equivalence,
        "v3_change_scope": "HOOK_FRAMEWORK_CALL_SIGNATURE_COMPATIBILITY_ONLY",
        "causal_metric_resolution": "HISTORICAL_COUPLED",
        "qualified_unsafe_denominators": EXPECTED_UNSAFE,
        "qualified_safe_prompt_ids": EXPECTED_SAFE,
        "per_condition_cardinality_errors": per_condition_errors,
        "scanner": {
            "required_semgrep_version": "1.175.0",
            "frozen_scanner_source_sha256": SCANNER_SHA256,
            "registry_config": "p/security-audit",
            "registry_snapshot_hash": "NOT_CONTENT_PINNED",
        },
        "heldout_accessed": False,
        "model1_modified": False,
        "scanner_run": False,
        "provenance": provenance,
        "errors": errors,
    }
    atomic_json(OUTPUT, result)
    print(json.dumps(result, indent=2))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"PHASE21_CAUSAL_GENERATION_VALIDATION_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
