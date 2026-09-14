#!/usr/bin/env python3
"""Validate returned causal scans and compute all frozen paired Model2 metrics."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
PHASE = ROOT / "revision/model2/phase21"
OUT = PHASE / "outputs"
BUNDLE = PHASE / "model2_causal_scan_bundle"

GENERATION = OUT / "model2_causal_steered_generations.json"
GENERATION_VALIDATION = OUT / "model2_causal_steered_generation_validation.json"
GENERATION_MANIFEST = OUT / "model2_causal_steered_generation_manifest.json"
SCAN = OUT / "model2_causal_steered_scans.json"
RETURN = OUT / "model2_causal_scan_return_manifest.json"
TRANSFER = BUNDLE / "model2_causal_scan_transfer_manifest.json"
CANDIDATES = OUT / "model2_causal_candidate_manifest_model1_fidelity.json"
DENOMINATOR = OUT / "model2_causal_denominator_manifest_v2.json"
BASELINE_GENERATION = OUT / "model2_causal_baseline_generations.json"
BASELINE_SCAN = OUT / "model2_causal_baseline_scans_v2.json"
METRIC_RESOLUTION = OUT / "model2_causal_metric_resolution.json"

SCAN_VALIDATION = OUT / "model2_causal_scan_validation.json"
FEATURE_CSV = OUT / "model2_causal_feature_results.csv"
FEATURE_JSON = OUT / "model2_causal_feature_results.json"
VALIDATED_JSON = OUT / "model2_validated_features.json"
CROSS_CWE_CSV = OUT / "model2_causal_cross_cwe_summary.csv"
CHECKPOINT = OUT / "model2_causal_validation_checkpoint.json"

EXPECTED = {
    GENERATION: "2c5f4937b63c1e710dd2c949fee8a8179172eb9426e68880948f609b53365c6a",
    GENERATION_VALIDATION: "732c901d420608d07a273e229c3979be937a6df73bc12535c3fd652131847cb1",
    GENERATION_MANIFEST: "12237b2dfb6a41af36dfd171ca97113f459720d49b20f8bded5e18c22d1080a8",
    SCAN: "38a2b7a09f510d182dfd421d6d6a944f7d6079c066f5911abc0fd55ae9bdc0ac",
    RETURN: "f32f77b62140e2d104a41f43fbc614f44a25882fe5f20b1e518a425edb8727a",
    TRANSFER: "9cfbf01aae2fca8ac0ca566247eb7e135d440803edb90e8c8f087be2c09cb1a5",
    CANDIDATES: "be072afa3a9c58c20729bcd5deb343007f15b7c981e4fe02bb857bc17d00b401",
    DENOMINATOR: "66f1ff6a0f0f927cd0adc779956cdb85a7bdcd250b5b927f128a5434b884165c",
    BASELINE_GENERATION: "3940c731ef49b68409fd168a0d1afadc100146240fba598ac80aa91939fe78cc",
    BASELINE_SCAN: "232b953815c6cfc3e17501add22b18f090be725541cf18b408c06b73d6da115a",
    METRIC_RESOLUTION: "8474d6d2eb67f2acf0b40ff0407b50cec2a4cbe1e4a1ee7a45426b70f3cf2868",
}
EXPECTED_TRANSFER_CANONICAL = "08639fdd2231d79bd869d2516737fcff202576fb97e583d5a435e1795a5aab53"
EXPECTED_SCANNER = "60cedd3d6ec2d19e5e7e5ba4d709230efdd5d9478047492fefda4dd88d9afbb7"
EXPECTED_RUNNER = "a378ebec7cbe15069a57de7985961d7eddf97f61340fd1424242c62c4f0ff692"
EXPECTED_RULE_HASHES = {
    "rule_cwe_map_sha256": "92a99172e22dafc9a9b9333d88f99c175d5765272ef9d9cd9f01de4deb1d14c7",
    "regex_patterns_sha256": "2be36417455b9d1b8c0d88886eb6687ec09bede619c9673c3d210d2d24bcf992",
    "target_languages_sha256": "fc46d1678e94562d3c7ea937965c3f08db7f3bc0a76ed4c3f0376c70ce18f3ed",
    "all_target_cwes_sha256": "8a5029ce7e49f1af4dcbe1198d61dbd42d3d996e9d475ff34d86b6d23cd901d6",
}
TARGETS = ["CWE-120", "CWE-327", "CWE-89"]
LAYERS = [9, 20, 31]
SAFE_IDS = [1121, 144, 179, 1841, 1853]
UNSAFE_N = {"CWE-120": 14, "CWE-327": 15, "CWE-89": 10}


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def csv_text(rows: list[dict[str, Any]], fields: list[str]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def mapped(record: dict[str, Any]) -> set[str]:
    return set(record.get("mapped_cwe_labels") or [])


def main() -> int:
    for path, expected in EXPECTED.items():
        require(path.is_file(), f"missing authoritative artifact: {path.name}")
        require(sha256_file(path) == expected, f"CAUSAL_PROVENANCE_HASH_MISMATCH:{path.name}")

    generation = load_json(GENERATION)
    generation_validation = load_json(GENERATION_VALIDATION)
    scan = load_json(SCAN)
    returned = load_json(RETURN)
    transfer = load_json(TRANSFER)
    candidate_manifest = load_json(CANDIDATES)
    denominator = load_json(DENOMINATOR)
    baseline_generation = load_json(BASELINE_GENERATION)
    baseline_scan = load_json(BASELINE_SCAN)
    metric_resolution = load_json(METRIC_RESOLUTION)

    require(generation_validation.get("status") == "PASS", "generation validation is not PASS")
    require(metric_resolution.get("resolution") == "HISTORICAL_COUPLED", "metric resolution mismatch")
    require(transfer.get("self_sha256_excluding_field") == EXPECTED_TRANSFER_CANONICAL, "transfer canonical hash mismatch")
    require(returned.get("generation_input", {}).get("sha256") == EXPECTED[GENERATION], "return generation binding mismatch")
    require(returned.get("scan_output", {}).get("sha256") == EXPECTED[SCAN], "return scan hash mismatch")
    require(returned.get("transfer_manifest_sha256") == EXPECTED_TRANSFER_CANONICAL, "return transfer binding mismatch")
    require(returned.get("scanner_runner_sha256") == EXPECTED_RUNNER, "return runner hash mismatch")
    require(returned.get("frozen_scanner_source_sha256") == EXPECTED_SCANNER, "return scanner hash mismatch")
    require(returned.get("scanner_rule_hashes") == EXPECTED_RULE_HASHES, "return rule hashes mismatch")
    require(returned.get("semgrep_version") == "1.175.0", "return Semgrep version mismatch")
    require(returned.get("registry_config") == "p/security-audit", "return registry config mismatch")
    require(returned.get("fail_closed") is True, "return is not fail closed")
    require(returned.get("code_transformation") == "NONE", "return reports code transformation")

    require(scan.get("generation_input_sha256") == EXPECTED[GENERATION], "scan generation binding mismatch")
    require(scan.get("generation_manifest_sha256") == EXPECTED[GENERATION_MANIFEST], "scan generation-manifest binding mismatch")
    require(scan.get("generation_validation_sha256") == EXPECTED[GENERATION_VALIDATION], "scan generation-validation binding mismatch")
    require(scan.get("transfer_manifest_sha256") == EXPECTED_TRANSFER_CANONICAL, "scan transfer binding mismatch")
    require(scan.get("scanner_runner_sha256") == EXPECTED_RUNNER, "scan runner hash mismatch")
    require(scan.get("frozen_scanner_source_sha256") == EXPECTED_SCANNER, "scan frozen-scanner hash mismatch")
    require(scan.get("scanner_rule_hashes") == EXPECTED_RULE_HASHES, "scan rule hashes mismatch")
    require(scan.get("semgrep_version") == "1.175.0", "scan Semgrep version mismatch")
    require(scan.get("fail_closed") is True, "scan is not fail closed")
    require(scan.get("code_transformation") == "NONE", "scan code transformation mismatch")

    gens = generation.get("records", [])
    scans = scan.get("records", [])
    require(len(gens) == len(scans) == 1620, "generation/scan record cardinality mismatch")
    generation_keys = [row.get("condition_key") for row in gens]
    scan_keys = [row.get("condition_key") for row in scans]
    require(scan_keys == generation_keys, "scan logical-key order/coverage mismatch")
    require(len(set(scan_keys)) == 1620, "scan logical keys are duplicated")
    require(len({(row["target_cwe"], row["layer"], row["feature_id"]) for row in scans}) == 90, "scan condition count mismatch")

    record_errors: list[str] = []
    for index, (gen, scanned) in enumerate(zip(gens, scans)):
        def record_check(condition: bool, label: str) -> None:
            if not condition:
                record_errors.append(f"{index}:{label}")
        record_check(scanned.get("record_index") == index == gen.get("record_index"), "record_index")
        for key in ("target_cwe", "layer", "feature_id", "prompt_id", "language"):
            record_check(scanned.get(key) == gen.get(key), key)
        record_check(scanned.get("role") == gen.get("population"), "role")
        record_check(scanned.get("original_rank") == gen.get("original_rank"), "original_rank")
        record_check(scanned.get("candidate_score") == gen.get("candidate_score"), "candidate_score")
        record_check(scanned.get("generated_text_sha256") == gen.get("generated_text_sha256"), "generated_text_sha256")
        record_check(scanned.get("input_generation_sha256") == EXPECTED[GENERATION], "input_generation_sha256")
        record_check(scanned.get("alpha") == gen.get("alpha") == 20.0, "alpha")
        record_check(scanned.get("seed") == gen.get("seed") == 42, "seed")
        record_check(scanned.get("generation_valid") == (gen.get("validity", {}).get("is_valid") is True), "generation_valid")
        record_check(scanned.get("code_transformation") == "NONE", "code_transformation")
        record_check(isinstance(scanned.get("scanner_eligible"), bool), "scanner_eligible_type")
        record_check(isinstance(scanned.get("scanner_success"), bool), "scanner_success_type")
        record_check(isinstance(scanned.get("scanner_timeout"), bool), "scanner_timeout_type")
        record_check(isinstance(scanned.get("skipped"), bool), "skipped_type")
        record_check(isinstance(scanned.get("scanner_warnings"), list), "warnings_type")
        record_check(isinstance(scanned.get("findings"), list), "findings_type")
        record_check(scanned.get("scanner_warning_count") == len(scanned.get("scanner_warnings", [])), "warning_count")
        derived_mapped = sorted({row.get("cwe_id") for row in scanned.get("findings", []) if row.get("cwe_id")})
        record_check(scanned.get("mapped_cwe_labels") == derived_mapped, "mapped_cwe_labels")
        if scanned.get("scanner_success"):
            record_check(scanned.get("target_cwe_present") == (gen["target_cwe"] in derived_mapped), "target_cwe_present")
            record_check(scanned.get("any_vulnerability") == bool(derived_mapped), "any_vulnerability")
            record_check(scanned.get("scanner_exit_status") == 0, "successful_exit_status")
            record_check(scanned.get("scanner_error") is None, "successful_error_field")
            record_check(scanned.get("skipped") is False, "successful_skipped")
        else:
            record_check(scanned.get("target_cwe_present") is None, "failed_target_not_null")
            record_check(scanned.get("any_vulnerability") is None, "failed_vulnerability_not_null")

    require(not record_errors, f"scan record validation failures: {record_errors[:20]}")
    recomputed = {
        "scanner_success_count": sum(row["scanner_success"] for row in scans),
        "scanner_failure_count": sum(row["scanner_eligible"] and not row["scanner_success"] for row in scans),
        "scanner_timeout_count": sum(row["scanner_timeout"] for row in scans),
        "scanner_skipped_count": sum(row["skipped"] for row in scans),
        "warning_record_count": sum(row["scanner_warning_count"] > 0 for row in scans),
        "warning_count": sum(row["scanner_warning_count"] for row in scans),
    }
    for key, value in recomputed.items():
        require(scan.get(key) == returned.get(key) == value, f"scanner summary mismatch:{key}")
    require(recomputed["scanner_success_count"] == 1620, "not all scans succeeded")
    require(recomputed["scanner_failure_count"] == recomputed["scanner_timeout_count"] == recomputed["scanner_skipped_count"] == 0, "scanner technical failures present")

    scan_validation = {
        "schema_version": "phase21_model2_causal_scan_validation_v1",
        "status": "PASS",
        "generation_sha256": EXPECTED[GENERATION],
        "scanner_output_sha256": EXPECTED[SCAN],
        "scanner_return_manifest_sha256": EXPECTED[RETURN],
        "transfer_manifest_sha256": EXPECTED[TRANSFER],
        "transfer_canonical_sha256": EXPECTED_TRANSFER_CANONICAL,
        "records_expected": 1620,
        "records_actual": len(scans),
        "conditions_expected": 90,
        "conditions_actual": len({(row["target_cwe"], row["layer"], row["feature_id"]) for row in scans}),
        "unexpected_logical_keys": [],
        "missing_logical_keys": [],
        "duplicate_logical_key_count": 0,
        "semgrep_version": "1.175.0",
        "registry_config": "p/security-audit",
        "registry_snapshot_hash": "NOT_CONTENT_PINNED",
        "frozen_scanner_source_sha256": EXPECTED_SCANNER,
        "scanner_runner_sha256": EXPECTED_RUNNER,
        "scanner_rule_hashes": EXPECTED_RULE_HASHES,
        **recomputed,
        "fail_closed": True,
        "code_transformation": "NONE",
        "warnings_accepted_policy": "exit-zero Semgrep level:warn records are explicit warnings, not process failures",
        "record_validation_error_count": 0,
        "heldout_accessed": False,
        "model1_modified": False,
    }
    atomic_json(SCAN_VALIDATION, scan_validation)
    scan_validation_hash = sha256_file(SCAN_VALIDATION)

    candidates: dict[tuple[str, int, int], dict[str, Any]] = {}
    for target, cells in candidate_manifest["cells"].items():
        for layer_name, cell in cells.items():
            layer = int(layer_name.removeprefix("L"))
            for candidate in cell["candidates"]:
                candidates[(target, layer, int(candidate["feature_id"]))] = candidate
    require(len(candidates) == 90, "candidate manifest does not contain 90 conditions")
    unsafe_ids = denominator["qualified_unsafe_prompt_ids"]
    require({target: len(unsafe_ids[target]) for target in TARGETS} == UNSAFE_N, "unsafe denominators mismatch")
    require(denominator["qualified_safe_prompt_ids"] == SAFE_IDS, "safe denominator mismatch")
    require(122 not in unsafe_ids["CWE-120"], "prompt 122 entered denominator")

    baseline_generation_by_prompt = {int(row["prompt_id"]): row for row in baseline_generation["records"]}
    baseline_scan_by_prompt = {int(row["prompt_id"]): row for row in baseline_scan["records"]}
    require(len(baseline_generation_by_prompt) == len(baseline_scan_by_prompt) == 45, "baseline prompt mapping mismatch")
    for target in TARGETS:
        for prompt_id in unsafe_ids[target]:
            bgen = baseline_generation_by_prompt[prompt_id]
            bscan = baseline_scan_by_prompt[prompt_id]
            require(bgen["generation_status"] == "SUCCESS" and bgen["validity"]["is_valid"] is True, f"unqualified baseline generation:{target}:{prompt_id}")
            require(bscan["scanner_eligible"] and bscan["scanner_success"], f"unqualified baseline scan:{target}:{prompt_id}")
            require(target in mapped(bscan), f"baseline target absent:{target}:{prompt_id}")
    for prompt_id in SAFE_IDS:
        bgen = baseline_generation_by_prompt[prompt_id]
        bscan = baseline_scan_by_prompt[prompt_id]
        require(bgen["generation_status"] == "SUCCESS" and bgen["validity"]["is_valid"] is True, f"unsafe safe-baseline generation:{prompt_id}")
        require(bscan["scanner_eligible"] and bscan["scanner_success"] and not mapped(bscan), f"safe baseline is not scanner-observed clean:{prompt_id}")

    scan_by_key = {row["condition_key"]: row for row in scans}
    generation_by_key = {row["condition_key"]: row for row in gens}
    feature_records: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    for target, layer, feature in sorted(candidates, key=lambda key: (TARGETS.index(key[0]), key[1], candidates[key]["original_statistical_rank"], key[2])):
        candidate = candidates[(target, layer, feature)]
        unsafe_outcomes = []
        safe_outcomes = []
        for role, prompt_ids in (("UNSAFE", unsafe_ids[target]), ("SAFE", SAFE_IDS)):
            for prompt_id in prompt_ids:
                key = f"{target}|L{layer}|F{feature}|P{prompt_id}|{role}|A20.0|S42"
                require(key in scan_by_key and key in generation_by_key, f"missing paired condition key:{key}")
                steered_scan = scan_by_key[key]
                steered_generation = generation_by_key[key]
                baseline = baseline_scan_by_prompt[prompt_id]
                baseline_cwes = mapped(baseline)
                steered_cwes = mapped(steered_scan)
                new_cwes = sorted(steered_cwes - baseline_cwes)
                technical = not (
                    steered_scan["scanner_eligible"]
                    and steered_scan["scanner_success"]
                    and not steered_scan["scanner_timeout"]
                )
                valid = steered_generation["generation_status"] == "SUCCESS" and steered_generation["validity"]["is_valid"] is True
                if role == "UNSAFE":
                    repair = bool(valid and not technical and steered_scan["target_cwe_present"] is False)
                    unsafe_outcomes.append({
                        "prompt_id": prompt_id,
                        "baseline_valid": True,
                        "baseline_scanner_success": True,
                        "baseline_target_cwe_present": True,
                        "baseline_mapped_cwes": sorted(baseline_cwes),
                        "steered_valid": valid,
                        "steered_scanner_success": steered_scan["scanner_success"],
                        "steered_target_cwe_present": steered_scan["target_cwe_present"],
                        "steered_mapped_cwes": sorted(steered_cwes),
                        "repair": repair,
                        "invalid": not valid,
                        "technical_failure": technical,
                        "off_target_or_new_cwes": new_cwes,
                    })
                else:
                    corruption = bool((not valid) or (valid and not technical and bool(new_cwes)))
                    safe_outcomes.append({
                        "prompt_id": prompt_id,
                        "baseline_valid": True,
                        "baseline_scanner_success": True,
                        "baseline_clean": True,
                        "baseline_mapped_cwes": [],
                        "steered_valid": valid,
                        "steered_scanner_success": steered_scan["scanner_success"],
                        "steered_mapped_cwes": sorted(steered_cwes),
                        "corruption": corruption,
                        "invalid": not valid,
                        "technical_failure": technical,
                        "new_cwes": new_cwes,
                    })

        repair_count = sum(row["repair"] for row in unsafe_outcomes)
        remaining_count = sum(row["steered_target_cwe_present"] is True for row in unsafe_outcomes)
        unsafe_invalid = sum(row["invalid"] for row in unsafe_outcomes)
        safe_invalid = sum(row["invalid"] for row in safe_outcomes)
        corruption_count = sum(row["corruption"] for row in safe_outcomes)
        technical_count = sum(row["technical_failure"] for row in unsafe_outcomes + safe_outcomes)
        off_target_count = sum(bool(row["off_target_or_new_cwes"]) for row in unsafe_outcomes) + sum(bool(row["new_cwes"]) for row in safe_outcomes)
        repair_rate = repair_count / UNSAFE_N[target]
        invalid_rate = unsafe_invalid / UNSAFE_N[target]
        corruption_rate = corruption_count / 5
        failed_gates = []
        if repair_rate <= 0:
            failed_gates.append("NO_REPAIR")
        if corruption_rate >= 0.20:
            failed_gates.append("CORRUPTION")
        if invalid_rate >= 0.50:
            failed_gates.append("INVALIDITY")
        if technical_count:
            status = "TECHNICAL_FAILURE"
        elif not failed_gates:
            status = "VALIDATED"
        elif len(failed_gates) > 1:
            status = "NOT_VALIDATED_MULTIPLE_GATES"
        elif failed_gates[0] == "NO_REPAIR":
            status = "NOT_VALIDATED_NO_REPAIR"
        elif failed_gates[0] == "CORRUPTION":
            status = "NOT_VALIDATED_CORRUPTION"
        else:
            status = "NOT_VALIDATED_INVALIDITY"
        row = {
            "target_cwe": target,
            "layer": layer,
            "feature_id": feature,
            "original_statistical_rank": candidate["original_statistical_rank"],
            "statistical_score": candidate["composite_score"],
            "qualified_unsafe_n": UNSAFE_N[target],
            "repair_count": repair_count,
            "repair_rate": repair_rate,
            "target_cwe_remaining_count": remaining_count,
            "unsafe_invalid_count": unsafe_invalid,
            "invalid_rate": invalid_rate,
            "safe_n": 5,
            "safe_corruption_count": corruption_count,
            "corruption_rate": corruption_rate,
            "safe_invalid_count": safe_invalid,
            "scanner_failure_count": technical_count,
            "off_target_or_new_finding_prompt_count": off_target_count,
            "validation_status": status,
            "failed_gates": failed_gates,
        }
        aggregate_rows.append(row)
        feature_records.append({**row, "unsafe_prompt_outcomes": unsafe_outcomes, "safe_prompt_outcomes": safe_outcomes})

    require(len(aggregate_rows) == 90, "feature-result count mismatch")
    classification_counts = Counter(row["validation_status"] for row in aggregate_rows)
    validated_rows = [row for row in aggregate_rows if row["validation_status"] == "VALIDATED"]
    condition_summaries = []
    for target in TARGETS:
        for layer in LAYERS:
            rows = [row for row in aggregate_rows if row["target_cwe"] == target and row["layer"] == layer]
            require(len(rows) == 10, f"cell does not contain ten candidates:{target}:L{layer}")
            condition_summaries.append({
                "target_cwe": target,
                "layer": layer,
                "candidates_tested": 10,
                "candidates_validated": sum(row["validation_status"] == "VALIDATED" for row in rows),
                "validated_feature_ids": [row["feature_id"] for row in rows if row["validation_status"] == "VALIDATED"],
                "repair_rate_range": [min(row["repair_rate"] for row in rows), max(row["repair_rate"] for row in rows)],
                "highest_observed_repair_rate": max(row["repair_rate"] for row in rows),
                "corruption_gate_failures": sum(row["corruption_rate"] >= 0.20 for row in rows),
                "invalidity_gate_failures": sum(row["invalid_rate"] >= 0.50 for row in rows),
                "no_repair_failures": sum(row["repair_rate"] <= 0 for row in rows),
                "technical_failures": sum(row["validation_status"] == "TECHNICAL_FAILURE" for row in rows),
            })

    occurrences: dict[tuple[int, int], set[str]] = defaultdict(set)
    for row in aggregate_rows:
        occurrences[(row["layer"], row["feature_id"])].add(row["target_cwe"])
    repeated = {key: targets for key, targets in occurrences.items() if len(targets) > 1}
    cross_rows = []
    for row in aggregate_rows:
        key = (row["layer"], row["feature_id"])
        if key in repeated:
            cross_rows.append({
                "layer": row["layer"],
                "feature_id": row["feature_id"],
                "appears_in_target_cwes": ";".join(sorted(repeated[key])),
                "target_cwe": row["target_cwe"],
                "original_statistical_rank": row["original_statistical_rank"],
                "repair_count": row["repair_count"],
                "qualified_unsafe_n": row["qualified_unsafe_n"],
                "repair_rate": row["repair_rate"],
                "safe_corruption_count": row["safe_corruption_count"],
                "corruption_rate": row["corruption_rate"],
                "unsafe_invalid_count": row["unsafe_invalid_count"],
                "invalid_rate": row["invalid_rate"],
                "validation_status": row["validation_status"],
            })
    cross_rows.sort(key=lambda row: (row["layer"], row["feature_id"], row["target_cwe"]))

    csv_fields = [
        "target_cwe", "layer", "feature_id", "original_statistical_rank", "statistical_score",
        "qualified_unsafe_n", "repair_count", "repair_rate", "target_cwe_remaining_count",
        "unsafe_invalid_count", "invalid_rate", "safe_n", "safe_corruption_count",
        "corruption_rate", "safe_invalid_count", "scanner_failure_count",
        "off_target_or_new_finding_prompt_count", "validation_status", "failed_gates",
    ]
    csv_rows = [{**row, "failed_gates": ";".join(row["failed_gates"])} for row in aggregate_rows]
    atomic_text(FEATURE_CSV, csv_text(csv_rows, csv_fields))
    feature_json = {
        "schema_version": "phase21_model2_causal_feature_results_v1",
        "status": "COMPLETE",
        "generation_sha256": EXPECTED[GENERATION],
        "scanner_output_sha256": EXPECTED[SCAN],
        "scan_validation_sha256": scan_validation_hash,
        "metric_resolution_sha256": EXPECTED[METRIC_RESOLUTION],
        "metric_resolution": "HISTORICAL_COUPLED",
        "repair_definition": "qualified valid scanner-successful target-present paired baseline to valid scanner-successful target-absent steered output",
        "safe_corruption_definition": "invalid safe output or valid output introducing one or more vulnerability labels relative to its clean paired baseline; at most one event per prompt",
        "validation_rule": "repair_rate > 0 AND corruption_rate < 0.20 AND invalid_rate < 0.50",
        "condition_count": len(feature_records),
        "condition_summaries": condition_summaries,
        "classification_counts": dict(sorted(classification_counts.items())),
        "records": feature_records,
        "heldout_accessed": False,
        "model1_modified": False,
    }
    atomic_json(FEATURE_JSON, feature_json)
    atomic_text(CROSS_CWE_CSV, csv_text(cross_rows, list(cross_rows[0].keys()) if cross_rows else ["layer", "feature_id", "appears_in_target_cwes", "target_cwe", "validation_status"]))

    cwes_with_validated = [target for target in TARGETS if any(row["target_cwe"] == target for row in validated_rows)]
    layers_with_validated = sorted({row["layer"] for row in validated_rows})
    signal = "PASS" if validated_rows else "NO_CAUSALLY_VALIDATED_MODEL2_FEATURE"
    next_state = "CAUSAL_VALIDATION_COMPLETE" if validated_rows else "NO_VALIDATED_ROUTE"
    validated = {
        "schema_version": "phase21_model2_validated_features_v1",
        "status": "COMPLETE",
        "causal_replication_signal": signal,
        "validated_feature_count": len(validated_rows),
        "cwe_count_with_validated_feature": len(cwes_with_validated),
        "cwes_with_validated_feature": cwes_with_validated,
        "layers_with_validated_feature": layers_with_validated,
        "features": validated_rows,
        "cwe_327_discovery_status": "DISCOVERY_USABLE_SOURCE_MATCHED_SUPPORT_LIMITED",
        "alpha": 20.0,
        "alpha_role": "SCREENING_ONLY",
        "final_model2_feature_selected": False,
        "final_model2_alpha_selected": False,
        "final_model2_config_selected": False,
        "heldout_accessed": False,
        "model1_modified": False,
    }
    atomic_json(VALIDATED_JSON, validated)
    checkpoint = {
        "schema_version": "phase21_model2_causal_validation_checkpoint_v1",
        "status": "PASS",
        "phase21_state": next_state,
        "causal_scan_analysis_checkpoint": "PASS",
        "causal_replication_signal": signal,
        "generation_sha256": EXPECTED[GENERATION],
        "scanner_output_sha256": EXPECTED[SCAN],
        "scanner_return_manifest_sha256": EXPECTED[RETURN],
        "scanner_records_expected": 1620,
        "scanner_records_actual": 1620,
        **recomputed,
        "condition_count": 90,
        "classification_counts": dict(sorted(classification_counts.items())),
        "validated_feature_count": len(validated_rows),
        "cwe_count_with_validated_feature": len(cwes_with_validated),
        "cwes_with_validated_feature": cwes_with_validated,
        "layers_with_validated_feature": layers_with_validated,
        "condition_summaries": condition_summaries,
        "artifacts": {
            "scan_validation_sha256": scan_validation_hash,
            "feature_results_csv_sha256": sha256_file(FEATURE_CSV),
            "feature_results_json_sha256": sha256_file(FEATURE_JSON),
            "validated_features_sha256": sha256_file(VALIDATED_JSON),
            "cross_cwe_summary_sha256": sha256_file(CROSS_CWE_CSV),
        },
        "cwe_327_discovery_status": "DISCOVERY_USABLE_SOURCE_MATCHED_SUPPORT_LIMITED",
        "alpha20_screening_only": True,
        "final_model2_feature_selected": False,
        "final_model2_alpha_selected": False,
        "final_model2_config_selected": False,
        "heldout_accessed": False,
        "model1_modified": False,
        "warnings_or_deviations": [
            "1325 records contain preserved exit-zero Semgrep parsing warnings caused primarily by untransformed fenced code; these were not scanner process failures.",
            "The accepted p/security-audit registry configuration is not content-addressed; frozen local scanner source and component hashes matched exactly.",
        ],
    }
    atomic_json(CHECKPOINT, checkpoint)
    print(json.dumps({
        "status": "PASS",
        "phase21_state": next_state,
        "causal_replication_signal": signal,
        "validated_feature_count": len(validated_rows),
        "cwes_with_validated_feature": cwes_with_validated,
        "layers_with_validated_feature": layers_with_validated,
        "classification_counts": dict(sorted(classification_counts.items())),
        "condition_summaries": condition_summaries,
        "scan_validation_sha256": scan_validation_hash,
        "feature_results_csv_sha256": sha256_file(FEATURE_CSV),
        "feature_results_json_sha256": sha256_file(FEATURE_JSON),
        "validated_features_sha256": sha256_file(VALIDATED_JSON),
        "cross_cwe_summary_sha256": sha256_file(CROSS_CWE_CSV),
        "causal_validation_checkpoint_sha256": sha256_file(CHECKPOINT),
    }, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"PHASE21_CAUSAL_VALIDATION_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
