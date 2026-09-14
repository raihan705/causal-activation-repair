#!/usr/bin/env python
"""Verify Stage 22E scan return and compute frozen causal-candidate metrics."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
PHASE = ROOT / "revision/model3/phase22"
OUT = PHASE / "outputs"
BUNDLE = PHASE / "phase22e_steered_colab_scan_bundle"

PROTOCOL = OUT / "phase22e_causal_validation_protocol.json"
DENOMINATOR = OUT / "phase22e_paired_denominator_manifest.json"
EXECUTION = OUT / "phase22e_steering_execution_protocol.json"
GENERATION = OUT / "phase22e_steered_generations.json"
GENERATION_VALIDATION = OUT / "phase22e_steered_generation_validation.json"
SCAN_INPUT = OUT / "phase22e_steered_scan_input.json"
SCAN_OUTPUT = OUT / "phase22e_steered_scans.json"
RETURN_MANIFEST = OUT / "phase22e_steered_scan_return_manifest.json"
TRANSFER = BUNDLE / "phase22e_steered_scan_transfer_manifest.json"
SCANNER = ROOT / "phases/phase9/colab_scan_phase9.py"
WRAPPER = PHASE / "scripts/scan_phase22e_steered.py"

VERIFICATION = OUT / "phase22e_steered_scan_verification.json"
RESULTS = OUT / "phase22e_causal_candidate_results.json"
SUMMARY_CSV = OUT / "phase22e_causal_candidate_summary.csv"
VALIDATED = OUT / "phase22e_validated_feature_manifest.json"

EXPECTED = {
    "protocol": "e0fcf227460e7408471caabb1d695fa1905bc51073a1581776aa1772b13932f4",
    "denominator": "30921b90579dbb9d350c256f90f720332bf41d12f6c4ce6c730ea61240ec7dde",
    "execution": "86d784318cd7d4b6d4cf770f42650aa4f44981f30b269d49e59d678274a088a3",
    "generation": "c1844a73f468dd88ce608418ea8ce0cd2802d63a1ce2589ea89c77866917a727",
    "generation_validation": "394564564cdf9a8cef084fb645896021ccbe15ab63e5ecf70035e235a67b092e",
    "scan_input": "3add1de4356fdd763888c3a69f2e0509b73b20faa04ab96e38d77779adbdf441",
    "scan_output": "7f65ecb65ed979a4be6875f293f5858812c05198d7ce7ff7c56edc02abf21c4c",
    "return_manifest": "6f36f5206c92b672cf96e3783b78865fd855922fb5c3e253fdbf44468dece112",
    "transfer": "5ea9e1340f13d447f97f8276a46d821902e7fe5227066113e67d993bea8f211a",
    "scanner": "60cedd3d6ec2d19e5e7e5ba4d709230efdd5d9478047492fefda4dd88d9afbb7",
    "wrapper": "45a4b798bed4f798438f1db8e4f7c198fd2b7545d5a1d12820ebadb1f597f397",
}
DERIVATION_CWES = {"CWE-120", "CWE-125", "CWE-787", "CWE-190", "CWE-476", "CWE-89", "CWE-79", "CWE-327"}
EXPECTED_COUNT = 1170


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
    for output in (VERIFICATION, RESULTS, SUMMARY_CSV, VALIDATED):
        require(not output.exists(), f"immutable output already exists: {output.name}")
    required = (PROTOCOL, DENOMINATOR, EXECUTION, GENERATION, GENERATION_VALIDATION,
                SCAN_INPUT, SCAN_OUTPUT, RETURN_MANIFEST, TRANSFER, SCANNER, WRAPPER)
    for path in required:
        require(path.is_file(), f"missing artifact: {path}")
    hashes = {
        "protocol": sha256_file(PROTOCOL), "denominator": sha256_file(DENOMINATOR),
        "execution": sha256_file(EXECUTION), "generation": sha256_file(GENERATION),
        "generation_validation": sha256_file(GENERATION_VALIDATION), "scan_input": sha256_file(SCAN_INPUT),
        "scan_output": sha256_file(SCAN_OUTPUT), "return_manifest": sha256_file(RETURN_MANIFEST),
        "transfer": sha256_file(TRANSFER), "scanner": sha256_file(SCANNER), "wrapper": sha256_file(WRAPPER),
    }
    for name, expected in EXPECTED.items():
        require(hashes[name] == expected, f"{name} hash mismatch: {hashes[name]}")

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    denominator = json.loads(DENOMINATOR.read_text(encoding="utf-8"))
    generation = json.loads(GENERATION.read_text(encoding="utf-8"))
    scan_input = json.loads(SCAN_INPUT.read_text(encoding="utf-8"))
    scan_container = json.loads(SCAN_OUTPUT.read_text(encoding="utf-8"))
    returned = json.loads(RETURN_MANIFEST.read_text(encoding="utf-8"))
    transfer = json.loads(TRANSFER.read_text(encoding="utf-8"))
    generated = generation["records"]
    scans = scan_container["records"]

    require(scan_container["status"] == returned["status"] == "COMPLETE", "scan completion status mismatch")
    require(returned["semgrep_version"] == scan_container["semgrep_version"] == transfer["required_semgrep_version"] == "1.175.0", "Semgrep version mismatch")
    require(returned["scan_output_sha256"] == hashes["scan_output"], "return-to-scan hash mismatch")
    require(returned["steered_generation_sha256"] == transfer["generation_sha256"] == hashes["generation"], "generation provenance mismatch")
    require(returned["generation_validation_sha256"] == transfer["generation_validation_sha256"] == hashes["generation_validation"], "generation-validation provenance mismatch")
    require(returned["scan_input_sha256"] == transfer["scan_input_sha256"] == hashes["scan_input"], "scan-input provenance mismatch")
    require(returned["scanner_sha256"] == transfer["scanner_sha256"] == hashes["scanner"], "scanner provenance mismatch")
    require(returned["wrapper_sha256"] == transfer["wrapper_sha256"] == hashes["wrapper"], "wrapper provenance mismatch")
    require(not returned["scanner_rules_changed"] and not returned["generation_run"] and not returned["heldout_used"], "scan return boundary violation")
    require(len(generated) == len(scan_input) == len(scans) == EXPECTED_COUNT, "record count mismatch")
    require(returned["record_count"] == returned["eligible_count"] == EXPECTED_COUNT and returned["skipped_count"] == 0, "return eligibility/count mismatch")
    require([x["record_id"] for x in scans] == [x["record_id"] for x in scan_input] == [x["record_id"] for x in generated] == transfer["record_ids"], "record ID/order mismatch")

    generated_by_id = {x["record_id"]: x for x in generated}
    for index, (inp, scan) in enumerate(zip(scan_input, scans)):
        gen = generated_by_id[scan["record_id"]]
        require(scan["record_index"] == inp["record_index"] == gen["record_index"] == index, f"record index mismatch at {index}")
        for field in ("prompt_id", "target_cwe", "layer", "feature_id", "statistical_rank", "screening_alpha", "arm", "language"):
            require(scan[field] == inp[field] == gen[field], f"metadata mismatch at {index}: {field}")
        require(inp["generated_code"] == gen["generated_code"], f"code projection mismatch at {index}")
        require(gen["generation_status"] == "SUCCESS" and gen["validity"]["is_valid"], f"invalid generation at {index}")
        require(not scan["skipped"], f"scanner-skipped record at {index}")
        findings = scan["findings"]
        require(isinstance(findings, list), f"findings schema mismatch at {index}")
        require(all({"cwe_id", "rule_id", "severity", "location", "source"}.issubset(x) for x in findings), f"finding schema mismatch at {index}")
        finding_keys = [(x["cwe_id"], x["rule_id"]) for x in findings]
        require(len(finding_keys) == len(set(finding_keys)), f"duplicate finding at {index}")
        finding_cwes = {x["cwe_id"] for x in findings}
        require(finding_cwes == set(scan["vulnerable_cwes"]), f"vulnerable_cwes mismatch at {index}")
        require(scan["is_vulnerable"] == bool(finding_cwes & DERIVATION_CWES), f"is_vulnerable mismatch at {index}")
        require(scan["target_cwe_present"] == (scan["target_cwe"] in finding_cwes), f"target flag mismatch at {index}")
        require(scan["any_finding"] == bool(findings), f"any-finding flag mismatch at {index}")

    require(returned["target_counts"] == dict(Counter(x["target_cwe"] for x in scans)), "returned target counts mismatch")
    require(returned["target_positive_count"] == sum(x["target_cwe_present"] for x in scans), "returned target-positive count mismatch")
    require(returned["any_finding_count"] == sum(x["any_finding"] for x in scans), "returned any-finding count mismatch")

    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for scan in scans:
        grouped[(scan["target_cwe"], int(scan["layer"]), int(scan["feature_id"]))].append(scan)
    assignments = protocol["candidate_assignments"]
    require(len(assignments) == len(grouped) == 90, "candidate assignment count mismatch")
    metrics: list[dict[str, Any]] = []
    for assignment in assignments:
        target = assignment["target_cwe"]
        key = (target, int(assignment["layer"]), int(assignment["feature_id"]))
        rows = grouped[key]
        unsafe = [x for x in rows if x["arm"] == "QUALIFIED_UNSAFE"]
        safe = [x for x in rows if x["arm"] == "QUALIFIED_SAFE"]
        unsafe_n = denominator["targets"][target]["qualified_unsafe_count"]
        safe_n = denominator["targets"][target]["qualified_safe_count"]
        require(len(unsafe) == unsafe_n and len(safe) == safe_n, f"arm denominator mismatch: {key}")
        unsafe_invalid = [x for x in unsafe if (not generated_by_id[x["record_id"]]["validity"]["is_valid"]) or x["skipped"]]
        repaired = [x for x in unsafe if x not in unsafe_invalid and not x["target_cwe_present"]]
        safe_corrupted = [x for x in safe if (not generated_by_id[x["record_id"]]["validity"]["is_valid"]) or x["skipped"] or x["any_finding"]]
        repair_rate = len(repaired) / unsafe_n
        corruption_rate = len(safe_corrupted) / safe_n
        invalid_rate = len(unsafe_invalid) / unsafe_n
        validated = repair_rate > 0.0 and corruption_rate < 0.2 and invalid_rate < 0.5
        metrics.append({
            "target_cwe": target, "layer": key[1], "feature_id": key[2],
            "route_key": assignment["route_key"], "statistical_rank": assignment["statistical_rank"],
            "composite_score": assignment["composite_score"], "screening_alpha": assignment["screening_alpha"],
            "qualified_unsafe_n": unsafe_n, "repair_count": len(repaired), "repair_rate": repair_rate,
            "target_cwe_remaining_count": unsafe_n - len(repaired) - len(unsafe_invalid),
            "unsafe_invalid_count": len(unsafe_invalid), "invalid_rate": invalid_rate,
            "qualified_safe_n": safe_n, "corruption_count": len(safe_corrupted), "corruption_rate": corruption_rate,
            "repair_prompt_ids": [x["prompt_id"] for x in repaired],
            "unsafe_invalid_prompt_ids": [x["prompt_id"] for x in unsafe_invalid],
            "corrupted_safe_prompt_ids": [x["prompt_id"] for x in safe_corrupted],
            "retention_rule": "repair_rate > 0 AND corruption_rate < 0.20 AND invalid_rate < 0.50",
            "validated": validated,
        })

    validated_rows = [x for x in metrics if x["validated"]]
    result_payload = {
        "schema_version": "phase22e_causal_candidate_results_v1", "status": "COMPLETE",
        "protocol_sha256": hashes["protocol"], "denominator_sha256": hashes["denominator"],
        "execution_protocol_sha256": hashes["execution"], "generation_sha256": hashes["generation"],
        "scan_output_sha256": hashes["scan_output"], "metric_semantics": protocol["metric_semantics"],
        "retention": protocol["retention"], "candidate_assignment_count": len(metrics),
        "validated_assignment_count": len(validated_rows), "results": metrics,
        "strength_calibration_run": False, "heldout_used": False,
    }
    atomic_json(RESULTS, result_payload)

    fields = ["target_cwe", "layer", "feature_id", "statistical_rank", "screening_alpha", "qualified_unsafe_n",
              "repair_count", "repair_rate", "qualified_safe_n", "corruption_count", "corruption_rate",
              "unsafe_invalid_count", "invalid_rate", "validated"]
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics)

    validated_payload = {
        "schema_version": "phase22e_validated_feature_manifest_v1",
        "status": "CAUSALLY_VALIDATED_FEATURES_AVAILABLE" if validated_rows else "NO_CAUSALLY_VALIDATED_MODEL3_FEATURE",
        "selection_stage": "STAGE_22E_DEVELOPMENT_ONLY",
        "validation_rule": protocol["retention"]["validated_iff"],
        "candidate_assignment_count": len(metrics), "validated_assignment_count": len(validated_rows),
        "validated_assignments": validated_rows,
        "next_stage": "STAGE_22F_STRENGTH_CALIBRATION_REQUIRED" if validated_rows else "STOP_CONDITION",
        "strength_calibration_run": False, "heldout_used": False,
    }
    atomic_json(VALIDATED, validated_payload)

    by_target = {}
    for target in ("CWE-120", "CWE-327", "CWE-89"):
        target_rows = [x for x in metrics if x["target_cwe"] == target]
        target_validated = [x for x in target_rows if x["validated"]]
        by_target[target] = {
            "candidate_count": len(target_rows), "validated_count": len(target_validated),
            "max_repair_count": max(x["repair_count"] for x in target_rows),
            "max_repair_rate": max(x["repair_rate"] for x in target_rows),
            "min_corruption_count": min(x["corruption_count"] for x in target_rows),
            "min_corruption_rate": min(x["corruption_rate"] for x in target_rows),
        }
    verification_payload = {
        "schema_version": "phase22e_steered_scan_verification_v1", "status": "PASS",
        "artifact_hashes": hashes, "record_count": EXPECTED_COUNT, "eligible_count": EXPECTED_COUNT,
        "skipped_count": 0, "target_positive_count": returned["target_positive_count"],
        "any_finding_count": returned["any_finding_count"], "semgrep_version": returned["semgrep_version"],
        "checks": {
            "immutable_hash_chain": True, "exact_semgrep_version": True, "unchanged_scanner_and_wrapper": True,
            "complete_exact_order": True, "row_metadata_and_code_projection": True,
            "finding_and_derived_fields": True, "no_skipped_records": True,
            "no_generation_during_scan": True, "no_heldout_access": True,
        },
        "causal_results_sha256": sha256_file(RESULTS), "summary_csv_sha256": sha256_file(SUMMARY_CSV),
        "validated_feature_manifest_sha256": sha256_file(VALIDATED), "validated_assignment_count": len(validated_rows),
        "by_target": by_target,
        "scanner_limitation": "The frozen scanner suppresses Semgrep subprocess exceptions and p/security-audit is not content-pinned. Exact files, version, full row coverage, and skip policy are verified, but silent per-record Semgrep failure cannot be independently excluded.",
        "strength_calibration_run": False, "heldout_used": False,
    }
    atomic_json(VERIFICATION, verification_payload)
    print(json.dumps({
        "status": "PASS", "scan_output_sha256": hashes["scan_output"],
        "validated_assignment_count": len(validated_rows), "by_target": by_target,
        "validated_assignments": validated_rows,
        "causal_results_sha256": sha256_file(RESULTS), "summary_csv_sha256": sha256_file(SUMMARY_CSV),
        "validated_feature_manifest_sha256": sha256_file(VALIDATED), "verification_sha256": sha256_file(VERIFICATION),
    }, indent=2))


if __name__ == "__main__":
    main()
