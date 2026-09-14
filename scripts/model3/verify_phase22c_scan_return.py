#!/usr/bin/env python
"""Verify returned Model3 22C scan and freeze B0 target denominators."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
PHASE = ROOT / "revision/model3/phase22"
OUT = PHASE / "outputs"
BUNDLE = PHASE / "phase22c_colab_scan_bundle"
POPULATION = OUT / "phase22c_source_population.json"
PROTOCOL = OUT / "phase22c_b0_protocol.json"
GENERATION = OUT / "phase22c_b0_dev_outputs.json"
GEN_MANIFEST = OUT / "phase22c_b0_generation_manifest.json"
VALIDATION = OUT / "phase22c_b0_local_validation.json"
SCAN = OUT / "phase22c_b0_dev_scan.json"
RETURN = OUT / "phase22c_b0_scan_return_manifest.json"
TRANSFER = BUNDLE / "phase22c_scan_transfer_manifest.json"
SCANNER = ROOT / "phases/phase9/colab_scan_phase9.py"
WRAPPER = PHASE / "scripts/scan_phase22c_manifest.py"
VERIFICATION = OUT / "phase22c_b0_scan_verification.json"
FROZEN = OUT / "phase22c_b0_population_manifest.json"
SUPPORT = OUT / "phase22c_cwe_support.csv"
EXPECTED_COUNT = 180
DERIVATION_CWES = {"CWE-120", "CWE-125", "CWE-787", "CWE-190", "CWE-476", "CWE-89", "CWE-79", "CWE-327"}
PLANNED_CWES = ("CWE-120", "CWE-327", "CWE-89")
TARGET_LANGUAGES = {"c", "cpp", "c++", "python", "java", "javascript", "js"}


def sha256_file(path: Path) -> str:
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


def main() -> None:
    for path in (POPULATION, PROTOCOL, GENERATION, GEN_MANIFEST, VALIDATION, SCAN, RETURN, TRANSFER, SCANNER, WRAPPER):
        require(path.is_file(), f"missing required artifact: {path}")
    population = json.loads(POPULATION.read_text(encoding="utf-8"))
    generation = json.loads(GENERATION.read_text(encoding="utf-8"))
    gen_manifest = json.loads(GEN_MANIFEST.read_text(encoding="utf-8"))
    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))
    scan = json.loads(SCAN.read_text(encoding="utf-8"))
    returned = json.loads(RETURN.read_text(encoding="utf-8"))
    transfer = json.loads(TRANSFER.read_text(encoding="utf-8"))
    generation_hash = sha256_file(GENERATION)
    scan_hash = sha256_file(SCAN)
    require(validation["validation_status"] == "PASS", "local generation validation not PASS")
    require(generation_hash == validation["generation_sha256"] == returned["generation_sha256"] == transfer["generation_sha256"], "generation provenance mismatch")
    require(returned.get("status") == "COMPLETE" and returned.get("generation_unchanged") is True, "return manifest incomplete")
    require(sha256_file(SCANNER) == returned["scanner_sha256"] == transfer["scanner_sha256"], "scanner hash mismatch")
    require(sha256_file(WRAPPER) == returned["wrapper_sha256"] == transfer["wrapper_sha256"], "wrapper hash mismatch")
    require(returned["semgrep_version"] == transfer["required_semgrep_version"] == "1.175.0", "Semgrep version mismatch")
    require(scan_hash == returned["scan_output_sha256"], "scan hash mismatch")
    require(returned["scanner_rules_changed"] is False and returned["heldout_used"] is False, "prohibited scanner/heldout report")
    require(len(population["records"]) == len(generation) == len(scan) == EXPECTED_COUNT, "record count mismatch")
    expected_ids = [int(x["prompt_id"]) for x in population["records"]]
    generation_ids = [int(x["prompt_id"]) for x in generation]
    scan_ids = [int(x["prompt_id"]) for x in scan]
    require(scan_ids == generation_ids == expected_ids == gen_manifest["completed_prompt_ids_source_order"] == transfer["prompt_ids_source_order"], "prompt IDs/order mismatch")
    require(len(set(scan_ids)) == EXPECTED_COUNT, "duplicate scan ID")
    required = {"prompt_id", "cwe_id", "language", "findings", "vulnerable_cwes", "is_vulnerable", "skipped"}
    finding_counts: Counter[str] = Counter()
    frozen_records = []
    for source, gen, row in zip(population["records"], generation, scan, strict=True):
        pid = int(gen["prompt_id"])
        require(required.issubset(row), f"scan schema missing fields: {pid}")
        require(int(row["prompt_id"]) == pid, f"scan ID mismatch: {pid}")
        require(row["cwe_id"] == gen["cwe_id"] == source["cwe_identifier"], f"scan CWE mismatch: {pid}")
        require(row["language"] == gen["language"].lower() == source["language"], f"scan language mismatch: {pid}")
        require(isinstance(row["findings"], list) and isinstance(row["vulnerable_cwes"], list), f"finding schema mismatch: {pid}")
        dedup = {(x["cwe_id"], x["rule_id"]) for x in row["findings"]}
        require(len(dedup) == len(row["findings"]), f"duplicate finding key: {pid}")
        found = {x["cwe_id"] for x in row["findings"]}
        require(found == set(row["vulnerable_cwes"]), f"vulnerable CWE list mismatch: {pid}")
        require(row["is_vulnerable"] == any(x in DERIVATION_CWES for x in found), f"is_vulnerable mismatch: {pid}")
        expected_skipped = gen["language"].lower() not in TARGET_LANGUAGES or not gen["generated_code"].strip()
        require(row["skipped"] == expected_skipped, f"skip policy mismatch: {pid}")
        if row["skipped"]:
            require(not row["findings"] and not row["vulnerable_cwes"] and row["is_vulnerable"] is False, f"skipped row contains findings: {pid}")
        finding_counts.update(found)
        target_vulnerable = not row["skipped"] and row["cwe_id"] in found
        frozen_records.append({
            "population_index": source["population_index"], "source_index": source["source_index"],
            "prompt_id": pid, "cwe_identifier": source["cwe_identifier"], "language": source["language"],
            "source_prompt_sha256": source["source_prompt_sha256"],
            "rendered_prompt_sha256": gen["rendered_prompt_sha256"],
            "generated_code_sha256": hashlib.sha256(gen["generated_code"].encode()).hexdigest(),
            "scanner_eligible": not row["skipped"], "target_vulnerable": target_vulnerable,
            "target_safe": not row["skipped"] and not target_vulnerable,
            "vulnerable_cwes": sorted(found),
        })
    eligible_count = sum(not x["skipped"] for x in scan)
    skipped_count = sum(x["skipped"] for x in scan)
    vulnerable_any = sum(x["is_vulnerable"] for x in scan)
    require((eligible_count, skipped_count, vulnerable_any) == (returned["eligible_count"], returned["skipped_count"], returned["vulnerable_count"]), "return count mismatch")
    support_rows = []
    for cwe in PLANNED_CWES:
        rows = [x for x in frozen_records if x["cwe_identifier"] == cwe]
        eligible = [x for x in rows if x["scanner_eligible"]]
        unsafe = [x for x in eligible if x["target_vulnerable"]]
        safe = [x for x in eligible if x["target_safe"]]
        support_rows.append({
            "cwe_id": cwe, "source_prompt_count": len(rows), "scanner_eligible_count": len(eligible),
            "target_vulnerable_count": len(unsafe), "target_safe_count": len(safe),
            "scanner_skipped_count": len(rows) - len(eligible),
            "feature_discovery_support_status": "SUPPORTED_NONZERO" if unsafe and safe else "INSUFFICIENT_SUPPORT",
        })
    tmp = SUPPORT.with_name(SUPPORT.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(support_rows[0].keys()))
        writer.writeheader()
        writer.writerows(support_rows)
    os.replace(tmp, SUPPORT)
    frozen = {
        "schema_version": "phase22c_model3_b0_population_manifest_v1", "status": "FROZEN_AFTER_VERIFIED_SCAN",
        "split": "DEVELOPMENT", "method": "B0", "seed": 42, "record_count": EXPECTED_COUNT,
        "eligible_count": eligible_count, "skipped_count": skipped_count,
        "target_vulnerable_total": sum(x["target_vulnerable"] for x in frozen_records),
        "target_safe_total": sum(x["target_safe"] for x in frozen_records),
        "by_source_cwe": {x["cwe_id"]: {"population": x["source_prompt_count"], "eligible": x["scanner_eligible_count"], "target_vulnerable": x["target_vulnerable_count"], "target_safe": x["target_safe_count"], "skipped": x["scanner_skipped_count"]} for x in support_rows},
        "prompt_ids_source_order": expected_ids,
        "prompt_ids_source_order_canonical_sha256": canonical_hash(expected_ids),
        "generation_sha256": generation_hash, "scan_sha256": scan_hash,
        "scanner_sha256": sha256_file(SCANNER), "semgrep_version": returned["semgrep_version"],
        "records": frozen_records, "source_cwe_label_treated_as_vulnerability_evidence": False,
        "heldout_used": False, "model1_or_model2_outcomes_imported": False,
    }
    atomic_json(FROZEN, frozen)
    verification = {
        "schema_version": "phase22c_model3_b0_scan_verification_v1", "verification_status": "PASS",
        "generation_sha256": generation_hash, "scan_output_sha256": scan_hash,
        "scan_return_manifest_sha256": sha256_file(RETURN), "transfer_manifest_sha256": sha256_file(TRANSFER),
        "scanner_sha256": sha256_file(SCANNER), "wrapper_sha256": sha256_file(WRAPPER),
        "semgrep_version": returned["semgrep_version"], "platform": returned["platform"],
        "record_count": EXPECTED_COUNT, "eligible_count": eligible_count, "skipped_count": skipped_count,
        "vulnerable_count_any_derivation_cwe": vulnerable_any,
        "target_vulnerable_total": frozen["target_vulnerable_total"], "target_safe_total": frozen["target_safe_total"],
        "finding_record_counts_by_cwe": dict(sorted(finding_counts.items())),
        "planned_cwe_support": support_rows, "frozen_population_manifest_sha256": sha256_file(FROZEN),
        "support_table_sha256": sha256_file(SUPPORT), "schema_valid": True, "deduplication_valid": True,
        "skip_policy_valid": True, "generation_unchanged": True, "scanner_configuration_drift": False,
        "heldout_used": False,
        "warnings": [
            "The frozen scanner suppresses Semgrep subprocess exceptions and lacks row-level scanner-success metadata; full row coverage, exact wrapper/scanner hashes, and the prescribed Semgrep version are verified, but silent per-record Semgrep failure cannot be independently excluded.",
            "The p/security-audit registry snapshot is not content-pinned; the frozen runtime version is recorded as Semgrep 1.175.0.",
        ],
        "activation_extraction_run": False, "causal_intervention_run": False, "model1_or_model2_modified": False,
    }
    atomic_json(VERIFICATION, verification)
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
