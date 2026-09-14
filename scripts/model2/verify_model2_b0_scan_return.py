#!/usr/bin/env python
"""Verify the returned Model2 B0 ICD scan and freeze planned-CWE support."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "revision/model2/phase21/outputs"
GENERATION = OUT / "model2_b0_dev_outputs.json"
GEN_MANIFEST = OUT / "model2_b0_dev_generation_manifest.json"
SCAN = OUT / "model2_b0_dev_scan.json"
RETURN = OUT / "model2_b0_dev_scan_return_manifest.json"
SCANNER = ROOT / "phases/phase9/colab_scan_phase9.py"
WRAPPER = ROOT / "revision/model2/phase21/scripts/scan_model2_b0_manifest.py"
VERIFICATION = OUT / "model2_b0_dev_scan_verification.json"
SUPPORT = OUT / "model2_dev_cwe_support.csv"
EXPECTED_GENERATION_HASH = "d90f58718a40aa702e25018265bb0ab843afd74b3cbfada81cdd139c2c9a6760"
EXPECTED_SCANNER_HASH = "60cedd3d6ec2d19e5e7e5ba4d709230efdd5d9478047492fefda4dd88d9afbb7"
EXPECTED_WRAPPER_HASH = "62434f87940c9c396397ff6db14182637b8ea31a634db3f9f486d2af93daffb8"
DERIVATION_CWES = {"CWE-120", "CWE-125", "CWE-787", "CWE-190", "CWE-476", "CWE-89", "CWE-79", "CWE-327"}
PLANNED_CWES = ("CWE-120", "CWE-327", "CWE-89")
TARGET_LANGUAGES = {"c", "cpp", "c++", "python", "java", "javascript", "js"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    for path in (GENERATION, GEN_MANIFEST, SCAN, RETURN, SCANNER, WRAPPER):
        require(path.is_file(), f"missing required artifact: {path}")
    generation = json.loads(GENERATION.read_text(encoding="utf-8"))
    generation_manifest = json.loads(GEN_MANIFEST.read_text(encoding="utf-8"))
    scan = json.loads(SCAN.read_text(encoding="utf-8"))
    returned = json.loads(RETURN.read_text(encoding="utf-8"))
    generation_hash = sha256_file(GENERATION)
    scan_hash = sha256_file(SCAN)
    require(generation_hash == EXPECTED_GENERATION_HASH == returned["generation_sha256"] == generation_manifest["generation_output"]["sha256"], "generation provenance hash mismatch")
    require(returned.get("status") == "COMPLETE" and returned.get("generation_unchanged") is True, "return manifest is incomplete or mutable")
    require(sha256_file(SCANNER) == EXPECTED_SCANNER_HASH == returned["scanner_sha256"], "scanner hash mismatch")
    require(sha256_file(WRAPPER) == EXPECTED_WRAPPER_HASH == returned["wrapper_sha256"], "wrapper hash mismatch")
    require(scan_hash == returned["scan_output_sha256"], "scan output hash mismatch")
    require(returned["scanner_rules_changed"] is False and returned["heldout_used"] is False, "return manifest reports prohibited change/use")
    require(isinstance(generation, list) and isinstance(scan, list) and len(generation) == len(scan) == 1341, "record count mismatch")
    generation_ids = [int(x["prompt_id"]) for x in generation]
    scan_ids = [int(x["prompt_id"]) for x in scan]
    require(scan_ids == generation_ids == generation_manifest["completed_prompt_ids_source_order"], "prompt IDs/order mismatch")
    require(len(set(scan_ids)) == 1341, "duplicate scan ID")
    required = {"prompt_id", "cwe_id", "language", "findings", "vulnerable_cwes", "is_vulnerable", "skipped"}
    findings_by_cwe: dict[str, int] = {}
    for gen, row in zip(generation, scan, strict=True):
        pid = int(gen["prompt_id"])
        require(required.issubset(row), f"scan schema missing fields for {pid}")
        require(row["cwe_id"] == gen["cwe_id"] and row["language"] == gen["language"].lower(), f"scan metadata mismatch for {pid}")
        require(isinstance(row["findings"], list) and isinstance(row["vulnerable_cwes"], list), f"finding schema mismatch for {pid}")
        dedup = {(x["cwe_id"], x["rule_id"]) for x in row["findings"]}
        require(len(dedup) == len(row["findings"]), f"duplicate finding key for {pid}")
        found_cwes = {x["cwe_id"] for x in row["findings"]}
        require(found_cwes == set(row["vulnerable_cwes"]), f"vulnerable CWE list mismatch for {pid}")
        require(row["is_vulnerable"] == any(x in DERIVATION_CWES for x in found_cwes), f"is_vulnerable mismatch for {pid}")
        expected_skipped = gen["language"].lower() not in TARGET_LANGUAGES or not gen["generated_code"].strip()
        require(row["skipped"] == expected_skipped, f"skip-policy mismatch for {pid}")
        if row["skipped"]:
            require(row["findings"] == [] and row["vulnerable_cwes"] == [] and row["is_vulnerable"] is False, f"skipped record contains finding for {pid}")
        for cwe in found_cwes:
            findings_by_cwe[cwe] = findings_by_cwe.get(cwe, 0) + 1
    eligible_count = sum(not x["skipped"] for x in scan)
    skipped_count = sum(x["skipped"] for x in scan)
    vulnerable_count = sum(x["is_vulnerable"] for x in scan)
    require((eligible_count, skipped_count, vulnerable_count) == (returned["eligible_count"], returned["skipped_count"], returned["vulnerable_count"]), "return summary count mismatch")
    support_rows = []
    for cwe in PLANNED_CWES:
        source_rows = [x for x in scan if x["cwe_id"] == cwe]
        source_eligible = [x for x in source_rows if not x["skipped"]]
        source_unsafe = [x for x in source_eligible if cwe in set(x["vulnerable_cwes"])]
        source_safe = [x for x in source_eligible if cwe not in set(x["vulnerable_cwes"])]
        global_eligible = [x for x in scan if not x["skipped"]]
        global_unsafe = [x for x in global_eligible if cwe in set(x["vulnerable_cwes"])]
        global_safe = [x for x in global_eligible if cwe not in set(x["vulnerable_cwes"])]
        support_rows.append({
            "cwe_id": cwe,
            "source_prompt_count": len(source_rows),
            "source_scanner_eligible_count": len(source_eligible),
            "source_target_vulnerable_count": len(source_unsafe),
            "source_target_safe_count": len(source_safe),
            "source_scanner_skipped_count": sum(x["skipped"] for x in source_rows),
            "global_eligible_target_unsafe_count_model1_partition": len(global_unsafe),
            "global_eligible_target_safe_count_before_balance": len(global_safe),
            "safe_balance_cap_5x_unsafe": min(len(global_safe), 5 * len(global_unsafe)),
            "feature_discovery_support_status": "SUPPORTED_NONZERO_MODEL1_PARTITION" if global_unsafe and global_safe else "INSUFFICIENT_SUPPORT",
            "limitation": "Source-CWE-matched unsafe count is small; ranking reuses the Model1 all-eligible target-CWE partition." if len(source_unsafe) < 5 else "",
        })
    fields = list(support_rows[0].keys())
    tmp = SUPPORT.with_name(SUPPORT.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(support_rows)
    os.replace(tmp, SUPPORT)
    value = {
        "schema_version": "phase21_model2_b0_scan_verification_v1",
        "verification_status": "PASS",
        "generation_sha256": generation_hash,
        "scan_output_sha256": scan_hash,
        "scan_return_manifest_sha256": sha256_file(RETURN),
        "scanner_sha256": sha256_file(SCANNER),
        "wrapper_sha256": sha256_file(WRAPPER),
        "semgrep_version": returned["semgrep_version"],
        "platform": returned["platform"],
        "record_count": 1341,
        "eligible_count": eligible_count,
        "skipped_count": skipped_count,
        "vulnerable_count_any_derivation_cwe": vulnerable_count,
        "finding_record_counts_by_cwe": dict(sorted(findings_by_cwe.items())),
        "prompt_ids_unique_and_source_ordered": True,
        "schema_valid": True,
        "deduplication_valid": True,
        "skip_policy_valid": True,
        "generation_unchanged": True,
        "scanner_configuration_drift": False,
        "heldout_used": False,
        "support_table_path": "revision/model2/phase21/outputs/model2_dev_cwe_support.csv",
        "support_table_sha256": sha256_file(SUPPORT),
        "planned_cwe_support": support_rows,
        "warnings": [
            "The frozen scanner catches and suppresses Semgrep subprocess exceptions and does not emit row-level scanner-success metadata; exact rule/wrapper hashes and full output coverage are verified, but silent per-record Semgrep failure cannot be independently excluded.",
            "The p/security-audit registry snapshot is not content-pinned; Semgrep version 1.175.0 is recorded for this execution.",
            "Source-CWE-matched support and Model1-style all-eligible per-CWE feature-discovery partitions are different populations and are reported separately.",
        ],
        "causal_intervention_run": False,
        "alpha_selection_run": False,
        "activation_extraction_run": False,
        "model1_modified": False,
    }
    atomic_json(VERIFICATION, value)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
