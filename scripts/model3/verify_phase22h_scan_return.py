#!/usr/bin/env python
"""Verify the returned Stage 22H Colab scan before any metric calculation."""

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
SCAN = OUT / "phase22h_scans.json"
RETURN = OUT / "phase22h_scan_return_manifest.json"
INPUT = OUT / "phase22h_scan_input.json"
BUNDLE_MANIFEST = OUT / "phase22h_scan_bundle_manifest.json"
TRANSFER = PHASE / "phase22h_colab_scan_bundle/phase22h_scan_transfer_manifest.json"
SCANNER = ROOT / "phases/phase9/colab_scan_phase9.py"
WRAPPER = PHASE / "scripts/scan_phase22h_manifest.py"
GENERATION = OUT / "phase22h_paired_generations.json"
VALIDATION = OUT / "phase22h_generation_validation.json"
PROTOCOL = OUT / "phase22g_prospective_evaluation_protocol.json"
POPULATION = OUT / "phase22g_evaluation_population.json"
OUTPUT = OUT / "phase22h_scan_verification.json"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    require(not OUTPUT.exists(), "immutable Stage 22H scan verification already exists")
    for path in (SCAN, RETURN, INPUT, BUNDLE_MANIFEST, TRANSFER, SCANNER, WRAPPER, GENERATION, VALIDATION, PROTOCOL, POPULATION):
        require(path.is_file(), f"missing scan lineage artifact: {path}")
    returned = json.loads(RETURN.read_text(encoding="utf-8"))
    container = json.loads(SCAN.read_text(encoding="utf-8"))
    inputs = json.loads(INPUT.read_text(encoding="utf-8"))
    transfer = json.loads(TRANSFER.read_text(encoding="utf-8"))
    bundle = json.loads(BUNDLE_MANIFEST.read_text(encoding="utf-8"))
    require(returned["status"] == "COMPLETE" and container["status"] == "COMPLETE", "returned scan not complete")
    require(returned["semgrep_version"] == container["semgrep_version"] == transfer["required_semgrep_version"] == "1.175.0", "Semgrep version mismatch")
    require(returned["scan_output_sha256"] == sha256_file(SCAN) == "59907ee84885dc200045197b0a365c38eb04da5b28eb7862c82cd623c3a36fa0", "scan output hash mismatch")
    lineage = {
        "generation_sha256": GENERATION,
        "generation_validation_sha256": VALIDATION,
        "protocol_sha256": PROTOCOL,
        "population_sha256": POPULATION,
        "scan_input_sha256": INPUT,
        "scanner_sha256": SCANNER,
        "wrapper_sha256": WRAPPER,
    }
    for field, path in lineage.items():
        actual = sha256_file(path)
        require(returned[field] == transfer[field] == actual, f"{field} lineage mismatch")
    require(bundle["scan_input_sha256"] == sha256_file(INPUT), "bundle/input mismatch")
    require(bundle["scanner_sha256"] == sha256_file(SCANNER) and bundle["wrapper_sha256"] == sha256_file(WRAPPER), "bundle scanner/wrapper mismatch")
    require(container["scan_input_sha256"] == sha256_file(INPUT), "scan container input mismatch")
    require(container["scanner_sha256"] == sha256_file(SCANNER) and container["wrapper_sha256"] == sha256_file(WRAPPER), "scan container implementation mismatch")

    records = container["records"]
    require(len(records) == len(inputs) == returned["record_count"] == transfer["record_count"] == 468, "row count mismatch")
    require([row["record_id"] for row in records] == [row["record_id"] for row in inputs] == transfer["record_ids"], "record ID/order mismatch")
    require(len({row["record_id"] for row in records}) == 468, "duplicate scan record IDs")
    require(canonical_sha256([row["record_id"] for row in records]) == transfer["record_ids_canonical_sha256"], "record ID canonical hash mismatch")
    fields = (
        "record_index", "record_id", "condition", "seed", "prompt_id", "target_cwe",
        "cwe_identifier", "language", "layer", "feature_id", "alpha",
        "generation_valid", "generation_invalid_reason", "generated_text_sha256",
    )
    for index, (record, inp) in enumerate(zip(records, inputs)):
        require(record["record_index"] == index, f"record index mismatch at {index}")
        require(all(record[field] == inp[field] for field in fields), f"scan projection metadata mismatch at {index}")
        keys = [(item["cwe_id"], item["rule_id"]) for item in record["findings"]]
        require(len(keys) == len(set(keys)), f"duplicate finding at {index}")
        cwes = {item["cwe_id"] for item in record["findings"]}
        require(cwes == set(record["vulnerable_cwes"]), f"vulnerable CWE mismatch at {index}")
        require(record["target_cwe_present"] == (record["target_cwe"] in cwes), f"target flag mismatch at {index}")
        require(record["any_finding"] == bool(record["findings"]), f"any-finding flag mismatch at {index}")
        require(record["skipped"] is False, f"unexpected scanner skip at {index}")

    condition_counts = Counter(row["condition"] for row in records)
    seed_counts = Counter(str(row["seed"]) for row in records)
    require(dict(condition_counts) == returned["condition_counts"] == transfer["condition_counts"], "condition count mismatch")
    require(dict(seed_counts) == returned["seed_counts"] == transfer["seed_counts"], "seed count mismatch")
    require(returned["eligible_count"] == 468 and returned["skipped_count"] == 0, "eligibility count mismatch")
    require(returned["target_positive_count"] == sum(row["target_cwe_present"] for row in records), "target-positive aggregate mismatch")
    require(returned["any_finding_count"] == sum(row["any_finding"] for row in records), "any-finding aggregate mismatch")
    result = {
        "schema_version": "phase22h_scan_verification_v1",
        "status": "PASS",
        "checks": {
            "return_and_scan_complete": True,
            "exact_semgrep_1_175_0": True,
            "scan_output_hash_matches": True,
            "generation_protocol_population_lineage_matches": True,
            "scanner_and_wrapper_hashes_match": True,
            "all_468_rows_in_frozen_order": True,
            "all_scan_metadata_matches_projection": True,
            "all_findings_and_flags_internally_consistent": True,
            "all_468_records_scanner_eligible": True,
            "zero_skipped_records": True,
            "scanner_rules_unchanged": True,
        },
        "scan_output_sha256": sha256_file(SCAN),
        "scan_return_manifest_sha256": sha256_file(RETURN),
        "scan_input_sha256": sha256_file(INPUT),
        "scanner_sha256": sha256_file(SCANNER),
        "wrapper_sha256": sha256_file(WRAPPER),
        "record_count": len(records),
        "eligible_count": sum(not row["skipped"] for row in records),
        "skipped_count": sum(row["skipped"] for row in records),
        "security_metrics_computed": False,
    }
    atomic_json(OUTPUT, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
