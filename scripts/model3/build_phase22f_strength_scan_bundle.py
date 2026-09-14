#!/usr/bin/env python
"""Build immutable Stage 22F strength-scan Colab bundle."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
PHASE = ROOT / "revision/model3/phase22"
OUT = PHASE / "outputs"
BUNDLE = PHASE / "phase22f_strength_colab_scan_bundle"
ZIP_PATH = PHASE / "phase22f_strength_colab_scan_bundle.zip"
GENERATION = OUT / "phase22f_strength_generations.json"
GENERATION_MANIFEST = OUT / "phase22f_strength_generation_manifest.json"
GENERATION_VALIDATION = OUT / "phase22f_strength_generation_validation.json"
PROTOCOL = OUT / "phase22f_strength_calibration_protocol.json"
EXECUTION = OUT / "phase22f_strength_execution_protocol.json"
SCAN_INPUT = OUT / "phase22f_strength_scan_input.json"
SCANNER = ROOT / "phases/phase9/colab_scan_phase9.py"
WRAPPER = PHASE / "scripts/scan_phase22f_strength.py"
TRANSFER_NAME = "phase22f_strength_scan_transfer_manifest.json"
EXPECTED = {
    GENERATION: "7d8996c7d62c8966eea62f51324a3a6c264c3a92284361cf59c495362fac24c9",
    GENERATION_MANIFEST: "1ee57453d5a34b7eef39ab51bc27f57a1af5fb525f658546141d43a7dc372169",
    GENERATION_VALIDATION: "4a89da1cf2de53a96193ffc962feb131a4086e1debd6ab7e20f77dedea08cb6d",
    PROTOCOL: "20e57079a57a65fe4569132c09b05b649a98627c85fe9c6effd8b3c20aac279c",
    EXECUTION: "1c4cccc2c76011783113022c5d836120515c818e367af9ba001d8857edc1cd72",
    SCANNER: "60cedd3d6ec2d19e5e7e5ba4d709230efdd5d9478047492fefda4dd88d9afbb7",
}


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
    require(not BUNDLE.exists() and not ZIP_PATH.exists() and not SCAN_INPUT.exists(), "immutable bundle/input already exists")
    for path, expected in EXPECTED.items():
        require(path.is_file() and sha256_file(path) == expected, f"frozen artifact mismatch: {path.name}")
    generation = json.loads(GENERATION.read_text(encoding="utf-8"))
    validation = json.loads(GENERATION_VALIDATION.read_text(encoding="utf-8"))
    records = generation["records"]
    require(generation["status"] == "COMPLETE" and validation["status"] == "PASS" and len(records) == 1083, "generation gate not passed")
    scan_rows = [{
        "record_index": x["record_index"], "record_id": x["record_id"], "prompt_id": x["prompt_id"],
        "target_cwe": x["target_cwe"], "layer": x["layer"], "feature_id": x["feature_id"],
        "statistical_rank": x["statistical_rank"], "screening_alpha": x["screening_alpha"],
        "strength_multiplier": x["strength_multiplier"], "calibration_alpha": x["calibration_alpha"],
        "arm": x["arm"], "language": x["language"], "generation_valid": x["validity"]["is_valid"],
        "generation_invalid_reason": x["validity"]["invalid_reason"], "generated_code": x["generated_code"],
    } for x in records]
    require(sum(not x["generation_valid"] for x in scan_rows) == 34, "invalid projection mismatch")
    atomic_json(SCAN_INPUT, scan_rows)
    BUNDLE.mkdir(parents=True)
    for source, name in ((SCAN_INPUT, SCAN_INPUT.name), (SCANNER, "colab_scan_phase9.py"), (WRAPPER, WRAPPER.name)):
        shutil.copy2(source, BUNDLE / name)
    record_ids = [x["record_id"] for x in scan_rows]
    transfer = {
        "schema_version": "phase22f_strength_scan_transfer_v1", "status": "FROZEN_BEFORE_SCAN",
        "generation_sha256": sha256_file(GENERATION), "generation_manifest_sha256": sha256_file(GENERATION_MANIFEST),
        "generation_validation_sha256": sha256_file(GENERATION_VALIDATION), "strength_protocol_sha256": sha256_file(PROTOCOL),
        "execution_protocol_sha256": sha256_file(EXECUTION), "scan_input_filename": SCAN_INPUT.name,
        "scan_input_sha256": sha256_file(SCAN_INPUT), "record_count": 1083, "record_ids": record_ids,
        "record_ids_canonical_sha256": hashlib.sha256(json.dumps(record_ids, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "target_counts": {"CWE-120": 810, "CWE-327": 33, "CWE-89": 240},
        "multiplier_counts": {"0.5": 361, "1.5": 361, "2.0": 361}, "invalid_generation_count": 34,
        "scanner_filename": "colab_scan_phase9.py", "scanner_sha256": sha256_file(SCANNER),
        "wrapper_filename": WRAPPER.name, "wrapper_sha256": sha256_file(WRAPPER),
        "required_semgrep_version": "1.175.0", "registry_config": "p/security-audit", "registry_snapshot_hash": "NOT_CONTENT_PINNED",
        "checkpoint_interval": 10, "expected_scan_output_filename": "phase22f_strength_scans.json",
        "expected_return_manifest_filename": "phase22f_strength_scan_return_manifest.json",
        "invalid_handling": "retain and scan where eligible; unsafe invalid cannot repair and counts invalid; safe invalid counts corruption",
        "scanner_rules_changed": False, "generation_run": False, "stage22g_started": False, "heldout_used": False,
    }
    atomic_json(BUNDLE / TRANSFER_NAME, transfer)
    (BUNDLE / "README.txt").write_text("Stage 22F immutable strength scan. Linux/Colab and Semgrep 1.175.0 required. Checkpoint every 10 rows. Return phase22f_strength_scans.json and phase22f_strength_scan_return_manifest.json.\n", encoding="utf-8", newline="\n")
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(BUNDLE.iterdir(), key=lambda x: x.name):
            archive.write(path, arcname=path.name)
    bundle_manifest = {
        "schema_version": "phase22f_strength_scan_bundle_v1", "status": "READY_FOR_COLAB",
        "bundle_zip": str(ZIP_PATH.relative_to(ROOT)).replace("\\", "/"), "bundle_zip_sha256": sha256_file(ZIP_PATH),
        "transfer_manifest_sha256": sha256_file(BUNDLE / TRANSFER_NAME), "scan_input_sha256": sha256_file(SCAN_INPUT),
        "wrapper_sha256": sha256_file(WRAPPER), "scanner_sha256": sha256_file(SCANNER),
        "record_count": 1083, "invalid_generation_count": 34,
        "return_files": ["phase22f_strength_scans.json", "phase22f_strength_scan_return_manifest.json"],
        "actual_scan_run": False, "stage22g_started": False, "heldout_used": False,
    }
    atomic_json(OUT / "phase22f_strength_scan_bundle_manifest.json", bundle_manifest)
    print(json.dumps(bundle_manifest, indent=2))


if __name__ == "__main__":
    main()
