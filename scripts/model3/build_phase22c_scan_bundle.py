#!/usr/bin/env python
"""Build the frozen Linux/Colab scan transfer for Model3 22C B0."""

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
BUNDLE = PHASE / "phase22c_colab_scan_bundle"
ZIP_PATH = PHASE / "phase22c_colab_scan_bundle.zip"
GENERATION = OUT / "phase22c_b0_dev_outputs.json"
GEN_MANIFEST = OUT / "phase22c_b0_generation_manifest.json"
VALIDATION = OUT / "phase22c_b0_local_validation.json"
PROTOCOL = OUT / "phase22c_b0_protocol.json"
POPULATION = OUT / "phase22c_source_population.json"
SCANNER_SOURCE = ROOT / "phases/phase9/colab_scan_phase9.py"
WRAPPER_SOURCE = PHASE / "scripts/scan_phase22c_manifest.py"
TRANSFER_NAME = "phase22c_scan_transfer_manifest.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))
    generation_manifest = json.loads(GEN_MANIFEST.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    require(validation["validation_status"] == "PASS", "local generation validation is not PASS")
    require(validation["failure_count"] == 0 and validation["empty_or_whitespace_count"] == 0, "scanner transfer requires 180 nonempty successful generations")
    require(sha256_file(GENERATION) == validation["generation_sha256"], "generation hash mismatch")
    require(sha256_file(SCANNER_SOURCE) == protocol["scanner"]["script_sha256"], "frozen scanner hash mismatch")
    require(not BUNDLE.exists() and not ZIP_PATH.exists(), "scan bundle or zip already exists")
    BUNDLE.mkdir(parents=True)
    shutil.copy2(GENERATION, BUNDLE / GENERATION.name)
    shutil.copy2(SCANNER_SOURCE, BUNDLE / "colab_scan_phase9.py")
    shutil.copy2(WRAPPER_SOURCE, BUNDLE / WRAPPER_SOURCE.name)
    transfer = {
        "schema_version": "phase22c_model3_b0_scan_transfer_v1", "status": "FROZEN_BEFORE_SCAN",
        "method": "B0", "seed": 42, "split": "DEVELOPMENT",
        "generation_filename": GENERATION.name, "generation_sha256": sha256_file(GENERATION),
        "generation_manifest_sha256": sha256_file(GEN_MANIFEST), "local_validation_sha256": sha256_file(VALIDATION),
        "protocol_sha256": sha256_file(PROTOCOL), "source_population_sha256": sha256_file(POPULATION),
        "record_count": 180, "prompt_ids_source_order": generation_manifest["completed_prompt_ids_source_order"],
        "scanner_filename": "colab_scan_phase9.py", "scanner_sha256": sha256_file(SCANNER_SOURCE),
        "scanner_policy": "UNCHANGED_FROZEN_MODEL1_ICD", "required_semgrep_version": "1.175.0",
        "wrapper_filename": WRAPPER_SOURCE.name, "wrapper_sha256": sha256_file(WRAPPER_SOURCE),
        "expected_scan_output_filename": "phase22c_b0_dev_scan.json",
        "expected_return_manifest_filename": "phase22c_b0_scan_return_manifest.json",
        "heldout_used": False, "scanner_rules_changed": False,
    }
    atomic_json(BUNDLE / TRANSFER_NAME, transfer)
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(BUNDLE.iterdir(), key=lambda x: x.name):
            zf.write(path, arcname=path.name)
    manifest = {
        "schema_version": "phase22c_model3_b0_scan_bundle_v1", "status": "READY_FOR_COLAB",
        "bundle_zip": "revision/model3/phase22/phase22c_colab_scan_bundle.zip",
        "bundle_zip_sha256": sha256_file(ZIP_PATH), "transfer_manifest_sha256": sha256_file(BUNDLE / TRANSFER_NAME),
        "files": {path.name: sha256_file(path) for path in sorted(BUNDLE.iterdir(), key=lambda x: x.name)},
        "actual_scan_run": False, "heldout_used": False,
    }
    atomic_json(OUT / "phase22c_b0_scan_bundle_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
