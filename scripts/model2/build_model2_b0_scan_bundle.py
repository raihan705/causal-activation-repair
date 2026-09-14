#!/usr/bin/env python
"""Build the immutable Model2 B0 Colab scanner bundle after local validation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
PHASE = ROOT / "revision/model2/phase21"
OUT = PHASE / "outputs"
BUNDLE = PHASE / "colab_scan_bundle"
GENERATION = OUT / "model2_b0_dev_outputs.json"
GEN_MANIFEST = OUT / "model2_b0_dev_generation_manifest.json"
VALIDATION = OUT / "model2_b0_dev_local_validation.json"
SCANNER_SOURCE = ROOT / "phases/phase9/colab_scan_phase9.py"
WRAPPER_SOURCE = PHASE / "scripts/scan_model2_b0_manifest.py"
TRANSFER_NAME = "model2_b0_dev_scan_transfer_manifest.json"
ZIP_PATH = PHASE / "model2_b0_dev_colab_scan_bundle.zip"
EXPECTED_GENERATION_SHA256 = "d90f58718a40aa702e25018265bb0ab843afd74b3cbfada81cdd139c2c9a6760"
EXPECTED_SCANNER_SHA256 = "60cedd3d6ec2d19e5e7e5ba4d709230efdd5d9478047492fefda4dd88d9afbb7"


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
    require(validation["validation_status"] == "PASS", "local generation validation is not PASS")
    require(sha256_file(GENERATION) == EXPECTED_GENERATION_SHA256 == validation["generation_sha256"], "generation hash mismatch")
    require(sha256_file(SCANNER_SOURCE) == EXPECTED_SCANNER_SHA256, "frozen scanner hash mismatch")
    if BUNDLE.exists():
        raise RuntimeError(f"bundle directory already exists: {BUNDLE}")
    BUNDLE.mkdir(parents=True)
    shutil.copy2(GENERATION, BUNDLE / GENERATION.name)
    shutil.copy2(SCANNER_SOURCE, BUNDLE / "colab_scan_phase9.py")
    shutil.copy2(WRAPPER_SOURCE, BUNDLE / WRAPPER_SOURCE.name)
    transfer = {
        "schema_version": "phase21_model2_b0_scan_transfer_v1",
        "status": "FROZEN_BEFORE_SCAN",
        "method": "B0", "seed": 42, "split": "DEVELOPMENT",
        "generation_filename": GENERATION.name,
        "generation_sha256": EXPECTED_GENERATION_SHA256,
        "generation_manifest_sha256": sha256_file(GEN_MANIFEST),
        "local_validation_sha256": sha256_file(VALIDATION),
        "record_count": 1341,
        "prompt_ids_source_order": generation_manifest["completed_prompt_ids_source_order"],
        "scanner_filename": "colab_scan_phase9.py",
        "scanner_sha256": EXPECTED_SCANNER_SHA256,
        "scanner_policy": "UNCHANGED_FROZEN_MODEL1_ICD",
        "wrapper_filename": WRAPPER_SOURCE.name,
        "wrapper_sha256": sha256_file(WRAPPER_SOURCE),
        "expected_scan_output_filename": "model2_b0_dev_scan.json",
        "expected_return_manifest_filename": "model2_b0_dev_scan_return_manifest.json",
        "heldout_used": False, "scanner_rules_changed": False,
    }
    atomic_json(BUNDLE / TRANSFER_NAME, transfer)
    if ZIP_PATH.exists():
        raise RuntimeError(f"bundle zip already exists: {ZIP_PATH}")
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(BUNDLE.iterdir(), key=lambda x: x.name):
            zf.write(path, arcname=path.name)
    manifest = {
        "schema_version": "phase21_model2_b0_scan_bundle_v1", "status": "READY_FOR_COLAB",
        "bundle_zip": "revision/model2/phase21/model2_b0_dev_colab_scan_bundle.zip",
        "bundle_zip_sha256": sha256_file(ZIP_PATH),
        "transfer_manifest_sha256": sha256_file(BUNDLE / TRANSFER_NAME),
        "files": {path.name: sha256_file(path) for path in sorted(BUNDLE.iterdir(), key=lambda x: x.name)},
        "actual_scan_run": False, "heldout_used": False,
    }
    atomic_json(OUT / "model2_b0_dev_scan_bundle_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
