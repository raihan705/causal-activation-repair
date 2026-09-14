#!/usr/bin/env python3
"""Freeze B0 repetition-1 outputs and build the Phase 18 Colab scan bundle."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
PHASE = ROOT / "revision/model1/phase18"
OUT = PHASE / "outputs"
RAW = OUT / "raw_timing/b0_timing.json"
SUBSET = OUT / "timing_subset_manifest.json"
SCANNER_SOURCE = ROOT / "revision/model1/phase13/colab_scan_bundle/colab_scan_phase9.py"
WRAPPER_SOURCE = PHASE / "scripts/scan_phase18_timing.py"
BUNDLE_DIR = PHASE / "colab_scan_bundle"
ZIP_PATH = OUT / "phase18_colab_scan_bundle.zip"
INPUT_NAME = "b0_repetition1_scanner_input.json"
MANIFEST_NAME = "phase18_scan_transfer_manifest.json"
EXPECTED_SUBSET_SHA = "b25fc05efc915f1f6fcde7b4a78b06f771605c356ce991551a0c38b451665c3c"
EXPECTED_SCANNER_SHA = "60cedd3d6ec2d19e5e7e5ba4d709230efdd5d9478047492fefda4dd88d9afbb7"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    require(SUBSET.is_file() and sha256(SUBSET) == EXPECTED_SUBSET_SHA, "Timing subset hash mismatch")
    require(RAW.is_file(), "B0 raw timing is missing")
    body = read_json(RAW)
    require(body.get("status") in ("IN_PROGRESS", "COMPLETE"), "B0 timing status is invalid")
    subset = read_json(SUBSET)
    first = [row for row in body["records"] if int(row["repetition"]) == 1]
    expected_ids = list(map(int, subset["ordered_prompt_ids"]))
    require([int(row["prompt_id"]) for row in first] == expected_ids, "B0 repetition-1 order mismatch")
    require(SCANNER_SOURCE.is_file() and sha256(SCANNER_SOURCE) == EXPECTED_SCANNER_SHA, "Frozen scanner mismatch")
    require(WRAPPER_SOURCE.is_file(), "Phase 18 scan wrapper missing")

    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    rows = [{
        "prompt_id": int(row["prompt_id"]), "generated_code": row["generated_code"],
        "language": row["language"], "cwe_id": row["cwe_identifier"], "method": "B0",
    } for row in first]
    input_path = BUNDLE_DIR / INPUT_NAME
    atomic_json(input_path, rows)
    shutil.copy2(SCANNER_SOURCE, BUNDLE_DIR / "colab_scan_phase9.py")
    shutil.copy2(WRAPPER_SOURCE, BUNDLE_DIR / "scan_phase18_timing.py")
    manifest = {
        "schema_version": "phase18_scan_transfer_manifest_v1", "phase": 18,
        "status": "READY_NOT_STARTED", "method": "B0", "generation_repetition": 1,
        "record_count": 50, "expected_prompt_ids": expected_ids,
        "timing_subset_sha256": EXPECTED_SUBSET_SHA,
        "b0_raw_timing_path": RAW.relative_to(ROOT).as_posix(),
        "b0_repetition1_record_count": 50,
        "scanner_input": INPUT_NAME, "scanner_input_sha256": sha256(input_path),
        "scanner": "colab_scan_phase9.py", "scanner_sha256": EXPECTED_SCANNER_SHA,
        "wrapper": "scan_phase18_timing.py", "wrapper_sha256": sha256(BUNDLE_DIR / "scan_phase18_timing.py"),
        "warmup_full_scan_count": 1, "warmup_excluded": True, "measured_full_scan_repetitions": 3,
        "same_fixed_input_each_scan": True, "heldout_used": False,
    }
    atomic_json(BUNDLE_DIR / MANIFEST_NAME, manifest)
    shutil.copy2(input_path, OUT / INPUT_NAME)
    shutil.copy2(BUNDLE_DIR / MANIFEST_NAME, OUT / MANIFEST_NAME)
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in (INPUT_NAME, MANIFEST_NAME, "colab_scan_phase9.py", "scan_phase18_timing.py"):
            archive.write(BUNDLE_DIR / name, arcname=name)
    print(json.dumps({"status": "READY_NOT_STARTED", "bundle_sha256": sha256(ZIP_PATH),
                      "input_sha256": manifest["scanner_input_sha256"], "manifest_sha256": sha256(OUT / MANIFEST_NAME)}, indent=2))


if __name__ == "__main__":
    main()
