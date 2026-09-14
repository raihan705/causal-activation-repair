#!/usr/bin/env python
"""Validate paired baselines and build the immutable Stage 22E Colab scan bundle."""

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
BUNDLE = PHASE / "phase22e_baseline_colab_scan_bundle"
ZIP_PATH = PHASE / "phase22e_baseline_colab_scan_bundle.zip"
PROTOCOL = OUT / "phase22e_causal_validation_protocol.json"
GENERATION = OUT / "phase22e_paired_baseline_generations.json"
GEN_MANIFEST = OUT / "phase22e_paired_baseline_generation_manifest.json"
RUN_MANIFEST = OUT / "phase22e_paired_baseline_run_manifest.json"
VALIDATION = OUT / "phase22e_paired_baseline_generation_validation.json"
SCAN_INPUT = OUT / "phase22e_paired_baseline_scan_input.json"
SCANNER = ROOT / "phases/phase9/colab_scan_phase9.py"
WRAPPER = PHASE / "scripts/scan_phase22e_paired_baseline.py"
TRANSFER_NAME = "phase22e_paired_baseline_scan_transfer_manifest.json"
EXPECTED_GENERATION_SHA256 = "e12e46bacb0fc3725e17e4e6258f71cc3f77001d2fe7b646b88b05408b684a26"


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
    for path in (PROTOCOL, GENERATION, GEN_MANIFEST, RUN_MANIFEST, SCANNER, WRAPPER):
        require(path.is_file(), f"missing required file: {path}")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    container = json.loads(GENERATION.read_text(encoding="utf-8"))
    manifest = json.loads(GEN_MANIFEST.read_text(encoding="utf-8"))
    run = json.loads(RUN_MANIFEST.read_text(encoding="utf-8"))
    records = container["records"]
    require(sha256_file(GENERATION) == EXPECTED_GENERATION_SHA256 == manifest["output_sha256"] == run["output_sha256"], "generation hash mismatch")
    require(container["protocol_sha256"] == manifest["protocol_sha256"] == sha256_file(PROTOCOL), "protocol provenance mismatch")
    require(manifest["status"] == run["status"] == "COMPLETE", "generation incomplete")
    require(len(records) == 40 and manifest["success_count"] == 40 and manifest["failure_count"] == 0 and manifest["invalid_count"] == 0, "paired baseline completion failure")
    plan = protocol["paired_baseline_prompts"]
    require([x["record_id"] for x in records] == [x["record_id"] for x in plan], "record plan order mismatch")
    require(len({int(x["prompt_id"]) for x in records}) == 40, "duplicate paired prompt ID")
    require(all(x["generated_code"].strip() and x["validity"]["is_valid"] for x in records), "invalid/empty scan input")
    scan_rows = [{
        "record_id": x["record_id"], "prompt_id": int(x["prompt_id"]),
        "cwe_id": x["target_cwe"], "language": x["language"],
        "generated_code": x["generated_code"],
    } for x in records]
    atomic_json(SCAN_INPUT, scan_rows)
    validation = {
        "schema_version": "phase22e_paired_baseline_generation_validation_v1", "status": "PASS",
        "protocol_sha256": sha256_file(PROTOCOL), "generation_sha256": sha256_file(GENERATION),
        "generation_manifest_sha256": sha256_file(GEN_MANIFEST), "run_manifest_sha256": sha256_file(RUN_MANIFEST),
        "scan_input_sha256": sha256_file(SCAN_INPUT), "record_count": 40,
        "success_count": 40, "invalid_count": 0, "unique_prompt_ids": True,
        "source_order_matches_frozen_plan": True, "sae_loaded": False,
        "steering_run": False, "scanner_run": False, "heldout_used": False,
    }
    atomic_json(VALIDATION, validation)
    require(not BUNDLE.exists() and not ZIP_PATH.exists(), "baseline scan bundle already exists")
    BUNDLE.mkdir(parents=True)
    shutil.copy2(SCAN_INPUT, BUNDLE / SCAN_INPUT.name)
    shutil.copy2(SCANNER, BUNDLE / "colab_scan_phase9.py")
    shutil.copy2(WRAPPER, BUNDLE / WRAPPER.name)
    transfer = {
        "schema_version": "phase22e_paired_baseline_scan_transfer_v1", "status": "FROZEN_BEFORE_SCAN",
        "generation_sha256": sha256_file(GENERATION), "generation_manifest_sha256": sha256_file(GEN_MANIFEST),
        "validation_sha256": sha256_file(VALIDATION), "protocol_sha256": sha256_file(PROTOCOL),
        "scan_input_filename": SCAN_INPUT.name, "scan_input_sha256": sha256_file(SCAN_INPUT),
        "record_count": 40, "record_ids": [x["record_id"] for x in records],
        "prompt_ids": [int(x["prompt_id"]) for x in records],
        "scanner_filename": "colab_scan_phase9.py", "scanner_sha256": sha256_file(SCANNER),
        "wrapper_filename": WRAPPER.name, "wrapper_sha256": sha256_file(WRAPPER),
        "required_semgrep_version": "1.175.0",
        "expected_scan_output_filename": "phase22e_paired_baseline_scans.json",
        "expected_return_manifest_filename": "phase22e_paired_baseline_scan_return_manifest.json",
        "scanner_rules_changed": False, "steering_run": False, "heldout_used": False,
    }
    atomic_json(BUNDLE / TRANSFER_NAME, transfer)
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(BUNDLE.iterdir(), key=lambda x: x.name):
            zf.write(path, arcname=path.name)
    bundle_manifest = {
        "schema_version": "phase22e_paired_baseline_scan_bundle_v1", "status": "READY_FOR_COLAB",
        "bundle_zip": "revision/model3/phase22/phase22e_baseline_colab_scan_bundle.zip",
        "bundle_zip_sha256": sha256_file(ZIP_PATH),
        "transfer_manifest_sha256": sha256_file(BUNDLE / TRANSFER_NAME),
        "files": {path.name: sha256_file(path) for path in sorted(BUNDLE.iterdir(), key=lambda x: x.name)},
        "actual_scan_run": False, "steering_run": False, "heldout_used": False,
    }
    atomic_json(OUT / "phase22e_paired_baseline_scan_bundle_manifest.json", bundle_manifest)
    print(json.dumps({"validation": validation, "bundle": bundle_manifest}, indent=2))


if __name__ == "__main__":
    main()
