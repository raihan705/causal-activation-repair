#!/usr/bin/env python
"""Build immutable Stage 22E steered-output Colab scanner handoff."""

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
BUNDLE = PHASE / "phase22e_steered_colab_scan_bundle"
ZIP_PATH = PHASE / "phase22e_steered_colab_scan_bundle.zip"
GENERATION = OUT / "phase22e_steered_generations.json"
GENERATION_MANIFEST = OUT / "phase22e_steered_generation_manifest.json"
GENERATION_VALIDATION = OUT / "phase22e_steered_generation_validation.json"
EXECUTION = OUT / "phase22e_steering_execution_protocol.json"
DENOMINATOR = OUT / "phase22e_paired_denominator_manifest.json"
SCAN_INPUT = OUT / "phase22e_steered_scan_input.json"
SCANNER = ROOT / "phases/phase9/colab_scan_phase9.py"
WRAPPER = PHASE / "scripts/scan_phase22e_steered.py"
TRANSFER_NAME = "phase22e_steered_scan_transfer_manifest.json"
EXPECTED = {
    GENERATION: "c1844a73f468dd88ce608418ea8ce0cd2802d63a1ce2589ea89c77866917a727",
    GENERATION_MANIFEST: "41fbc2934724fb5c42d7dafba26d93279c0d3185344c5d8aafa208e618e419b7",
    GENERATION_VALIDATION: "394564564cdf9a8cef084fb645896021ccbe15ab63e5ecf70035e235a67b092e",
    EXECUTION: "86d784318cd7d4b6d4cf770f42650aa4f44981f30b269d49e59d678274a088a3",
    DENOMINATOR: "30921b90579dbb9d350c256f90f720332bf41d12f6c4ce6c730ea61240ec7dde",
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
    require(not BUNDLE.exists() and not ZIP_PATH.exists() and not SCAN_INPUT.exists(), "immutable scan bundle/input already exists")
    for path, expected in EXPECTED.items():
        require(path.is_file() and sha256_file(path) == expected, f"frozen artifact mismatch: {path.name}")
    require(WRAPPER.is_file(), "scan wrapper missing")
    generation = json.loads(GENERATION.read_text(encoding="utf-8"))
    validation = json.loads(GENERATION_VALIDATION.read_text(encoding="utf-8"))
    records = generation["records"]
    require(generation["status"] == "COMPLETE" and validation["status"] == "PASS", "generation validation gate not passed")
    require(len(records) == 1170 and validation["record_count"] == 1170, "generation count mismatch")
    scan_rows = [{
        "record_index": x["record_index"], "record_id": x["record_id"],
        "prompt_id": x["prompt_id"], "target_cwe": x["target_cwe"],
        "layer": x["layer"], "feature_id": x["feature_id"],
        "statistical_rank": x["statistical_rank"], "screening_alpha": x["screening_alpha"],
        "arm": x["arm"], "language": x["language"], "generated_code": x["generated_code"],
    } for x in records]
    atomic_json(SCAN_INPUT, scan_rows)
    BUNDLE.mkdir(parents=True)
    for source, name in ((SCAN_INPUT, SCAN_INPUT.name), (SCANNER, "colab_scan_phase9.py"), (WRAPPER, WRAPPER.name)):
        shutil.copy2(source, BUNDLE / name)
    transfer = {
        "schema_version": "phase22e_steered_scan_transfer_v1", "status": "FROZEN_BEFORE_SCAN",
        "generation_sha256": sha256_file(GENERATION), "generation_manifest_sha256": sha256_file(GENERATION_MANIFEST),
        "generation_validation_sha256": sha256_file(GENERATION_VALIDATION),
        "execution_protocol_sha256": sha256_file(EXECUTION), "paired_denominator_sha256": sha256_file(DENOMINATOR),
        "scan_input_filename": SCAN_INPUT.name, "scan_input_sha256": sha256_file(SCAN_INPUT),
        "record_count": 1170, "record_ids": [x["record_id"] for x in scan_rows],
        "record_ids_canonical_sha256": hashlib.sha256(json.dumps([x["record_id"] for x in scan_rows], sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "target_counts": {"CWE-120": 540, "CWE-327": 330, "CWE-89": 300},
        "scanner_filename": "colab_scan_phase9.py", "scanner_sha256": sha256_file(SCANNER),
        "wrapper_filename": WRAPPER.name, "wrapper_sha256": sha256_file(WRAPPER),
        "required_semgrep_version": "1.175.0", "registry_config": "p/security-audit",
        "registry_snapshot_hash": "NOT_CONTENT_PINNED", "checkpoint_interval": 10,
        "expected_scan_output_filename": "phase22e_steered_scans.json",
        "expected_return_manifest_filename": "phase22e_steered_scan_return_manifest.json",
        "scanner_rules_changed": False, "generation_run": False, "heldout_used": False,
    }
    atomic_json(BUNDLE / TRANSFER_NAME, transfer)
    (BUNDLE / "README.txt").write_text(
        "Stage 22E immutable steered-output scan. Linux/Colab and exact Semgrep 1.175.0 required. "
        "Run scan_phase22e_steered.py with --workdir and --resume after interruption. Return "
        "phase22e_steered_scans.json and phase22e_steered_scan_return_manifest.json.\n",
        encoding="utf-8", newline="\n",
    )
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(BUNDLE.iterdir(), key=lambda x: x.name):
            archive.write(path, arcname=path.name)
    bundle_manifest = {
        "schema_version": "phase22e_steered_scan_bundle_v1", "status": "READY_FOR_COLAB",
        "bundle_zip": str(ZIP_PATH.relative_to(ROOT)).replace("\\", "/"), "bundle_zip_sha256": sha256_file(ZIP_PATH),
        "transfer_manifest_sha256": sha256_file(BUNDLE / TRANSFER_NAME),
        "scan_input_sha256": sha256_file(SCAN_INPUT), "wrapper_sha256": sha256_file(WRAPPER),
        "scanner_sha256": sha256_file(SCANNER), "record_count": 1170,
        "return_files": ["phase22e_steered_scans.json", "phase22e_steered_scan_return_manifest.json"],
        "actual_scan_run": False, "heldout_used": False,
    }
    atomic_json(OUT / "phase22e_steered_scan_bundle_manifest.json", bundle_manifest)
    print(json.dumps(bundle_manifest, indent=2))


if __name__ == "__main__":
    main()
