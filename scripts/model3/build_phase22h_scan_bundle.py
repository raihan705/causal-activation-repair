#!/usr/bin/env python
"""Build the immutable Stage 22H Colab scan bundle after generation validation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
PHASE = ROOT / "revision/model3/phase22"
OUT = PHASE / "outputs"
BUNDLE = PHASE / "phase22h_colab_scan_bundle"
ZIP_PATH = PHASE / "phase22h_colab_scan_bundle.zip"
GENERATION = OUT / "phase22h_paired_generations.json"
GENERATION_MANIFEST = OUT / "phase22h_paired_generation_manifest.json"
GENERATION_VALIDATION = OUT / "phase22h_generation_validation.json"
PROTOCOL = OUT / "phase22g_prospective_evaluation_protocol.json"
POPULATION = OUT / "phase22g_evaluation_population.json"
SCAN_INPUT = OUT / "phase22h_scan_input.json"
SCANNER = ROOT / "phases/phase9/colab_scan_phase9.py"
WRAPPER = PHASE / "scripts/scan_phase22h_manifest.py"
TRANSFER_NAME = "phase22h_scan_transfer_manifest.json"
EXPECTED = {
    GENERATION: "c6b55804084290f7efb170fdfddbddb0c2f23ef4400e170941765760e1ba1077",
    GENERATION_MANIFEST: "d9c4909333d5a22ac5d71367f247d60bddb797acde9f8f798145b3026251e30a",
    GENERATION_VALIDATION: "b2a56807a5b601438ca08ac0fa708f3d2afde26e9e503f9ffd5daa5de8b309ab",
    PROTOCOL: "ed400877e33e784f33247f82d4186f673caee0d861d5f0b5717bfc4a9afed904",
    POPULATION: "1a8ce89e42a075fd09ecc626f51b95561621c70f8c4a6942b113b9213869e2b8",
    SCANNER: "60cedd3d6ec2d19e5e7e5ba4d709230efdd5d9478047492fefda4dd88d9afbb7",
    WRAPPER: "a90a02cdaf7801581af05fb29fde04cec7fc2012d738dcf98f58f39d0e88110c",
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


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    require(not BUNDLE.exists() and not ZIP_PATH.exists() and not SCAN_INPUT.exists(), "immutable Stage 22H bundle/input already exists")
    for path, expected in EXPECTED.items():
        require(path.is_file() and sha256_file(path) == expected, f"frozen artifact mismatch: {path.name}")
    generation = json.loads(GENERATION.read_text(encoding="utf-8"))
    validation = json.loads(GENERATION_VALIDATION.read_text(encoding="utf-8"))
    records = generation["records"]
    require(generation["status"] == "COMPLETE" and validation["status"] == "PASS" and len(records) == 468, "generation gate not passed")
    require(all(row["generation_status"] == "SUCCESS" and row["validity"]["is_valid"] for row in records), "failed/invalid generation in projection")

    scan_rows = [{
        "record_index": row["record_index"],
        "record_id": row["record_id"],
        "condition": row["condition"],
        "seed": row["seed"],
        "prompt_id": row["prompt_id"],
        "target_cwe": row["cwe_identifier"],
        "cwe_identifier": row["cwe_identifier"],
        "language": row["language"],
        "layer": row["layer"],
        "feature_id": row["feature_id"],
        "alpha": row["alpha"],
        "generation_valid": row["validity"]["is_valid"],
        "generation_invalid_reason": row["validity"]["invalid_reason"],
        "generated_text_sha256": row["generated_text_sha256"],
        "generated_code": row["generated_code"],
    } for row in records]
    require(Counter(row["condition"] for row in scan_rows) == Counter({"B0": 234, "MODEL3_CWE_LABEL_ROUTED": 234}), "condition projection mismatch")
    require(Counter(int(row["seed"]) for row in scan_rows) == Counter({42: 156, 43: 156, 44: 156}), "seed projection mismatch")
    atomic_json(SCAN_INPUT, scan_rows)

    BUNDLE.mkdir(parents=True)
    for source, name in ((SCAN_INPUT, SCAN_INPUT.name), (SCANNER, "colab_scan_phase9.py"), (WRAPPER, WRAPPER.name)):
        shutil.copy2(source, BUNDLE / name)
    record_ids = [row["record_id"] for row in scan_rows]
    transfer = {
        "schema_version": "phase22h_scan_transfer_v1",
        "status": "FROZEN_BEFORE_SCAN",
        "generation_sha256": sha256_file(GENERATION),
        "generation_manifest_sha256": sha256_file(GENERATION_MANIFEST),
        "generation_validation_sha256": sha256_file(GENERATION_VALIDATION),
        "protocol_sha256": sha256_file(PROTOCOL),
        "population_sha256": sha256_file(POPULATION),
        "scan_input_filename": SCAN_INPUT.name,
        "scan_input_sha256": sha256_file(SCAN_INPUT),
        "record_count": 468,
        "record_ids": record_ids,
        "record_ids_canonical_sha256": canonical_sha256(record_ids),
        "condition_counts": {"B0": 234, "MODEL3_CWE_LABEL_ROUTED": 234},
        "seed_counts": {"42": 156, "43": 156, "44": 156},
        "target_counts": {"CWE-120": 288, "CWE-327": 120, "CWE-89": 60},
        "invalid_generation_count": 0,
        "scanner_filename": "colab_scan_phase9.py",
        "scanner_sha256": sha256_file(SCANNER),
        "wrapper_filename": WRAPPER.name,
        "wrapper_sha256": sha256_file(WRAPPER),
        "required_semgrep_version": "1.175.0",
        "registry_config": "p/security-audit",
        "registry_snapshot_hash": "NOT_CONTENT_PINNED",
        "checkpoint_interval": 10,
        "expected_scan_output_filename": "phase22h_scans.json",
        "expected_return_manifest_filename": "phase22h_scan_return_manifest.json",
        "scanner_rules_changed": False,
        "generation_run_in_colab": False,
        "heldout_split": True,
    }
    atomic_json(BUNDLE / TRANSFER_NAME, transfer)
    (BUNDLE / "README.txt").write_text(
        "Stage 22H immutable paired held-out scan. Linux/Colab and exact Semgrep 1.175.0 required. "
        "Checkpoint every 10 rows. Return phase22h_scans.json and phase22h_scan_return_manifest.json.\n",
        encoding="utf-8",
        newline="\n",
    )
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(BUNDLE.iterdir(), key=lambda item: item.name):
            archive.write(path, arcname=path.name)
    bundle_manifest = {
        "schema_version": "phase22h_scan_bundle_v1",
        "status": "READY_FOR_COLAB",
        "bundle_zip": str(ZIP_PATH.relative_to(ROOT)).replace("\\", "/"),
        "bundle_zip_sha256": sha256_file(ZIP_PATH),
        "transfer_manifest_sha256": sha256_file(BUNDLE / TRANSFER_NAME),
        "scan_input_sha256": sha256_file(SCAN_INPUT),
        "wrapper_sha256": sha256_file(WRAPPER),
        "scanner_sha256": sha256_file(SCANNER),
        "record_count": 468,
        "return_files": ["phase22h_scans.json", "phase22h_scan_return_manifest.json"],
        "actual_scan_run": False,
        "heldout_split": True,
    }
    atomic_json(OUT / "phase22h_scan_bundle_manifest.json", bundle_manifest)
    print(json.dumps(bundle_manifest, indent=2))


if __name__ == "__main__":
    main()
