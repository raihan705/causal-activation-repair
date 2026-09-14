#!/usr/bin/env python
"""Build the immutable warning-aware v2 Colab rescan bundle."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
PHASE = ROOT / "revision/model2/phase21"
OUT = PHASE / "outputs"
SCRIPTS = PHASE / "scripts"
BUNDLE = PHASE / "model2_causal_baseline_scan_bundle_v2"
ZIP = OUT / "model2_causal_baseline_scan_bundle_v2.zip"
GENERATION = OUT / "model2_causal_baseline_generations.json"
GENERATION_MANIFEST = OUT / "model2_causal_baseline_generation_manifest.json"
AMENDMENT = OUT / "model2_causal_baseline_scanner_warning_amendment_v2.json"
V1_SCAN = OUT / "model2_causal_baseline_scans.json"
SCANNER = SCRIPTS / "scan_model2_causal_baselines_v2.py"
FROZEN_SCANNER = ROOT / "phases/phase9/colab_scan_phase9.py"
TRANSFER = "model2_causal_baseline_scan_transfer_manifest_v2.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    expected = {
        GENERATION: "3940c731ef49b68409fd168a0d1afadc100146240fba598ac80aa91939fe78cc",
        GENERATION_MANIFEST: "6f69b231115251a4a6033e0949b63c61e4ecd831e011cd63c13a1e6c8133bd1c",
        V1_SCAN: "e3ed20e2f53072eece94edeae9fdc008eb4142c49d83d57b57303e4e75de261b",
        SCANNER: "9e704162aaa7cc9bda515f02d99a95f021741a98b8fa3f65b18bf59176ccf9cb",
        FROZEN_SCANNER: "60cedd3d6ec2d19e5e7e5ba4d709230efdd5d9478047492fefda4dd88d9afbb7",
    }
    for path, digest in expected.items():
        require(path.is_file() and sha256_file(path) == digest, f"missing/drifted frozen file: {path}")
    require(AMENDMENT.is_file(), "warning amendment is missing")
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    require(amendment["status"] == "FROZEN_BEFORE_RESCAN", "warning amendment is not frozen")
    require(amendment["runner"]["sha256"] == sha256_file(SCANNER), "amendment/runner mismatch")
    require(amendment["code_transformation"]["generated_text_modified"] is False, "unauthorized code transformation")
    generation = json.loads(GENERATION.read_text(encoding="utf-8"))
    require(len(generation["records"]) == 45, "generation cardinality mismatch")

    require(not BUNDLE.exists(), f"v2 bundle directory already exists: {BUNDLE}")
    require(not ZIP.exists(), f"v2 bundle zip already exists: {ZIP}")
    BUNDLE.mkdir(parents=True)
    copies = [
        (GENERATION, "model2_causal_baseline_generations.json"),
        (GENERATION_MANIFEST, "model2_causal_baseline_generation_manifest.json"),
        (AMENDMENT, "model2_causal_baseline_scanner_warning_amendment_v2.json"),
        (SCANNER, "scan_model2_causal_baselines_v2.py"),
        (FROZEN_SCANNER, "colab_scan_phase9_frozen.py"),
    ]
    for source, name in copies:
        shutil.copy2(source, BUNDLE / name)
    immutable = [{"name": name, "sha256": sha256_file(BUNDLE / name), "size_bytes": (BUNDLE / name).stat().st_size} for _, name in copies]
    manifest = {
        "schema_version": "phase21_model2_causal_baseline_scan_transfer_manifest_v2",
        "status": "FROZEN_FOR_WARNING_AWARE_RESCAN",
        "required_semgrep_version": "1.175.0",
        "registry_config": "p/security-audit",
        "generation_input": {"name": GENERATION.name, "sha256": sha256_file(GENERATION)},
        "generation_manifest": {"name": GENERATION_MANIFEST.name, "sha256": sha256_file(GENERATION_MANIFEST)},
        "warning_amendment": {"name": AMENDMENT.name, "sha256": sha256_file(AMENDMENT)},
        "preserved_v1_scan": {"name": V1_SCAN.name, "sha256": sha256_file(V1_SCAN)},
        "scanner_runner": {"name": SCANNER.name, "sha256": sha256_file(SCANNER)},
        "frozen_scanner_source": {"name": "colab_scan_phase9_frozen.py", "sha256": sha256_file(FROZEN_SCANNER)},
        "immutable_files": immutable,
        "expected_physical_records": 45,
        "expected_logical_records": 45,
        "warning_policy": "exit-0 level:warn is preserved and is not process failure",
        "fatal_policy": "timeout, exception, nonzero exit, invalid JSON, or non-warning reported error fails closed",
        "code_transformation": "NONE",
        "feature_steering_authorized": False,
        "heldout_accessed": False,
    }
    write_json(BUNDLE / TRANSFER, manifest)
    (BUNDLE / "README.txt").write_text(
        "Phase21 causal-baseline warning-aware v2 rescan. No generation, code transformation, SAE, or steering. Requires Semgrep 1.175.0. Return model2_causal_baseline_scans_v2.json and model2_causal_baseline_scan_return_manifest_v2.json.\n",
        encoding="utf-8", newline="\n"
    )
    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(BUNDLE.iterdir(), key=lambda x: x.name):
            zf.write(path, arcname=path.name)
    print(json.dumps({
        "status": "COMPLETE",
        "zip_path": str(ZIP.relative_to(ROOT)).replace("\\", "/"),
        "zip_sha256": sha256_file(ZIP),
        "transfer_manifest_sha256": sha256_file(BUNDLE / TRANSFER),
        "amendment_sha256": sha256_file(AMENDMENT),
        "generation_sha256": sha256_file(GENERATION),
        "return_files": ["model2_causal_baseline_scans_v2.json", "model2_causal_baseline_scan_return_manifest_v2.json"]
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
