#!/usr/bin/env python3
"""Build the immutable Phase21 causal-steered Linux scanner handoff."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
PHASE = ROOT / "revision/model2/phase21"
OUT = PHASE / "outputs"
SCRIPTS = PHASE / "scripts"
BUNDLE = PHASE / "model2_causal_scan_bundle"
ZIP_PATH = OUT / "model2_causal_scan_bundle.zip"

GENERATION = OUT / "model2_causal_steered_generations.json"
GENERATION_MANIFEST = OUT / "model2_causal_steered_generation_manifest.json"
GENERATION_VALIDATION = OUT / "model2_causal_steered_generation_validation.json"
SCANNER_RUNNER = SCRIPTS / "scan_model2_causal_steered.py"
FROZEN_SCANNER = ROOT / "phases/phase9/colab_scan_phase9.py"
SCANNER_PROVENANCE = OUT / "model2_scanner_provenance.json"
TRANSFER_NAME = "model2_causal_scan_transfer_manifest.json"

EXPECTED = {
    GENERATION: "2c5f4937b63c1e710dd2c949fee8a8179172eb9426e68880948f609b53365c6a",
    GENERATION_MANIFEST: "12237b2dfb6a41af36dfd171ca97113f459720d49b20f8bded5e18c22d1080a8",
    FROZEN_SCANNER: "60cedd3d6ec2d19e5e7e5ba4d709230efdd5d9478047492fefda4dd88 contested",
    SCANNER_PROVENANCE: "e249982d929f13c7a0cda0211f73e4cd721331b98cf483e9b205f3c4936d64a9",
}
EXPECTED[FROZEN_SCANNER] = "60cedd3d6ec2d19e5e7e5ba4d709230efdd5d9478047492fefda4dd88d9afbb7"

PROTOCOL_CHAIN = {
    "model1_fidelity_candidate_manifest_sha256": "be072afa3a9c58c20729bcd5deb343007f15b7c981e4fe02bb857bc17d00b401",
    "causal_validation_protocol_sha256": "b3aac3b21f52270c231e1c077adeb0cc1ef2a2ebce1f156c08a13df2695d414e",
    "causal_metric_resolution_sha256": "8474d6d2eb67f2acf0b40ff0407b50cec2a4cbe1e4a1ee7a45426b70f3cf2868",
    "causal_execution_protocol_v2_sha256": "ae790dd42538ac172dd7ca7b178840afa794dc371c1da015ba59a4cdecfd88f8",
    "causal_execution_protocol_v3_sha256": "a2a0da4393a5826aa6dae9c7b2ec1c3dd08454768aa2af2ab78a730f5e8e3153",
    "causal_baseline_generation_sha256": "3940c731ef49b68409fd168a0d1afadc100146240fba598ac80aa91939fe78cc",
    "causal_baseline_scan_v2_sha256": "232b953815c6cfc3e17501add22b18f090be725541cf18b408c06b73d6da115a",
    "causal_denominator_manifest_v2_sha256": "66f1ff6a0f0f927cd0adc779956cdb85a7bdcd250b5b927f128a5434b884165c",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def load_scanner(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("phase21_bundle_frozen_scanner", path)
    require(spec is not None and spec.loader is not None, "cannot import frozen scanner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    for path, expected in EXPECTED.items():
        require(path.is_file(), f"missing frozen input: {path}")
        require(sha256_file(path) == expected, f"CAUSAL_PROVENANCE_HASH_MISMATCH:{path.name}")
    require(SCANNER_RUNNER.is_file(), "missing causal scanner runner")
    require(GENERATION_VALIDATION.is_file(), "missing generation validation")
    validation = json.loads(GENERATION_VALIDATION.read_text(encoding="utf-8"))
    require(validation.get("status") == "PASS", "generation validation is not PASS")
    require(validation.get("generation_sha256") == EXPECTED[GENERATION], "generation-validation binding mismatch")
    require(validation.get("errors") == [], "generation validation contains errors")
    generation = json.loads(GENERATION.read_text(encoding="utf-8"))
    records = generation.get("records", [])
    require(len(records) == 1620, "generation record count mismatch")
    require(len({row["condition_key"] for row in records}) == 1620, "generation key count mismatch")
    require(len({(row["target_cwe"], row["layer"], row["feature_id"]) for row in records}) == 90, "condition count mismatch")

    require(not BUNDLE.exists(), f"bundle already exists: {BUNDLE}")
    require(not ZIP_PATH.exists(), f"bundle zip already exists: {ZIP_PATH}")
    BUNDLE.mkdir(parents=True)
    copies = [
        (GENERATION, GENERATION.name),
        (GENERATION_MANIFEST, GENERATION_MANIFEST.name),
        (GENERATION_VALIDATION, GENERATION_VALIDATION.name),
        (SCANNER_RUNNER, SCANNER_RUNNER.name),
        (FROZEN_SCANNER, "colab_scan_phase9_frozen.py"),
        (SCANNER_PROVENANCE, SCANNER_PROVENANCE.name),
    ]
    for source, name in copies:
        shutil.copy2(source, BUNDLE / name)

    frozen = load_scanner(BUNDLE / "colab_scan_phase9_frozen.py")
    rule_hashes = {
        "rule_cwe_map_sha256": sha256_json(frozen.RULE_CWE_MAP),
        "regex_patterns_sha256": sha256_json(frozen.REGEX_PATTERNS),
        "target_languages_sha256": sha256_json(sorted(frozen.TARGET_LANGUAGES)),
        "all_target_cwes_sha256": sha256_json(sorted(frozen.ALL_TARGET_CWES)),
    }
    immutable = [
        {"name": name, "sha256": sha256_file(BUNDLE / name), "size_bytes": (BUNDLE / name).stat().st_size}
        for _, name in copies
    ]
    manifest = {
        "schema_version": "phase21_model2_causal_scan_transfer_manifest_v1",
        "status": "FROZEN_WAITING_CAUSAL_SCAN",
        "required_semgrep_version": "1.175.0",
        "registry_config": "p/security-audit",
        "registry_snapshot_hash": "NOT_CONTENT_PINNED",
        "accepted_scanner_identity": "frozen Phase9 scanner used for Model2 baseline v2",
        "generation_input": {"name": GENERATION.name, "sha256": sha256_file(GENERATION)},
        "generation_manifest": {"name": GENERATION_MANIFEST.name, "sha256": sha256_file(GENERATION_MANIFEST)},
        "generation_validation": {"name": GENERATION_VALIDATION.name, "sha256": sha256_file(GENERATION_VALIDATION)},
        "scanner_runner": {"name": SCANNER_RUNNER.name, "sha256": sha256_file(SCANNER_RUNNER)},
        "frozen_scanner_source": {"name": "colab_scan_phase9_frozen.py", "sha256": sha256_file(FROZEN_SCANNER)},
        "scanner_provenance": {"name": SCANNER_PROVENANCE.name, "sha256": sha256_file(SCANNER_PROVENANCE)},
        "scanner_rule_hashes": rule_hashes,
        "protocol_chain": PROTOCOL_CHAIN,
        "immutable_files": immutable,
        "expected_record_count": 1620,
        "expected_condition_count": 90,
        "expected_target_record_counts": {"CWE-120": 570, "CWE-327": 600, "CWE-89": 450},
        "logical_key_fields": ["target_cwe", "layer", "feature_id", "prompt_id", "role", "alpha", "seed"],
        "logical_key_format": "{target_cwe}|L{layer}|F{feature_id}|P{prompt_id}|{role}|A20.0|S42",
        "scanner_record_requirements": [
            "scanner_success", "scanner_eligible", "findings", "mapped_cwe_labels",
            "target_cwe_present", "any_vulnerability", "scanner_warnings",
            "scanner_timeout", "scanner_error", "scanner_exit_status",
        ],
        "fail_closed": True,
        "exit_zero_warnings_are_process_failures": False,
        "code_transformation": "NONE",
        "resume_checkpoint_interval": 10,
        "heldout_accessed": False,
        "model1_modified": False,
    }
    manifest["self_sha256_excluding_field"] = sha256_json(manifest)
    transfer_path = BUNDLE / TRANSFER_NAME
    write_json(transfer_path, manifest)
    (BUNDLE / "README.txt").write_text(
        "Phase21 Model2 causal-steered immutable scan. Requires Linux/Colab and Semgrep 1.175.0. "
        "No generation or code transformation. The scanner is fail-closed and resumable. Return "
        "model2_causal_steered_scans.json and model2_causal_scan_return_manifest.json.\n",
        encoding="utf-8", newline="\n",
    )
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(BUNDLE.iterdir(), key=lambda item: item.name):
            archive.write(path, arcname=path.name)
    result = {
        "status": "COMPLETE",
        "bundle_path": str(BUNDLE.relative_to(ROOT)).replace("\\", "/"),
        "zip_path": str(ZIP_PATH.relative_to(ROOT)).replace("\\", "/"),
        "zip_sha256": sha256_file(ZIP_PATH),
        "transfer_manifest_sha256": sha256_file(transfer_path),
        "transfer_canonical_sha256": manifest["self_sha256_excluding_field"],
        "generation_sha256": sha256_file(GENERATION),
        "generation_validation_sha256": sha256_file(GENERATION_VALIDATION),
        "scanner_runner_sha256": sha256_file(SCANNER_RUNNER),
        "frozen_scanner_source_sha256": sha256_file(FROZEN_SCANNER),
        "scanner_rule_hashes": rule_hashes,
        "record_count": 1620,
        "condition_count": 90,
        "return_files": ["model2_causal_steered_scans.json", "model2_causal_scan_return_manifest.json"],
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"PHASE21_CAUSAL_SCAN_BUNDLE_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
