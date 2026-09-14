#!/usr/bin/env python3
"""Create the deterministic, approval-gated Phase 17 Colab generation bundle."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

from phase17_common import OUTPUTS, PHASE17, ROOT, atomic_json, read_json, require, sha256_path


BUNDLE = PHASE17 / "phase17_colab_generation_bundle.zip"
BUNDLE_MANIFEST = OUTPUTS / "phase17_generation_bundle_manifest.json"
APPROVAL = OUTPUTS / "phase17_execution_approval.json"
READINESS = OUTPUTS / "phase17_implementation_preflight.json"

FILES = (
    "revision/model1/phase17/COLAB_GENERATION_INSTRUCTIONS.md",
    "revision/model1/phase17/scripts/phase17_common.py",
    "revision/model1/phase17/scripts/run_alwayson_security.py",
    "revision/model1/phase17/scripts/run_alwayson_utility.py",
    "revision/model1/phase17/outputs/phase17_input_manifest.json",
    "revision/model1/phase17/outputs/phase17_preparation_checkpoint.json",
    "revision/model1/phase17/outputs/phase17_execution_approval.json",
    "revision/model1/phase16/outputs/phase16_checkpoint.json",
    "revision/model1/phase15/outputs/corrected_revision_freeze_manifest.json",
    "revision/model1/phase14/outputs/phase14_method_statuses.json",
    "revision/model1/phase14/outputs/bstar_alwayson_l19_config.json",
    "revision/model1/phase2/phase2_denominator_audit.json",
    "data/cyberseceval/test_prompts.json",
    "phases/phase9/colab_scan_phase9.py",
)


def deterministic_zip(path: Path, members: list[tuple[str, bytes]]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, payload in members:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, payload)


def main() -> int:
    require(APPROVAL.is_file(), "Phase 17 execution approval is missing")
    approval = read_json(APPROVAL)
    readiness = read_json(READINESS)
    require(approval.get("status") == "APPROVED_FOR_GPU_EXECUTION", "Approval status mismatch")
    require(readiness.get("status") == "PASS_APPROVED", "Execution readiness is not PASS_APPROVED")
    require(readiness.get("execution_approval_sha256") == sha256_path(APPROVAL),
            "Readiness/approval hash mismatch")
    require(readiness.get("generation_run") is False and readiness.get("model_loaded") is False,
            "Generation occurred before bundle construction")

    members: list[tuple[str, bytes]] = []
    file_records: list[dict[str, Any]] = []
    for relative in FILES:
        source = ROOT / relative
        require(source.is_file(), f"Missing bundle input: {relative}")
        payload = source.read_bytes()
        members.append((relative, payload))
        file_records.append({"path": relative, "sha256": hashlib.sha256(payload).hexdigest(),
                             "size_bytes": len(payload)})

    deterministic_zip(BUNDLE, members)
    manifest = {
        "schema_version": "phase17_generation_bundle_manifest_v1",
        "phase": 17,
        "status": "READY_FOR_COLAB_GPU_GENERATION",
        "bundle_path": str(BUNDLE.relative_to(ROOT)).replace("\\", "/"),
        "bundle_sha256": sha256_path(BUNDLE),
        "bundle_size_bytes": BUNDLE.stat().st_size,
        "file_count": len(file_records),
        "files": file_records,
        "execution_approval_sha256": sha256_path(APPROVAL),
        "execution_readiness_sha256": sha256_path(READINESS),
        "expected_generation_outputs": {
            "security": 575,
            "utility": 1716,
        },
        "scanner_run": False,
        "evaluator_run": False,
        "phase18_started": False,
    }
    atomic_json(BUNDLE_MANIFEST, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
