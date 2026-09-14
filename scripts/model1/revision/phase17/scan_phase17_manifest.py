#!/usr/bin/env python3
"""Run the single frozen Phase 17 security scan in a Linux/Colab handoff."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_RECORDS = 575
EXPECTED_SCANNER_SHA256 = "60cedd3d6ec2d19e5e7e5ba4d709230efdd5d9478047492fefda4dd88d9afbb7"
EXPECTED_INPUT_SHA256 = "0985b3c034aa77bc948612c9658c4852aea68d5afa2e3609ae08ee4dab0ee069"
RETURN_NAME = "phase17_scan_return_manifest.json"
OUTPUT_NAME = "alwayson_security_seed42_icd.json"


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def semgrep_version(required: bool) -> str:
    result = subprocess.run(
        ["semgrep", "--version"], capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=30, check=False,
    )
    if result.returncode != 0:
        if required:
            raise ValidationError(f"Semgrep version check failed: {result.stderr.strip()}")
        return "NOT_AVAILABLE_PREFLIGHT_ONLY"
    return (result.stdout or result.stderr).strip()


def validate_icd(path: Path, expected_ids: list[int]) -> dict[str, int]:
    rows = read_json(path)
    require(isinstance(rows, list) and len(rows) == EXPECTED_RECORDS, "ICD count mismatch")
    require([int(row.get("prompt_id")) for row in rows] == expected_ids, "ICD prompt order mismatch")
    required = {"prompt_id", "cwe_id", "language", "findings", "vulnerable_cwes", "is_vulnerable", "skipped"}
    for index, row in enumerate(rows):
        require(isinstance(row, dict) and required.issubset(row), f"ICD row {index} schema mismatch")
        require(isinstance(row["is_vulnerable"], bool), f"ICD row {index} vulnerability is not Boolean")
        require(isinstance(row["skipped"], bool), f"ICD row {index} skipped is not Boolean")
    skipped = sum(row["skipped"] for row in rows)
    vulnerable = sum(row["is_vulnerable"] for row in rows)
    return {"record_count": len(rows), "eligible_count": len(rows) - skipped,
            "skipped_count": skipped, "raw_vulnerable_count": vulnerable}


def load_scanner(path: Path):
    spec = importlib.util.spec_from_file_location("phase17_frozen_scanner", path)
    require(spec is not None and spec.loader is not None, "Cannot import frozen scanner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(callable(getattr(module, "scan_file", None)), "Frozen scanner lacks scan_file")
    return module.scan_file


def preflight(workdir: Path, require_semgrep: bool) -> dict[str, Any]:
    manifest_path = workdir / "phase17_scan_transfer_manifest.json"
    input_path = workdir / "alwayson_security_seed42_scanner_input.json"
    scanner_path = workdir / "colab_scan_phase9.py"
    for path in (manifest_path, input_path, scanner_path):
        require(path.is_file(), f"Missing required file: {path.name}")
    transfer = read_json(manifest_path)
    require(transfer.get("status") == "READY_NOT_STARTED", "Transfer status mismatch")
    require(transfer.get("method") == "B*-AlwaysOn-L19", "Method mismatch")
    require(transfer.get("seed") == 42 and transfer.get("record_count") == EXPECTED_RECORDS,
            "Seed or population mismatch")
    require(transfer.get("generation_runner_consumed_cwe_metadata") is False,
            "Generation routing metadata guard failed")
    require(transfer.get("scanner_input_sha256") == EXPECTED_INPUT_SHA256, "Manifest input hash mismatch")
    require(transfer.get("scanner_sha256") == EXPECTED_SCANNER_SHA256, "Manifest scanner hash mismatch")
    require(sha256(input_path) == EXPECTED_INPUT_SHA256, "Scanner input hash mismatch")
    require(sha256(scanner_path) == EXPECTED_SCANNER_SHA256, "Frozen scanner hash mismatch")
    rows = read_json(input_path)
    expected_ids = [int(value) for value in transfer.get("expected_prompt_ids", [])]
    require(isinstance(rows, list) and len(rows) == EXPECTED_RECORDS, "Scanner input count mismatch")
    require([int(row.get("prompt_id")) for row in rows] == expected_ids, "Scanner input order mismatch")
    return {
        "status": "PASS", "record_count": EXPECTED_RECORDS,
        "manifest_sha256": sha256(manifest_path), "scanner_input_sha256": sha256(input_path),
        "scanner_sha256": sha256(scanner_path), "wrapper_sha256": sha256(Path(__file__).resolve()),
        "semgrep_version": semgrep_version(require_semgrep), "platform": platform.platform(),
        "python_version": platform.python_version(), "expected_ids": expected_ids,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, default=Path("."))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    workdir = args.workdir.resolve()
    state = preflight(workdir, require_semgrep=not args.preflight_only)
    atomic_json(workdir / "phase17_scan_preflight.json", {k: v for k, v in state.items() if k != "expected_ids"})
    print(f"PREFLIGHT PASS: {EXPECTED_RECORDS} records; Semgrep {state['semgrep_version']}")
    if args.preflight_only:
        return 0

    output = workdir / OUTPUT_NAME
    return_path = workdir / RETURN_NAME
    if output.is_file():
        require(args.resume, f"{OUTPUT_NAME} exists; use --resume to validate and reuse")
        counts = validate_icd(output, state["expected_ids"])
    else:
        temporary = workdir / f".{OUTPUT_NAME}.scan-tmp"
        if temporary.exists():
            temporary.unlink()
        load_scanner(workdir / "colab_scan_phase9.py")(
            workdir / "alwayson_security_seed42_scanner_input.json", temporary
        )
        require(temporary.is_file(), "Frozen scanner did not create its output")
        counts = validate_icd(temporary, state["expected_ids"])
        os.replace(temporary, output)

    result = {
        "schema_version": "phase17_scan_return_manifest_v1", "status": "SCANS_COMPLETE",
        "scan_status": "COMPLETE", "completed_at_utc": utc_now(),
        "method": "B*-AlwaysOn-L19", "seed": 42, "information_tier": "METADATA_FREE",
        **{k: v for k, v in state.items() if k != "expected_ids"}, **counts,
        "icd_path": OUTPUT_NAME, "icd_sha256": sha256(output),
        "security_metrics_computed": False,
    }
    atomic_json(return_path, result)
    print(json.dumps({"status": "COMPLETE", **counts, "icd_sha256": result["icd_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        raise SystemExit(2)
