#!/usr/bin/env python
"""Validate and scan the immutable Model2 B0 file with the unchanged ICD."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPECTED_GENERATION_SHA256 = "d90f58718a40aa702e25018265bb0ab843afd74b3cbfada81cdd139c2c9a6760"
EXPECTED_SCANNER_SHA256 = "60cedd3d6ec2d19e5e7e5ba4d709230efdd5d9478047492fefda4dd88d9afbb7"
EXPECTED_COUNT = 1341
OUTPUT_NAME = "model2_b0_dev_scan.json"
RETURN_NAME = "model2_b0_dev_scan_return_manifest.json"


class ValidationError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise ValidationError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_generation(path: Path, transfer: dict[str, Any]) -> list[int]:
    require(sha256_file(path) == EXPECTED_GENERATION_SHA256 == transfer["generation_sha256"], "generation hash mismatch")
    records = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(records, list) and len(records) == EXPECTED_COUNT, "generation count mismatch")
    ids = [int(x["prompt_id"]) for x in records]
    require(len(set(ids)) == EXPECTED_COUNT, "duplicate generation prompt ID")
    require(ids == transfer["prompt_ids_source_order"], "generation source order mismatch")
    require(all(x.get("generation_status") == "SUCCESS" for x in records), "generation contains non-success status")
    require(all(x.get("generated_code", "").strip() for x in records), "generation contains empty scanner input")
    require(all(x.get("method") == "B0" and x.get("seed") == 42 for x in records), "generation method/seed mismatch")
    return ids


def validate_scan(path: Path, ids: list[int]) -> dict[str, Any]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(rows, list) and len(rows) == EXPECTED_COUNT, "scan count mismatch")
    scan_ids = [int(x["prompt_id"]) for x in rows]
    require(scan_ids == ids and len(set(scan_ids)) == EXPECTED_COUNT, "scan IDs/order mismatch")
    required = {"prompt_id", "cwe_id", "language", "findings", "vulnerable_cwes", "is_vulnerable", "skipped"}
    require(all(isinstance(x, dict) and required.issubset(x) for x in rows), "scan schema mismatch")
    return {"record_count": len(rows), "eligible_count": sum(not x["skipped"] for x in rows), "skipped_count": sum(x["skipped"] for x in rows), "vulnerable_count": sum(x["is_vulnerable"] for x in rows), "sha256": sha256_file(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("model2_b0_dev_scan_transfer_manifest.json"))
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    workdir = args.workdir.resolve()
    transfer_path = args.manifest if args.manifest.is_absolute() else workdir / args.manifest
    transfer = json.loads(transfer_path.read_text(encoding="utf-8"))
    generation = workdir / transfer["generation_filename"]
    scanner = workdir / transfer["scanner_filename"]
    wrapper = Path(__file__).resolve()
    require(sha256_file(scanner) == EXPECTED_SCANNER_SHA256 == transfer["scanner_sha256"], "frozen scanner hash mismatch")
    require(sha256_file(wrapper) == transfer["wrapper_sha256"], "wrapper hash mismatch")
    ids = validate_generation(generation, transfer)
    preflight = {"status": "PASS", "generation_sha256": sha256_file(generation), "scanner_sha256": sha256_file(scanner), "wrapper_sha256": sha256_file(wrapper), "record_count": len(ids), "heldout_used": False}
    if args.preflight_only:
        print(json.dumps(preflight, indent=2))
        return 0
    require(platform.system() == "Linux", "actual Semgrep scan requires Linux/Colab")
    result = subprocess.run(["semgrep", "--version"], capture_output=True, text=True, check=False)
    require(result.returncode == 0 and (result.stdout or result.stderr).strip(), "Semgrep version unavailable")
    output = workdir / OUTPUT_NAME
    before_hash = sha256_file(generation)
    if output.exists():
        require(args.resume, "scan output exists; use --resume for validation-only reuse")
    else:
        spec = importlib.util.spec_from_file_location("frozen_model1_icd", scanner)
        require(spec is not None and spec.loader is not None, "cannot import frozen scanner")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        require(callable(getattr(module, "scan_file", None)), "frozen scanner lacks scan_file")
        module.scan_file(generation, output)
    scan = validate_scan(output, ids)
    after_hash = sha256_file(generation)
    require(before_hash == after_hash == EXPECTED_GENERATION_SHA256, "generation changed during scanning")
    ret = {
        "schema_version": "phase21_model2_b0_scan_return_v1", "status": "COMPLETE",
        "completed_at_utc": now(), "platform": platform.platform(),
        "python_version": platform.python_version(), "semgrep_version": (result.stdout or result.stderr).strip(),
        "generation_sha256": after_hash, "generation_unchanged": True,
        "scanner_sha256": sha256_file(scanner), "wrapper_sha256": sha256_file(wrapper),
        "scan_output_filename": OUTPUT_NAME, "scan_output_sha256": scan["sha256"],
        "record_count": scan["record_count"], "eligible_count": scan["eligible_count"],
        "skipped_count": scan["skipped_count"], "vulnerable_count": scan["vulnerable_count"],
        "heldout_used": False, "scanner_rules_changed": False,
    }
    atomic_json(workdir / RETURN_NAME, ret)
    print(json.dumps(ret, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
