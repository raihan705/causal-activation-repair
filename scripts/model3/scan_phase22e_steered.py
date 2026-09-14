#!/usr/bin/env python
"""Resumably scan frozen Stage 22E steered outputs on Linux/Colab."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPECTED_COUNT = 1170
CHECKPOINT_NAME = "phase22e_steered_scans_checkpoint.json"
OUTPUT_NAME = "phase22e_steered_scans.json"
RETURN_NAME = "phase22e_steered_scan_return_manifest.json"


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


def load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("phase22e_frozen_scanner", path)
    require(spec is not None and spec.loader is not None, "cannot import frozen scanner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_input(path: Path, transfer: dict[str, Any]) -> list[dict[str, Any]]:
    require(sha256_file(path) == transfer["scan_input_sha256"], "scan input hash mismatch")
    rows = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(rows, list) and len(rows) == EXPECTED_COUNT, "scan input count mismatch")
    require([x["record_id"] for x in rows] == transfer["record_ids"], "scan input record ID/order mismatch")
    require(len({x["record_id"] for x in rows}) == EXPECTED_COUNT, "duplicate scan record IDs")
    require(all(x["generated_code"].strip() for x in rows), "empty scan input code")
    return rows


def validate_records(records: list[dict[str, Any]], inputs: list[dict[str, Any]]) -> None:
    require(len(records) <= EXPECTED_COUNT, "too many scan records")
    for index, (row, inp) in enumerate(zip(records, inputs)):
        require(row["record_index"] == index and row["record_id"] == inp["record_id"], f"scan prefix order mismatch at {index}")
        for field in ("prompt_id", "target_cwe", "layer", "feature_id", "statistical_rank", "screening_alpha", "arm", "language"):
            require(row[field] == inp[field], f"scan metadata mismatch at {index}: {field}")
        require(isinstance(row["findings"], list) and isinstance(row["vulnerable_cwes"], list), f"scan list schema mismatch at {index}")
        keys = [(x["cwe_id"], x["rule_id"]) for x in row["findings"]]
        require(len(keys) == len(set(keys)), f"duplicate finding at {index}")
        finding_cwes = {x["cwe_id"] for x in row["findings"]}
        require(finding_cwes == set(row["vulnerable_cwes"]), f"vulnerable_cwes mismatch at {index}")
        require(row["target_cwe_present"] == (row["target_cwe"] in finding_cwes), f"target flag mismatch at {index}")
        require(row["any_finding"] == bool(row["findings"]), f"any-finding flag mismatch at {index}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("phase22e_steered_scan_transfer_manifest.json"))
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    workdir = args.workdir.resolve()
    manifest_path = args.manifest if args.manifest.is_absolute() else workdir / args.manifest
    transfer = json.loads(manifest_path.read_text(encoding="utf-8"))
    scan_input_path = workdir / transfer["scan_input_filename"]
    scanner_path = workdir / transfer["scanner_filename"]
    wrapper_path = Path(__file__).resolve()
    require(sha256_file(scanner_path) == transfer["scanner_sha256"], "scanner hash mismatch")
    require(sha256_file(wrapper_path) == transfer["wrapper_sha256"], "wrapper hash mismatch")
    inputs = validate_input(scan_input_path, transfer)
    preflight = {
        "schema_version": "phase22e_steered_scan_preflight_v1", "status": "PASS",
        "record_count": len(inputs), "scan_input_sha256": sha256_file(scan_input_path),
        "scanner_sha256": sha256_file(scanner_path), "wrapper_sha256": sha256_file(wrapper_path),
        "actual_scan_run": False, "heldout_used": False,
    }
    if args.preflight_only:
        print(json.dumps(preflight, indent=2))
        return 0
    require(platform.system() == "Linux", "actual scan requires Linux/Colab")
    version_result = subprocess.run(["semgrep", "--version"], capture_output=True, text=True, check=False)
    version = (version_result.stdout or version_result.stderr).strip()
    require(version_result.returncode == 0 and version == transfer["required_semgrep_version"], f"Semgrep version mismatch: {version}")
    scanner = load_module(scanner_path)
    checkpoint_path = workdir / CHECKPOINT_NAME
    output_path = workdir / OUTPUT_NAME
    return_path = workdir / RETURN_NAME
    records: list[dict[str, Any]] = []
    if output_path.exists():
        require(args.resume, "final scan output exists; use --resume for validation-only reuse")
        container = json.loads(output_path.read_text(encoding="utf-8"))
        require(container["status"] == "COMPLETE", "existing final scan is incomplete")
        records = container["records"]
    elif checkpoint_path.exists():
        require(args.resume, "scan checkpoint exists; use --resume")
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        require(checkpoint["scan_input_sha256"] == transfer["scan_input_sha256"], "checkpoint input drift")
        require(checkpoint["scanner_sha256"] == transfer["scanner_sha256"], "checkpoint scanner drift")
        require(checkpoint["wrapper_sha256"] == transfer["wrapper_sha256"], "checkpoint wrapper drift")
        records = checkpoint["records"]
        validate_records(records, inputs)
    else:
        require(not args.resume, "--resume supplied without checkpoint/final output")

    for index in range(len(records), len(inputs)):
        inp = inputs[index]
        language = inp["language"].lower()
        code = inp["generated_code"]
        skipped = language not in scanner.TARGET_LANGUAGES or not code.strip()
        if skipped:
            findings: list[dict[str, Any]] = []
        else:
            findings = scanner.dedup_findings(scanner.run_semgrep(code, language) + scanner.run_regex(code))
        finding_cwes = sorted({x["cwe_id"] for x in findings})
        records.append({
            "schema_version": "phase22e_steered_scan_record_v1", "record_index": index,
            "record_id": inp["record_id"], "prompt_id": inp["prompt_id"],
            "target_cwe": inp["target_cwe"], "cwe_id": inp["target_cwe"],
            "layer": inp["layer"], "feature_id": inp["feature_id"],
            "statistical_rank": inp["statistical_rank"], "screening_alpha": inp["screening_alpha"],
            "arm": inp["arm"], "language": language, "findings": findings,
            "vulnerable_cwes": finding_cwes,
            "is_vulnerable": any(x["cwe_id"] in scanner.DERIVATION_CWES for x in findings),
            "target_cwe_present": inp["target_cwe"] in finding_cwes,
            "any_finding": bool(findings), "skipped": skipped,
        })
        if len(records) % 10 == 0 or len(records) == EXPECTED_COUNT:
            atomic_json(checkpoint_path, {
                "schema_version": "phase22e_steered_scan_checkpoint_v1", "status": "IN_PROGRESS",
                "scan_input_sha256": transfer["scan_input_sha256"], "scanner_sha256": transfer["scanner_sha256"],
                "wrapper_sha256": transfer["wrapper_sha256"], "completed_records": len(records), "records": records,
            })
            print(f"PHASE22E SCAN {len(records)}/{EXPECTED_COUNT}", flush=True)
    validate_records(records, inputs)
    container = {
        "schema_version": "phase22e_steered_scans_v1", "status": "COMPLETE",
        "scan_input_sha256": transfer["scan_input_sha256"], "scanner_sha256": transfer["scanner_sha256"],
        "wrapper_sha256": transfer["wrapper_sha256"], "semgrep_version": version, "records": records,
    }
    atomic_json(output_path, container)
    target_counts = Counter(x["target_cwe"] for x in records)
    returned = {
        "schema_version": "phase22e_steered_scan_return_v1", "status": "COMPLETE",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
        "python_version": platform.python_version(), "semgrep_version": version,
        "steered_generation_sha256": transfer["generation_sha256"],
        "generation_validation_sha256": transfer["generation_validation_sha256"],
        "scan_input_sha256": transfer["scan_input_sha256"], "scanner_sha256": transfer["scanner_sha256"],
        "wrapper_sha256": transfer["wrapper_sha256"], "scan_output_filename": OUTPUT_NAME,
        "scan_output_sha256": sha256_file(output_path), "record_count": len(records),
        "target_counts": dict(target_counts), "eligible_count": sum(not x["skipped"] for x in records),
        "skipped_count": sum(x["skipped"] for x in records),
        "target_positive_count": sum(x["target_cwe_present"] for x in records),
        "any_finding_count": sum(x["any_finding"] for x in records),
        "scanner_rules_changed": False, "generation_run": False, "heldout_used": False,
    }
    atomic_json(return_path, returned)
    print(json.dumps(returned, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"PHASE22E_STEERED_SCAN_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
