#!/usr/bin/env python
"""Resumably scan the frozen Stage 22H paired outputs on Linux/Colab."""

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

EXPECTED_COUNT = 468
CHECKPOINT_NAME = "phase22h_scans_checkpoint.json"
OUTPUT_NAME = "phase22h_scans.json"
RETURN_NAME = "phase22h_scan_return_manifest.json"


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
    spec = importlib.util.spec_from_file_location("phase22h_frozen_scanner", path)
    require(spec is not None and spec.loader is not None, "cannot import frozen scanner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_input(path: Path, transfer: dict[str, Any]) -> list[dict[str, Any]]:
    require(sha256_file(path) == transfer["scan_input_sha256"], "scan input hash mismatch")
    rows = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(rows, list) and len(rows) == EXPECTED_COUNT, "scan input count mismatch")
    require([row["record_id"] for row in rows] == transfer["record_ids"], "scan input record order mismatch")
    require(len({row["record_id"] for row in rows}) == EXPECTED_COUNT, "duplicate scan record IDs")
    require(sum(not row["generation_valid"] for row in rows) == 0, "generation-validity count mismatch")
    require(Counter(row["condition"] for row in rows) == Counter({"B0": 234, "MODEL3_CWE_LABEL_ROUTED": 234}), "condition count mismatch")
    require(Counter(int(row["seed"]) for row in rows) == Counter({42: 156, 43: 156, 44: 156}), "seed count mismatch")
    return rows


def validate_records(records: list[dict[str, Any]], inputs: list[dict[str, Any]]) -> None:
    require(len(records) <= EXPECTED_COUNT, "too many scan rows")
    fields = (
        "record_index", "record_id", "condition", "seed", "prompt_id", "target_cwe",
        "cwe_identifier", "language", "layer", "feature_id", "alpha",
        "generation_valid", "generation_invalid_reason", "generated_text_sha256",
    )
    for index, (row, inp) in enumerate(zip(records, inputs)):
        require(all(row[field] == inp[field] for field in fields), f"scan metadata mismatch at {index}")
        require(row["record_index"] == index, f"scan order mismatch at {index}")
        keys = [(item["cwe_id"], item["rule_id"]) for item in row["findings"]]
        require(len(keys) == len(set(keys)), f"duplicate finding at {index}")
        cwes = {item["cwe_id"] for item in row["findings"]}
        require(cwes == set(row["vulnerable_cwes"]), f"vulnerable_cwes mismatch at {index}")
        require(row["target_cwe_present"] == (row["target_cwe"] in cwes), f"target flag mismatch at {index}")
        require(row["any_finding"] == bool(row["findings"]), f"any-finding mismatch at {index}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("phase22h_scan_transfer_manifest.json"))
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    workdir = args.workdir.resolve()
    manifest_path = args.manifest if args.manifest.is_absolute() else workdir / args.manifest
    transfer = json.loads(manifest_path.read_text(encoding="utf-8"))
    input_path = workdir / transfer["scan_input_filename"]
    scanner_path = workdir / transfer["scanner_filename"]
    wrapper_path = Path(__file__).resolve()
    require(sha256_file(scanner_path) == transfer["scanner_sha256"], "scanner hash mismatch")
    require(sha256_file(wrapper_path) == transfer["wrapper_sha256"], "wrapper hash mismatch")
    inputs = validate_input(input_path, transfer)
    preflight = {
        "schema_version": "phase22h_scan_preflight_v1",
        "status": "PASS",
        "record_count": len(inputs),
        "condition_counts": dict(Counter(row["condition"] for row in inputs)),
        "seed_counts": {str(key): value for key, value in sorted(Counter(int(row["seed"]) for row in inputs).items())},
        "scan_input_sha256": sha256_file(input_path),
        "scanner_sha256": sha256_file(scanner_path),
        "wrapper_sha256": sha256_file(wrapper_path),
        "actual_scan_run": False,
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
    records: list[dict[str, Any]] = []
    if output_path.exists():
        require(args.resume, "final scan exists; use --resume for validation-only reuse")
        container = json.loads(output_path.read_text(encoding="utf-8"))
        require(container["status"] == "COMPLETE", "existing final scan incomplete")
        records = container["records"]
    elif checkpoint_path.exists():
        require(args.resume, "checkpoint exists; use --resume")
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        require(checkpoint["scan_input_sha256"] == transfer["scan_input_sha256"], "checkpoint input drift")
        require(checkpoint["scanner_sha256"] == transfer["scanner_sha256"] and checkpoint["wrapper_sha256"] == transfer["wrapper_sha256"], "checkpoint scanner/wrapper drift")
        records = checkpoint["records"]
        validate_records(records, inputs)
    else:
        require(not args.resume, "--resume supplied without checkpoint/final")

    for index in range(len(records), len(inputs)):
        inp = inputs[index]
        language = inp["language"].lower()
        code = inp["generated_code"]
        skipped = language not in scanner.TARGET_LANGUAGES or not code.strip()
        findings = [] if skipped else scanner.dedup_findings(scanner.run_semgrep(code, language) + scanner.run_regex(code))
        cwes = sorted({item["cwe_id"] for item in findings})
        records.append({
            "schema_version": "phase22h_scan_record_v1",
            "record_index": index,
            "record_id": inp["record_id"],
            "condition": inp["condition"],
            "seed": inp["seed"],
            "prompt_id": inp["prompt_id"],
            "target_cwe": inp["target_cwe"],
            "cwe_identifier": inp["cwe_identifier"],
            "cwe_id": inp["target_cwe"],
            "language": language,
            "layer": inp["layer"],
            "feature_id": inp["feature_id"],
            "alpha": inp["alpha"],
            "generation_valid": inp["generation_valid"],
            "generation_invalid_reason": inp["generation_invalid_reason"],
            "generated_text_sha256": inp["generated_text_sha256"],
            "findings": findings,
            "vulnerable_cwes": cwes,
            "is_vulnerable": any(item["cwe_id"] in scanner.DERIVATION_CWES for item in findings),
            "target_cwe_present": inp["target_cwe"] in cwes,
            "any_finding": bool(findings),
            "skipped": skipped,
        })
        if len(records) % 10 == 0 or len(records) == EXPECTED_COUNT:
            atomic_json(checkpoint_path, {
                "schema_version": "phase22h_scan_checkpoint_v1",
                "status": "IN_PROGRESS",
                "scan_input_sha256": transfer["scan_input_sha256"],
                "scanner_sha256": transfer["scanner_sha256"],
                "wrapper_sha256": transfer["wrapper_sha256"],
                "completed_records": len(records),
                "records": records,
            })
            print(f"PHASE22H SCAN {len(records)}/{EXPECTED_COUNT}", flush=True)

    validate_records(records, inputs)
    container = {
        "schema_version": "phase22h_scans_v1",
        "status": "COMPLETE",
        "scan_input_sha256": transfer["scan_input_sha256"],
        "scanner_sha256": transfer["scanner_sha256"],
        "wrapper_sha256": transfer["wrapper_sha256"],
        "semgrep_version": version,
        "records": records,
    }
    atomic_json(output_path, container)
    returned = {
        "schema_version": "phase22h_scan_return_v1",
        "status": "COMPLETE",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "semgrep_version": version,
        "generation_sha256": transfer["generation_sha256"],
        "generation_validation_sha256": transfer["generation_validation_sha256"],
        "protocol_sha256": transfer["protocol_sha256"],
        "population_sha256": transfer["population_sha256"],
        "scan_input_sha256": transfer["scan_input_sha256"],
        "scanner_sha256": transfer["scanner_sha256"],
        "wrapper_sha256": transfer["wrapper_sha256"],
        "scan_output_filename": OUTPUT_NAME,
        "scan_output_sha256": sha256_file(output_path),
        "record_count": len(records),
        "condition_counts": dict(Counter(row["condition"] for row in records)),
        "seed_counts": {str(key): value for key, value in sorted(Counter(int(row["seed"]) for row in records).items())},
        "eligible_count": sum(not row["skipped"] for row in records),
        "skipped_count": sum(row["skipped"] for row in records),
        "target_positive_count": sum(row["target_cwe_present"] for row in records),
        "any_finding_count": sum(row["any_finding"] for row in records),
        "scanner_rules_changed": False,
        "heldout_split_scanned": True,
    }
    atomic_json(workdir / RETURN_NAME, returned)
    print(json.dumps(returned, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"PHASE22H_SCAN_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
