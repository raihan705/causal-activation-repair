#!/usr/bin/env python3
"""Resumable, warning-aware, fail-closed scan of Model2 causal outputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_SEMGREP = "1.175.0"
GENERATION_NAME = "model2_causal_steered_generations.json"
GENERATION_VALIDATION_NAME = "model2_causal_steered_generation_validation.json"
GENERATION_MANIFEST_NAME = "model2_causal_steered_generation_manifest.json"
TRANSFER_NAME = "model2_causal_scan_transfer_manifest.json"
FROZEN_SCANNER_NAME = "colab_scan_phase9_frozen.py"
CHECKPOINT_NAME = "model2_causal_steered_scans_checkpoint.json"
OUTPUT_NAME = "model2_causal_steered_scans.json"
RETURN_NAME = "model2_causal_scan_return_manifest.json"
EXPECTED_RECORDS = 1620
EXPECTED_CONDITIONS = 90
EXPECTED_GENERATION_SHA256 = "2c5f4937b63c1e710dd2c949fee8a8179172eb9426e68880948f609b53365c6a"
EXPECTED_FROZEN_SCANNER_SHA256 = "60cedd3d6ec2d19e5e7e5ba4d709230efdd5d9478047492fefda4dd88d9afbb7"


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("phase21_frozen_scanner", path)
    require(spec is not None and spec.loader is not None, "cannot load frozen scanner source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for attr in ("RULE_CWE_MAP", "REGEX_PATTERNS", "TARGET_LANGUAGES", "ALL_TARGET_CWES", "get_ext", "run_regex", "dedup_findings"):
        require(hasattr(module, attr), f"frozen scanner lacks {attr}")
    return module


def semgrep_version(required: bool) -> str:
    try:
        result = subprocess.run(
            ["semgrep", "--version"], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=30, check=False,
        )
    except Exception as exc:
        if required:
            raise ValidationError(f"semgrep version check failed: {type(exc).__name__}: {exc}") from exc
        return "NOT_AVAILABLE_PREFLIGHT_ONLY"
    if result.returncode != 0:
        if required:
            raise ValidationError(f"semgrep version check failed: {result.stderr.strip()}")
        return "NOT_AVAILABLE_PREFLIGHT_ONLY"
    version = (result.stdout or result.stderr).strip().splitlines()[-1].strip()
    if required:
        require(version == REQUIRED_SEMGREP, f"Semgrep version mismatch: expected {REQUIRED_SEMGREP}, got {version}")
    return version


def validate_bundle(workdir: Path, require_semgrep: bool) -> tuple[dict[str, Any], dict[str, Any], Any, str]:
    transfer_path = workdir / TRANSFER_NAME
    require(transfer_path.is_file(), f"missing {TRANSFER_NAME}")
    transfer = read_json(transfer_path)
    require(transfer.get("schema_version") == "phase21_model2_causal_scan_transfer_manifest_v1", "transfer schema mismatch")
    require(transfer.get("status") == "FROZEN_WAITING_CAUSAL_SCAN", "transfer is not frozen")
    require(transfer.get("required_semgrep_version") == REQUIRED_SEMGREP, "transfer Semgrep version mismatch")
    require(transfer.get("expected_record_count") == EXPECTED_RECORDS, "transfer record count mismatch")
    require(transfer.get("expected_condition_count") == EXPECTED_CONDITIONS, "transfer condition count mismatch")
    for item in transfer.get("immutable_files", []):
        path = workdir / item["name"]
        require(path.is_file(), f"missing immutable file: {item['name']}")
        require(sha256_file(path) == item["sha256"], f"immutable file drift: {item['name']}")

    generation_path = workdir / GENERATION_NAME
    validation_path = workdir / GENERATION_VALIDATION_NAME
    scanner_path = workdir / FROZEN_SCANNER_NAME
    require(sha256_file(generation_path) == EXPECTED_GENERATION_SHA256, "generation hash mismatch")
    require(sha256_file(scanner_path) == EXPECTED_FROZEN_SCANNER_SHA256, "frozen scanner hash mismatch")
    validation = read_json(validation_path)
    require(validation.get("status") == "PASS", "generation validation is not PASS")
    require(validation.get("generation_sha256") == EXPECTED_GENERATION_SHA256, "validation/generation binding mismatch")
    generation = read_json(generation_path)
    records = generation.get("records", [])
    require(len(records) == EXPECTED_RECORDS, "generation cardinality mismatch")
    keys = [row.get("condition_key") for row in records]
    require(len(set(keys)) == EXPECTED_RECORDS and None not in keys, "generation logical keys are not unique/complete")
    conditions = {(row.get("target_cwe"), row.get("layer"), row.get("feature_id")) for row in records}
    require(len(conditions) == EXPECTED_CONDITIONS, "generation condition count mismatch")

    scanner = load_module(scanner_path)
    require(sha256_json(scanner.RULE_CWE_MAP) == transfer["scanner_rule_hashes"]["rule_cwe_map_sha256"], "rule/CWE map drift")
    require(sha256_json(scanner.REGEX_PATTERNS) == transfer["scanner_rule_hashes"]["regex_patterns_sha256"], "regex rules drift")
    require(sha256_json(sorted(scanner.TARGET_LANGUAGES)) == transfer["scanner_rule_hashes"]["target_languages_sha256"], "language routing drift")
    version = semgrep_version(require_semgrep)
    return transfer, generation, scanner, version


def run_semgrep(code: str, language: str, frozen: Any) -> dict[str, Any]:
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=frozen.get_ext(language), mode="w", delete=False, encoding="utf-8"
        ) as handle:
            handle.write(code[:3000])
            temporary_path = handle.name
        result = subprocess.run(
            ["semgrep", "--config", "p/security-audit", "--json", "--quiet",
             "--timeout", "10", "--max-memory", "1000", temporary_path],
            capture_output=True, encoding="utf-8", errors="replace", timeout=30, check=False,
        )
        if result.returncode != 0:
            return {
                "success": False, "exit_status": result.returncode, "timeout": False,
                "error": f"Semgrep nonzero exit: {result.stderr.strip()}",
                "warnings": [], "findings": [],
            }
        try:
            data = json.loads(result.stdout)
        except Exception as exc:
            return {
                "success": False, "exit_status": result.returncode, "timeout": False,
                "error": f"Semgrep JSON parse failure: {type(exc).__name__}: {exc}",
                "warnings": [], "findings": [],
            }
        reported = list(data.get("errors") or [])
        warnings = [item for item in reported if str(item.get("level", "")).lower() == "warn"]
        fatal = [item for item in reported if str(item.get("level", "")).lower() != "warn"]
        if fatal:
            return {
                "success": False, "exit_status": result.returncode, "timeout": False,
                "error": f"Semgrep reported fatal errors: {json.dumps(fatal, ensure_ascii=False)}",
                "warnings": warnings, "findings": [],
            }
        findings = []
        for row in data.get("results", []):
            rule_id = row.get("check_id", "")
            cwe = frozen.RULE_CWE_MAP.get(rule_id)
            if cwe is None:
                for key, value in frozen.RULE_CWE_MAP.items():
                    if key in rule_id:
                        cwe = value
                        break
            if cwe and cwe in frozen.ALL_TARGET_CWES:
                findings.append({
                    "cwe_id": cwe,
                    "rule_id": rule_id,
                    "severity": row.get("extra", {}).get("severity", ""),
                    "location": row.get("start", {}).get("line", -1),
                    "source": "semgrep",
                })
        return {
            "success": True, "exit_status": result.returncode, "timeout": False,
            "error": None, "warnings": warnings, "findings": findings,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "success": False, "exit_status": None, "timeout": True,
            "error": f"Semgrep timeout after {exc.timeout} seconds", "warnings": [], "findings": [],
        }
    except Exception as exc:
        return {
            "success": False, "exit_status": None, "timeout": False,
            "error": f"Semgrep exception: {type(exc).__name__}: {exc}", "warnings": [], "findings": [],
        }
    finally:
        if temporary_path:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass


def scan_record(record: dict[str, Any], frozen: Any, generation_hash: str) -> dict[str, Any]:
    code = str(record.get("generated_code", ""))
    language = str(record.get("language", "")).lower()
    eligible = language in frozen.TARGET_LANGUAGES and bool(code.strip())
    if eligible:
        semgrep = run_semgrep(code, language, frozen)
        regex_findings = frozen.run_regex(code)
        all_findings = frozen.dedup_findings(semgrep["findings"] + regex_findings)
        status = (
            "SUCCESS_WITH_WARNINGS" if semgrep["success"] and semgrep["warnings"]
            else "SUCCESS" if semgrep["success"]
            else "FAILED_CLOSED"
        )
    else:
        semgrep = {
            "success": False, "exit_status": None, "timeout": False,
            "error": "SCANNER_INELIGIBLE_UNSUPPORTED_LANGUAGE_OR_EMPTY_CODE",
            "warnings": [], "findings": [],
        }
        regex_findings = []
        all_findings = []
        status = "SKIPPED_INELIGIBLE"
    mapped = sorted({item["cwe_id"] for item in all_findings})
    scanner_success = bool(semgrep["success"])
    target = record["target_cwe"]
    return {
        "schema_version": "phase21_model2_causal_steered_scan_record_v1",
        "record_index": record["record_index"],
        "condition_key": record["condition_key"],
        "target_cwe": target,
        "layer": record["layer"],
        "feature_id": record["feature_id"],
        "original_rank": record["original_rank"],
        "candidate_score": record["candidate_score"],
        "prompt_id": record["prompt_id"],
        "role": record["population"],
        "language": language,
        "generation_valid": record.get("validity", {}).get("is_valid") is True,
        "scanner_eligible": eligible,
        "scanner_success": scanner_success,
        "scanner_status": status,
        "scanner_exit_status": semgrep["exit_status"],
        "scanner_timeout": semgrep["timeout"],
        "scanner_error": semgrep["error"],
        "scanner_warnings": semgrep["warnings"],
        "scanner_warning_count": len(semgrep["warnings"]),
        "parse_complete": len(semgrep["warnings"]) == 0 if scanner_success else None,
        "semgrep_findings": semgrep["findings"],
        "regex_findings": regex_findings,
        "findings": all_findings,
        "mapped_cwe_labels": mapped,
        "target_cwe_present": target in mapped if scanner_success else None,
        "any_vulnerability": bool(all_findings) if scanner_success else None,
        "skipped": not eligible,
        "input_generation_sha256": generation_hash,
        "generated_text_sha256": record["generated_text_sha256"],
        "alpha": record["alpha"],
        "seed": record["seed"],
        "code_transformation": "NONE",
    }


def new_checkpoint(transfer: dict[str, Any], version: str) -> dict[str, Any]:
    return {
        "schema_version": "phase21_model2_causal_scan_checkpoint_v1",
        "status": "IN_PROGRESS",
        "generation_sha256": EXPECTED_GENERATION_SHA256,
        "transfer_manifest_sha256": transfer["self_sha256_excluding_field"],
        "scanner_runner_sha256": sha256_file(Path(__file__).resolve()),
        "frozen_scanner_source_sha256": EXPECTED_FROZEN_SCANNER_SHA256,
        "semgrep_version": version,
        "expected_record_count": EXPECTED_RECORDS,
        "completed_record_count": 0,
        "scanner_failures_fail_closed": True,
        "started_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "records": [],
    }


def validate_checkpoint(checkpoint: dict[str, Any], transfer: dict[str, Any], generation: dict[str, Any], version: str) -> None:
    require(checkpoint.get("schema_version") == "phase21_model2_causal_scan_checkpoint_v1", "checkpoint schema mismatch")
    require(checkpoint.get("status") == "IN_PROGRESS", "existing checkpoint is not IN_PROGRESS")
    require(checkpoint.get("generation_sha256") == EXPECTED_GENERATION_SHA256, "checkpoint generation hash mismatch")
    require(checkpoint.get("transfer_manifest_sha256") == transfer["self_sha256_excluding_field"], "checkpoint transfer binding mismatch")
    require(checkpoint.get("scanner_runner_sha256") == sha256_file(Path(__file__).resolve()), "checkpoint runner drift")
    require(checkpoint.get("frozen_scanner_source_sha256") == EXPECTED_FROZEN_SCANNER_SHA256, "checkpoint scanner drift")
    require(checkpoint.get("semgrep_version") == version == REQUIRED_SEMGREP, "checkpoint Semgrep version mismatch")
    rows = checkpoint.get("records", [])
    require(checkpoint.get("completed_record_count") == len(rows), "checkpoint completed count mismatch")
    require(len(rows) <= EXPECTED_RECORDS, "checkpoint has excess records")
    expected = generation["records"][:len(rows)]
    require([row.get("condition_key") for row in rows] == [row.get("condition_key") for row in expected], "checkpoint is not the expected logical-key prefix")
    require(all(row.get("input_generation_sha256") == EXPECTED_GENERATION_SHA256 for row in rows), "checkpoint input binding mismatch")


def summarize(records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "scanner_success_count": sum(row["scanner_success"] for row in records),
        "scanner_failure_count": sum(row["scanner_eligible"] and not row["scanner_success"] for row in records),
        "scanner_timeout_count": sum(row["scanner_timeout"] for row in records),
        "scanner_skipped_count": sum(row["skipped"] for row in records),
        "warning_record_count": sum(row["scanner_warning_count"] > 0 for row in records),
        "warning_count": sum(row["scanner_warning_count"] for row in records),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, default=Path("."))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    workdir = args.workdir.resolve()
    transfer, generation, frozen, version = validate_bundle(workdir, require_semgrep=not args.preflight_only)
    preflight = {
        "schema_version": "phase21_model2_causal_scan_preflight_v1",
        "status": "PASS",
        "generation_sha256": EXPECTED_GENERATION_SHA256,
        "record_count": len(generation["records"]),
        "condition_count": len({(row["target_cwe"], row["layer"], row["feature_id"]) for row in generation["records"]}),
        "semgrep_version": version,
        "frozen_scanner_source_sha256": EXPECTED_FROZEN_SCANNER_SHA256,
        "scanner_runner_sha256": sha256_file(Path(__file__).resolve()),
        "platform": platform.platform(),
        "scanner_run": False,
    }
    print(json.dumps(preflight, indent=2), flush=True)
    if args.preflight_only:
        return 0
    drive_root = Path("/content/drive").resolve()
    require(workdir == drive_root or drive_root in workdir.parents, "actual scans require a persistent Colab Drive workdir")

    checkpoint_path = workdir / CHECKPOINT_NAME
    output_path = workdir / OUTPUT_NAME
    return_path = workdir / RETURN_NAME
    require(not output_path.exists(), f"{OUTPUT_NAME} already exists; return it for validation rather than rescanning")
    require(not return_path.exists(), f"{RETURN_NAME} already exists; return it for validation rather than rescanning")
    if checkpoint_path.exists():
        require(args.resume, f"checkpoint exists; rerun with --resume: {checkpoint_path}")
        checkpoint = read_json(checkpoint_path)
        validate_checkpoint(checkpoint, transfer, generation, version)
    else:
        checkpoint = new_checkpoint(transfer, version)
        atomic_json(checkpoint_path, checkpoint)

    records = checkpoint["records"]
    start = len(records)
    for index in range(start, EXPECTED_RECORDS):
        # Revalidate immutable inputs before each durable 10-record block.
        if index % 10 == 0:
            validate_bundle(workdir, require_semgrep=True)
        scanned = scan_record(generation["records"][index], frozen, EXPECTED_GENERATION_SHA256)
        records.append(scanned)
        if len(records) % 10 == 0 or len(records) == EXPECTED_RECORDS:
            checkpoint["completed_record_count"] = len(records)
            checkpoint["updated_at_utc"] = utc_now()
            checkpoint.update(summarize(records))
            atomic_json(checkpoint_path, checkpoint)
            print(
                f"SCAN {len(records)}/{EXPECTED_RECORDS} failures={checkpoint['scanner_failure_count']} "
                f"timeouts={checkpoint['scanner_timeout_count']} warnings={checkpoint['warning_record_count']}",
                flush=True,
            )

    require(len(records) == EXPECTED_RECORDS, "scan record count incomplete")
    require(len({row["condition_key"] for row in records}) == EXPECTED_RECORDS, "scan logical keys not unique")
    summary = summarize(records)
    output = {
        "schema_version": "phase21_model2_causal_steered_scans_v1",
        "status": "COMPLETE" if summary["scanner_failure_count"] == 0 and summary["scanner_skipped_count"] == 0 else "COMPLETE_WITH_EXPLICIT_SCANNER_FAILURES",
        "generation_input_sha256": EXPECTED_GENERATION_SHA256,
        "generation_manifest_sha256": transfer["generation_manifest"]["sha256"],
        "generation_validation_sha256": transfer["generation_validation"]["sha256"],
        "transfer_manifest_sha256": transfer["self_sha256_excluding_field"],
        "semgrep_version": REQUIRED_SEMGREP,
        "registry_config": "p/security-audit",
        "registry_snapshot_hash": "NOT_CONTENT_PINNED",
        "frozen_scanner_source_sha256": EXPECTED_FROZEN_SCANNER_SHA256,
        "scanner_runner_sha256": sha256_file(Path(__file__).resolve()),
        "scanner_rule_hashes": transfer["scanner_rule_hashes"],
        "fail_closed": True,
        "exit_zero_warnings_are_process_failures": False,
        "code_transformation": "NONE",
        "record_count": len(records),
        "condition_count": EXPECTED_CONDITIONS,
        **summary,
        "records": records,
    }
    atomic_json(output_path, output)
    output_hash = sha256_file(output_path)
    returned = {
        "schema_version": "phase21_model2_causal_scan_return_manifest_v1",
        "status": output["status"],
        "completed_at_utc_diagnostic": utc_now(),
        "generation_input": {"name": GENERATION_NAME, "sha256": EXPECTED_GENERATION_SHA256},
        "scan_output": {"name": OUTPUT_NAME, "sha256": output_hash},
        "transfer_manifest_sha256": transfer["self_sha256_excluding_field"],
        "scanner_runner_sha256": sha256_file(Path(__file__).resolve()),
        "frozen_scanner_source_sha256": EXPECTED_FROZEN_SCANNER_SHA256,
        "scanner_rule_hashes": transfer["scanner_rule_hashes"],
        "semgrep_version": REQUIRED_SEMGREP,
        "registry_config": "p/security-audit",
        "registry_snapshot_hash": "NOT_CONTENT_PINNED",
        "fail_closed": True,
        "code_transformation": "NONE",
        "record_count": len(records),
        "condition_count": EXPECTED_CONDITIONS,
        **summary,
    }
    atomic_json(return_path, returned)
    checkpoint["status"] = "COMPLETE"
    checkpoint["scan_output_sha256"] = output_hash
    checkpoint["return_manifest_sha256"] = sha256_file(return_path)
    checkpoint["completed_at_utc"] = utc_now()
    atomic_json(checkpoint_path, checkpoint)
    print(json.dumps(returned, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        print(f"PHASE21_CAUSAL_SCAN_STOP: {exc}", file=sys.stderr)
        raise SystemExit(2)
