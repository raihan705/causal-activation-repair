#!/usr/bin/env python3
"""Fail-closed, resumable Linux scanner for frozen Model2 strength outputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_SEMGREP = "1.175.0"
ALPHAS = (20, 40)
EXPECTED_RECORDS = 180
SCANNER_NAME = "colab_scan_phase9_frozen.py"
SCANNER_SHA256 = "60cedd3d6ec2d19e5e7e5ba4d709230efdd5d9478047492fefda4dd88d9afbb7"
TRANSFER_NAME = "model2_strength_scan_manifest.json"
RETURN_NAME = "model2_strength_scan_return_manifest.json"
EXPECTED_GENERATION_HASHES = {
    20: "c0afad33ec4b21d232d312561184fac4d6e4b82bda7349d53e0ac0ddf5508503",
    40: "ea8f93597768552135909d99862e3de2b8e354364ced3369f0e0c6b266cf5f84",
}


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
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_hash(value: Any) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_scanner(path: Path) -> Any:
    require(sha256_file(path) == SCANNER_SHA256, "Frozen scanner source hash mismatch")
    spec = importlib.util.spec_from_file_location("frozen_model2_scanner", path)
    require(spec is not None and spec.loader is not None, "Cannot load frozen scanner source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("RULE_CWE_MAP", "REGEX_PATTERNS", "TARGET_LANGUAGES", "ALL_TARGET_CWES", "get_ext", "run_regex", "dedup_findings"):
        require(hasattr(module, name), f"Frozen scanner missing {name}")
    return module


def semgrep_version(required: bool) -> str:
    try:
        result = subprocess.run(
            ["semgrep", "--version"], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=30, check=False,
        )
    except Exception as exc:
        if required:
            raise ValidationError(f"Semgrep version check failed: {exc}") from exc
        return "NOT_AVAILABLE_PREFLIGHT_ONLY"
    if result.returncode != 0:
        if required:
            raise ValidationError(f"Semgrep version check failed: {result.stderr.strip()}")
        return "NOT_AVAILABLE_PREFLIGHT_ONLY"
    version = (result.stdout or result.stderr).strip().splitlines()[-1].strip()
    if required:
        require(version == REQUIRED_SEMGREP, f"Expected Semgrep {REQUIRED_SEMGREP}, got {version}")
    return version


def validate_bundle(workdir: Path, require_semgrep: bool) -> tuple[dict[str, Any], dict[int, list[dict[str, Any]]], Any, str]:
    manifest_path = workdir / TRANSFER_NAME
    require(manifest_path.is_file(), f"Missing {TRANSFER_NAME}")
    manifest = load_json(manifest_path)
    require(manifest["schema_version"] == "phase21_model2_strength_scan_manifest_v1", "Scan manifest schema mismatch")
    require(manifest["status"] == "FROZEN_WAITING_MODEL2_STRENGTH_SCAN", "Scan manifest is not frozen")
    require(manifest["required_semgrep_version"] == REQUIRED_SEMGREP, "Scan manifest Semgrep version mismatch")
    require(manifest["scan_order"] == [20, 40], "Scan order changed")
    require(manifest["expected_total_records"] == 360, "Expected scan workload changed")
    for item in manifest["immutable_files"]:
        path = workdir / item["name"]
        require(path.is_file(), f"Missing immutable bundle file: {item['name']}")
        require(sha256_file(path) == item["sha256"], f"Immutable bundle file drift: {item['name']}")
    scanner = load_scanner(workdir / SCANNER_NAME)
    hashes = manifest["scanner_rule_hashes"]
    require(canonical_hash(scanner.RULE_CWE_MAP) == hashes["rule_cwe_map_sha256"], "Scanner rule map changed")
    require(canonical_hash(scanner.REGEX_PATTERNS) == hashes["regex_patterns_sha256"], "Scanner regex rules changed")
    require(canonical_hash(sorted(scanner.TARGET_LANGUAGES)) == hashes["target_languages_sha256"], "Scanner languages changed")
    generations: dict[int, list[dict[str, Any]]] = {}
    for entry in manifest["conditions"]:
        alpha = int(entry["alpha"])
        require(alpha in ALPHAS, f"Unauthorized alpha in scan manifest: {alpha}")
        path = workdir / entry["generation_filename"]
        require(sha256_file(path) == EXPECTED_GENERATION_HASHES[alpha] == entry["generation_sha256"], f"Alpha{alpha} generation hash mismatch")
        records = load_json(path)
        require(isinstance(records, list) and len(records) == EXPECTED_RECORDS, f"Alpha{alpha} generation count mismatch")
        require([record["prompt_id"] for record in records] == manifest["prompt_ids_source_order"], f"Alpha{alpha} prompt order mismatch")
        require(len({record["condition_key"] for record in records}) == EXPECTED_RECORDS, f"Alpha{alpha} duplicate generation keys")
        require(all(int(record["alpha"]) == alpha for record in records), f"Alpha{alpha} record alpha mismatch")
        require(all(int(record["seed"]) == 42 for record in records), f"Alpha{alpha} record seed mismatch")
        generations[alpha] = records
    require(set(generations) == set(ALPHAS), "Exactly alpha20/alpha40 must be present")
    return manifest, generations, scanner, semgrep_version(require_semgrep)


def run_semgrep(code: str, language: str, scanner: Any) -> dict[str, Any]:
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=scanner.get_ext(language), mode="w", delete=False, encoding="utf-8") as handle:
            handle.write(code[:3000])
            temporary_path = handle.name
        result = subprocess.run(
            ["semgrep", "--config", "p/security-audit", "--json", "--quiet", "--timeout", "10", "--max-memory", "1000", temporary_path],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30, check=False,
        )
        if result.returncode != 0:
            return {"success": False, "exit_status": result.returncode, "timeout": False, "error": f"Semgrep nonzero exit: {result.stderr.strip()}", "warnings": [], "findings": []}
        try:
            payload = json.loads(result.stdout)
        except Exception as exc:
            return {"success": False, "exit_status": result.returncode, "timeout": False, "error": f"Semgrep JSON parse failure: {type(exc).__name__}: {exc}", "warnings": [], "findings": []}
        reported = list(payload.get("errors") or [])
        warnings = [item for item in reported if str(item.get("level", "")).lower() == "warn"]
        fatal = [item for item in reported if str(item.get("level", "")).lower() != "warn"]
        if fatal:
            return {"success": False, "exit_status": result.returncode, "timeout": False, "error": f"Semgrep reported fatal errors: {json.dumps(fatal, ensure_ascii=False)}", "warnings": warnings, "findings": []}
        findings = []
        for row in payload.get("results", []):
            rule_id = row.get("check_id", "")
            cwe = scanner.RULE_CWE_MAP.get(rule_id)
            if cwe is None:
                for key, value in scanner.RULE_CWE_MAP.items():
                    if key in rule_id:
                        cwe = value
                        break
            if cwe and cwe in scanner.ALL_TARGET_CWES:
                findings.append({
                    "cwe_id": cwe,
                    "rule_id": rule_id,
                    "severity": row.get("extra", {}).get("severity", ""),
                    "location": row.get("start", {}).get("line", -1),
                    "source": "semgrep",
                })
        return {"success": True, "exit_status": result.returncode, "timeout": False, "error": None, "warnings": warnings, "findings": findings}
    except subprocess.TimeoutExpired as exc:
        return {"success": False, "exit_status": None, "timeout": True, "error": f"Semgrep timeout after {exc.timeout} seconds", "warnings": [], "findings": []}
    except Exception as exc:
        return {"success": False, "exit_status": None, "timeout": False, "error": f"Semgrep exception: {type(exc).__name__}: {exc}", "warnings": [], "findings": []}
    finally:
        if temporary_path:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass


def scan_record(record: dict[str, Any], scanner: Any, generation_hash: str) -> dict[str, Any]:
    code = str(record.get("generated_code", ""))
    language = str(record.get("language", "")).lower()
    eligible = language in scanner.TARGET_LANGUAGES and bool(code.strip())
    if not eligible:
        return {
            "prompt_id": record["prompt_id"], "source_cwe": record["source_cwe"], "cwe_id": record["source_cwe"],
            "language": language, "alpha": record["alpha"], "layer": record["layer"], "feature_id": record["feature_id"],
            "condition_key": record["condition_key"], "generation_sha256": generation_hash,
            "generated_code_sha256": sha256_text(code), "scan_status": "SKIPPED_INELIGIBLE",
            "scanner_success": False, "scanner_failure": False, "scanner_timeout": False,
            "scanner_error": "SCANNER_INELIGIBLE_UNSUPPORTED_LANGUAGE_OR_EMPTY_CODE", "semgrep_exit_status": None,
            "semgrep_warning_count": 0, "semgrep_warnings": [], "findings": [], "vulnerable_cwes": [],
            "is_vulnerable": None, "target_cwe_present": None, "skipped": True,
        }
    semgrep = run_semgrep(code, language, scanner)
    regex_findings = scanner.run_regex(code)
    if semgrep["success"]:
        findings = scanner.dedup_findings(semgrep["findings"] + regex_findings)
        status = "SUCCESS_WITH_WARNINGS" if semgrep["warnings"] else "SUCCESS"
        vulnerable_cwes = sorted({item["cwe_id"] for item in findings})
        is_vulnerable: bool | None = any(cwe in scanner.DERIVATION_CWES for cwe in vulnerable_cwes)
        target_present: bool | None = record["source_cwe"] in vulnerable_cwes
    else:
        findings = []
        status = "FAILED_CLOSED"
        vulnerable_cwes = []
        is_vulnerable = None
        target_present = None
    return {
        "prompt_id": record["prompt_id"], "source_cwe": record["source_cwe"], "cwe_id": record["source_cwe"],
        "language": language, "alpha": record["alpha"], "layer": record["layer"], "feature_id": record["feature_id"],
        "condition_key": record["condition_key"], "generation_sha256": generation_hash,
        "generated_code_sha256": sha256_text(code), "scan_status": status,
        "scanner_success": semgrep["success"], "scanner_failure": not semgrep["success"], "scanner_timeout": semgrep["timeout"],
        "scanner_error": semgrep["error"], "semgrep_exit_status": semgrep["exit_status"],
        "semgrep_warning_count": len(semgrep["warnings"]), "semgrep_warnings": semgrep["warnings"],
        "regex_findings_diagnostic": regex_findings, "findings": findings, "vulnerable_cwes": vulnerable_cwes,
        "is_vulnerable": is_vulnerable, "target_cwe_present": target_present, "skipped": False,
    }


def scan_condition(alpha: int, records: list[dict[str, Any]], scanner: Any, workdir: Path, resume: bool) -> dict[str, Any]:
    output = workdir / f"model2_alpha{alpha}_dev_scan.json"
    checkpoint = workdir / f"model2_alpha{alpha}_dev_scan_checkpoint.json"
    generation_hash = EXPECTED_GENERATION_HASHES[alpha]
    if output.exists():
        existing = load_json(output)
        require(isinstance(existing, list) and len(existing) == 180, f"Alpha{alpha} existing scan count mismatch")
        require([row["condition_key"] for row in existing] == [row["condition_key"] for row in records], f"Alpha{alpha} existing scan order mismatch")
        print(f"ALPHA{alpha} ALREADY_COMPLETE", flush=True)
        return {"alpha": alpha, "status": "REUSED_VALID_COMPLETE", "output": output.name, "sha256": sha256_file(output), "records": 180}
    completed: list[dict[str, Any]] = []
    if checkpoint.exists():
        require(resume, f"Alpha{alpha} checkpoint exists; use --resume")
        payload = load_json(checkpoint)
        completed = payload["records"]
        require([row["condition_key"] for row in completed] == [row["condition_key"] for row in records[:len(completed)]], f"Alpha{alpha} checkpoint prefix mismatch")
    for index in range(len(completed), len(records)):
        completed.append(scan_record(records[index], scanner, generation_hash))
        if len(completed) % 25 == 0 or len(completed) == 180:
            atomic_json(checkpoint, {
                "schema_version": "phase21_model2_strength_scan_checkpoint_v1",
                "status": "IN_PROGRESS" if len(completed) < 180 else "SCAN_COMPLETE_PENDING_FINALIZATION",
                "alpha": alpha, "generation_sha256": generation_hash,
                "completed_records": len(completed), "records": completed,
            })
            failures = sum(row["scanner_failure"] for row in completed)
            print(f"ALPHA{alpha} {len(completed)}/180 scanner_failures={failures}", flush=True)
    require(len(completed) == 180, f"Alpha{alpha} scan count mismatch")
    require([row["condition_key"] for row in completed] == [row["condition_key"] for row in records], f"Alpha{alpha} scan order mismatch")
    atomic_json(output, completed)
    atomic_json(checkpoint, {
        "schema_version": "phase21_model2_strength_scan_checkpoint_v1", "status": "COMPLETE",
        "alpha": alpha, "generation_sha256": generation_hash, "completed_records": 180,
        "scan_output_sha256": sha256_file(output), "records": completed,
    })
    return {"alpha": alpha, "status": "EXECUTED_FRESH", "output": output.name, "sha256": sha256_file(output), "records": 180}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=TRANSFER_NAME)
    parser.add_argument("--workdir", type=Path, default=Path("."))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    workdir = args.workdir.resolve()
    require(Path(args.manifest).name == TRANSFER_NAME, "Only the frozen strength scan manifest is accepted")
    manifest, generations, scanner, version = validate_bundle(workdir, not args.preflight_only)
    print(json.dumps({
        "status": "PASS", "workdir": str(workdir), "semgrep_version": version,
        "scan_order": [20, 40], "records_per_alpha": 180,
        "scanner_sha256": SCANNER_SHA256, "scanner_run": False,
    }, indent=2), flush=True)
    if args.preflight_only:
        return 0
    require(args.all, "Use --all so alpha20 and alpha40 are both scanned")
    entries = []
    for alpha in ALPHAS:
        resume = args.resume or (workdir / f"model2_alpha{alpha}_dev_scan_checkpoint.json").exists()
        entries.append(scan_condition(alpha, generations[alpha], scanner, workdir, resume))
    return_manifest = {
        "schema_version": "phase21_model2_strength_scan_return_manifest_v1",
        "status": "COMPLETE",
        "completed_at_utc": utc_now(),
        "execution_environment": {
            "platform": platform.platform(), "python_version": platform.python_version(),
            "semgrep_version": version, "scanner_source_sha256": SCANNER_SHA256,
            "wrapper_sha256": sha256_file(Path(__file__).resolve()),
        },
        "transfer_manifest_sha256": sha256_file(workdir / TRANSFER_NAME),
        "scan_order": [20, 40], "entries": entries,
        "total_records": sum(entry["records"] for entry in entries),
        "scanner_failures": sum(sum(row["scanner_failure"] for row in load_json(workdir / entry["output"])) for entry in entries),
        "scanner_timeouts": sum(sum(row["scanner_timeout"] for row in load_json(workdir / entry["output"])) for entry in entries),
        "heldout_accessed": False, "scanner_rules_changed": False,
    }
    atomic_json(workdir / RETURN_NAME, return_manifest)
    print(json.dumps({
        "status": "COMPLETE", "total_records": return_manifest["total_records"],
        "scanner_failures": return_manifest["scanner_failures"],
        "return_manifest_sha256": sha256_file(workdir / RETURN_NAME),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"MODEL2_STRENGTH_SCAN_ERROR: {type(exc).__name__}: {exc}", flush=True)
        raise
