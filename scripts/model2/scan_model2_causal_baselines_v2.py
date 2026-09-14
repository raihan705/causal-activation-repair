#!/usr/bin/env python3
"""Versioned warning-aware, fail-closed scan of Model2 causal baselines."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REQUIRED_SEMGREP = "1.175.0"
DEFAULT_MANIFEST = "model2_causal_baseline_scan_transfer_manifest_v2.json"
OUTPUT_NAME = "model2_causal_baseline_scans_v2.json"
RETURN_NAME = "model2_causal_baseline_scan_return_manifest_v2.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def semgrep_version() -> str:
    result = subprocess.run(["semgrep", "--version"], capture_output=True, text=True, timeout=30)
    require(result.returncode == 0, f"semgrep --version failed: {result.stderr.strip()}")
    return (result.stdout or result.stderr).strip().splitlines()[-1].strip()


def load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("frozen_model2_scanner", path)
    require(spec is not None and spec.loader is not None, "cannot load frozen scanner source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_bundle(workdir: Path, manifest_name: str) -> tuple[dict[str, Any], Any]:
    path = workdir / manifest_name
    require(path.is_file(), f"missing transfer manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    require(manifest.get("schema_version") == "phase21_model2_causal_baseline_scan_transfer_manifest_v2", "transfer schema mismatch")
    require(manifest.get("status") == "FROZEN_FOR_WARNING_AWARE_RESCAN", "transfer manifest is not frozen")
    for item in manifest["immutable_files"]:
        item_path = workdir / item["name"]
        require(item_path.is_file(), f"missing immutable file: {item['name']}")
        require(sha256_file(item_path) == item["sha256"], f"immutable file drift: {item['name']}")
    version = semgrep_version()
    require(version == REQUIRED_SEMGREP, f"Semgrep version mismatch: expected {REQUIRED_SEMGREP}, got {version}")
    frozen = load_module(workdir / manifest["frozen_scanner_source"]["name"])
    return manifest, frozen


def run_semgrep(code: str, language: str, frozen: Any) -> dict[str, Any]:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=frozen.get_ext(language), mode="w", delete=False, encoding="utf-8") as tmp:
            tmp.write(code[:3000])
            tmp_path = tmp.name
        result = subprocess.run(
            ["semgrep", "--config", "p/security-audit", "--json", "--quiet", "--timeout", "10", "--max-memory", "1000", tmp_path],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        if result.returncode != 0:
            return {"success": False, "exit_code": result.returncode, "timeout": False, "error": f"Semgrep nonzero exit: {result.stderr.strip()}", "warnings": [], "findings": []}
        try:
            data = json.loads(result.stdout)
        except Exception as exc:
            return {"success": False, "exit_code": result.returncode, "timeout": False, "error": f"Semgrep JSON parse failure: {type(exc).__name__}: {exc}", "warnings": [], "findings": []}
        reported = list(data.get("errors") or [])
        warnings = [x for x in reported if str(x.get("level", "")).lower() == "warn"]
        fatal = [x for x in reported if str(x.get("level", "")).lower() != "warn"]
        if fatal:
            return {"success": False, "exit_code": result.returncode, "timeout": False, "error": f"Semgrep reported fatal errors: {json.dumps(fatal, ensure_ascii=False)}", "warnings": warnings, "findings": []}
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
        return {"success": True, "exit_code": result.returncode, "timeout": False, "error": None, "warnings": warnings, "findings": findings}
    except subprocess.TimeoutExpired as exc:
        return {"success": False, "exit_code": None, "timeout": True, "error": f"Semgrep timeout after {exc.timeout} seconds", "warnings": [], "findings": []}
    except Exception as exc:
        return {"success": False, "exit_code": None, "timeout": False, "error": f"Semgrep exception: {type(exc).__name__}: {exc}", "warnings": [], "findings": []}
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def preflight(workdir: Path, manifest_name: str) -> dict[str, Any]:
    manifest, _ = validate_bundle(workdir, manifest_name)
    generation_path = workdir / manifest["generation_input"]["name"]
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    require(len(generation.get("records", [])) == 45, "generation cardinality mismatch")
    return {
        "schema_version": "phase21_model2_causal_baseline_scan_preflight_v2",
        "status": "PASS",
        "semgrep_version": REQUIRED_SEMGREP,
        "generation_sha256": sha256_file(generation_path),
        "physical_record_count": 45,
        "warning_policy": "exit-0 level:warn entries are preserved but are not process failures",
        "fatal_policy": "timeout, exception, nonzero exit, invalid JSON, or non-warning Semgrep error fails closed",
        "code_transformation": "NONE",
        "scanner_run": False,
    }


def scan(workdir: Path, manifest_name: str) -> None:
    manifest, frozen = validate_bundle(workdir, manifest_name)
    generation_path = workdir / manifest["generation_input"]["name"]
    generation_hash = sha256_file(generation_path)
    require(generation_hash == manifest["generation_input"]["sha256"], "generation hash mismatch")
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    records = generation["records"]
    require(len(records) == 45 and sum(len(x["logical_mappings"]) for x in records) == 45, "generation accounting mismatch")
    results = []
    for index, rec in enumerate(records):
        code = rec.get("generated_code", "")
        language = str(rec.get("language", "")).lower()
        eligible = language in frozen.TARGET_LANGUAGES and bool(code.strip())
        if not eligible:
            semgrep = {"success": False, "exit_code": None, "timeout": False, "error": "SCANNER_INELIGIBLE_UNSUPPORTED_LANGUAGE_OR_EMPTY_CODE", "warnings": [], "findings": []}
            regex_findings = []
            all_findings = []
            status = "SKIPPED_INELIGIBLE"
        else:
            semgrep = run_semgrep(code, language, frozen)
            regex_findings = frozen.run_regex(code)
            all_findings = frozen.dedup_findings(semgrep["findings"] + regex_findings) if semgrep["success"] else regex_findings
            status = "SUCCESS_WITH_WARNINGS" if semgrep["success"] and semgrep["warnings"] else ("SUCCESS" if semgrep["success"] else "FAILED_CLOSED")
        mapped = sorted({x["cwe_id"] for x in all_findings})
        flags = []
        for mapping in rec["logical_mappings"]:
            target = mapping.get("target_cwe")
            flags.append({
                "logical_index": mapping["logical_index"],
                "population": mapping["population"],
                "target_cwe": target,
                "target_cwe_present": bool(target and target in mapped) if semgrep["success"] else None,
            })
        results.append({
            "schema_version": "phase21_model2_causal_baseline_scan_record_v2",
            "physical_index": rec["physical_index"],
            "prompt_id": rec["prompt_id"],
            "source_index": rec["source_index"],
            "logical_mappings": rec["logical_mappings"],
            "language": language,
            "scanner_eligible": eligible,
            "scanner_success": bool(semgrep["success"]),
            "scanner_status": status,
            "scanner_exit_status": semgrep["exit_code"],
            "scanner_timeout": semgrep["timeout"],
            "scanner_error": semgrep["error"],
            "scanner_warnings": semgrep["warnings"],
            "scanner_warning_count": len(semgrep["warnings"]),
            "parse_complete": len(semgrep["warnings"]) == 0 if semgrep["success"] else None,
            "findings": all_findings,
            "mapped_cwe_labels": mapped,
            "target_cwe_flags": flags,
            "any_vulnerability": bool(all_findings) if semgrep["success"] else None,
            "historical_derivation_is_vulnerable": any(x["cwe_id"] in frozen.DERIVATION_CWES for x in all_findings) if semgrep["success"] else None,
            "skipped": not eligible,
            "input_generation_sha256": generation_hash,
            "generated_text_sha256": rec["generated_text_sha256"],
            "code_transformation": "NONE",
        })
        print(f"RESCAN {index + 1}/45 prompt_id={rec['prompt_id']} status={status} warnings={len(semgrep['warnings'])} findings={len(all_findings)}", flush=True)

    failure_count = sum(x["scanner_eligible"] and not x["scanner_success"] for x in results)
    skipped_count = sum(x["skipped"] for x in results)
    warning_record_count = sum(x["scanner_warning_count"] > 0 for x in results)
    output = {
        "schema_version": "phase21_model2_causal_baseline_scans_v2",
        "status": "COMPLETE" if failure_count == 0 else "COMPLETE_WITH_EXPLICIT_SCANNER_FAILURES",
        "supersedes_for_qualification_only": {
            "path": "revision/model2/phase21/outputs/model2_causal_baseline_scans.json",
            "sha256": manifest["preserved_v1_scan"]["sha256"],
        },
        "generation_input_sha256": generation_hash,
        "generation_manifest_sha256": manifest["generation_manifest"]["sha256"],
        "warning_amendment_sha256": manifest["warning_amendment"]["sha256"],
        "transfer_manifest_sha256": sha256_file(workdir / manifest_name),
        "semgrep_version": REQUIRED_SEMGREP,
        "registry_config": "p/security-audit",
        "frozen_scanner_source_sha256": manifest["frozen_scanner_source"]["sha256"],
        "fail_closed": True,
        "exit_zero_warnings_are_process_failures": False,
        "code_transformation": "NONE",
        "physical_record_count": len(results),
        "logical_record_count": sum(len(x["logical_mappings"]) for x in results),
        "scanner_success_count": sum(x["scanner_success"] for x in results),
        "scanner_failure_count": failure_count,
        "scanner_skipped_count": skipped_count,
        "warning_record_count": warning_record_count,
        "records": results,
    }
    output_path = workdir / OUTPUT_NAME
    atomic_json(output_path, output)
    returned = {
        "schema_version": "phase21_model2_causal_baseline_scan_return_manifest_v2",
        "status": output["status"],
        "completed_at_utc_diagnostic": datetime.now(timezone.utc).isoformat(),
        "generation_input": {"name": generation_path.name, "sha256": generation_hash},
        "scan_output": {"name": OUTPUT_NAME, "sha256": sha256_file(output_path)},
        "warning_amendment_sha256": manifest["warning_amendment"]["sha256"],
        "transfer_manifest_sha256": sha256_file(workdir / manifest_name),
        "scanner_runner_sha256": sha256_file(Path(__file__).resolve()),
        "frozen_scanner_source_sha256": manifest["frozen_scanner_source"]["sha256"],
        "semgrep_version": REQUIRED_SEMGREP,
        "registry_config": "p/security-audit",
        "fail_closed": True,
        "exit_zero_warnings_are_process_failures": False,
        "code_transformation": "NONE",
        "physical_record_count": len(results),
        "logical_record_count": output["logical_record_count"],
        "scanner_success_count": output["scanner_success_count"],
        "scanner_failure_count": failure_count,
        "scanner_skipped_count": skipped_count,
        "warning_record_count": warning_record_count,
    }
    atomic_json(workdir / RETURN_NAME, returned)
    print(json.dumps(returned, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default=".")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    workdir = Path(args.workdir).resolve()
    print(json.dumps(preflight(workdir, args.manifest), indent=2))
    if not args.preflight_only:
        scan(workdir, args.manifest)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"PHASE21_CAUSAL_BASELINE_RESCAN_V2_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
