#!/usr/bin/env python3
"""Fail-closed Colab scanner for frozen Model2 causal-baseline generations."""

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
DEFAULT_MANIFEST = "model2_causal_baseline_scan_transfer_manifest.json"
OUTPUT_NAME = "model2_causal_baseline_scans.json"
RETURN_NAME = "model2_causal_baseline_scan_return_manifest.json"


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
    text = (result.stdout or result.stderr).strip()
    return text.splitlines()[-1].strip()


def load_frozen_scanner(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("frozen_model2_scanner", path)
    require(spec is not None and spec.loader is not None, "cannot load frozen scanner source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_bundle(workdir: Path, manifest_name: str) -> tuple[dict[str, Any], Any]:
    manifest_path = workdir / manifest_name
    require(manifest_path.is_file(), f"missing transfer manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("schema_version") == "phase21_model2_causal_baseline_scan_transfer_manifest_v1", "transfer manifest schema mismatch")
    require(manifest.get("status") == "FROZEN_FOR_COLAB_SCAN", "transfer manifest is not frozen")
    for item in manifest["immutable_files"]:
        path = workdir / item["name"]
        require(path.is_file(), f"missing immutable bundle file: {item['name']}")
        require(sha256_file(path) == item["sha256"], f"immutable bundle file drift: {item['name']}")
    version = semgrep_version()
    require(version == REQUIRED_SEMGREP, f"Semgrep version mismatch: expected {REQUIRED_SEMGREP}, got {version}")
    frozen_path = workdir / manifest["frozen_scanner_source"]["name"]
    frozen = load_frozen_scanner(frozen_path)
    return manifest, frozen


def run_semgrep_fail_closed(code: str, language: str, frozen: Any) -> dict[str, Any]:
    ext = frozen.get_ext(language)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, mode="w", delete=False, encoding="utf-8") as tmp:
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
            return {
                "success": False,
                "exit_code": result.returncode,
                "timeout": False,
                "error": f"Semgrep nonzero exit: {result.stderr.strip()}",
                "findings": [],
            }
        try:
            data = json.loads(result.stdout)
        except Exception as exc:
            return {"success": False, "exit_code": result.returncode, "timeout": False, "error": f"Semgrep JSON parse failure: {type(exc).__name__}: {exc}", "findings": []}
        if data.get("errors"):
            return {"success": False, "exit_code": result.returncode, "timeout": False, "error": f"Semgrep reported errors: {json.dumps(data['errors'], ensure_ascii=False)}", "findings": []}
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
        return {"success": True, "exit_code": result.returncode, "timeout": False, "error": None, "findings": findings}
    except subprocess.TimeoutExpired as exc:
        return {"success": False, "exit_code": None, "timeout": True, "error": f"Semgrep timeout after {exc.timeout} seconds", "findings": []}
    except Exception as exc:
        return {"success": False, "exit_code": None, "timeout": False, "error": f"Semgrep exception: {type(exc).__name__}: {exc}", "findings": []}
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def preflight(workdir: Path, manifest_name: str) -> dict[str, Any]:
    manifest, _ = validate_bundle(workdir, manifest_name)
    generation_path = workdir / manifest["generation_input"]["name"]
    generations = json.loads(generation_path.read_text(encoding="utf-8"))
    require(generations.get("physical_record_count") == 45, "generation physical count mismatch")
    require(len(generations.get("records", [])) == 45, "generation record list mismatch")
    return {
        "schema_version": "phase21_model2_causal_baseline_scan_preflight_v1",
        "status": "PASS",
        "semgrep_version": REQUIRED_SEMGREP,
        "generation_sha256": sha256_file(generation_path),
        "physical_record_count": 45,
        "scanner_run": False,
        "fail_closed": True,
    }


def scan(workdir: Path, manifest_name: str) -> None:
    manifest, frozen = validate_bundle(workdir, manifest_name)
    generation_path = workdir / manifest["generation_input"]["name"]
    generation_hash = sha256_file(generation_path)
    require(generation_hash == manifest["generation_input"]["sha256"], "generation hash mismatch")
    generations = json.loads(generation_path.read_text(encoding="utf-8"))
    records = generations["records"]
    require(len(records) == 45, "expected exactly 45 physical generation records")
    require(sum(len(x["logical_mappings"]) for x in records) == 45, "logical generation accounting mismatch")
    results = []
    for index, rec in enumerate(records):
        code = rec.get("generated_code", "")
        language = str(rec.get("language", "")).lower()
        eligible = language in frozen.TARGET_LANGUAGES and bool(code.strip())
        if not eligible:
            semgrep = {"success": False, "exit_code": None, "timeout": False, "error": "SCANNER_INELIGIBLE_UNSUPPORTED_LANGUAGE_OR_EMPTY_CODE", "findings": []}
            regex_findings = []
            all_findings = []
            scanner_status = "SKIPPED_INELIGIBLE"
        else:
            semgrep = run_semgrep_fail_closed(code, language, frozen)
            regex_findings = frozen.run_regex(code)
            all_findings = frozen.dedup_findings(semgrep["findings"] + regex_findings) if semgrep["success"] else regex_findings
            scanner_status = "SUCCESS" if semgrep["success"] else "FAILED_CLOSED"
        vulnerable_cwes = sorted({x["cwe_id"] for x in all_findings})
        target_flags = []
        for mapping in rec["logical_mappings"]:
            target = mapping.get("target_cwe")
            target_flags.append({
                "logical_index": mapping["logical_index"],
                "population": mapping["population"],
                "target_cwe": target,
                "target_cwe_present": bool(target and target in vulnerable_cwes) if semgrep["success"] else None,
            })
        results.append({
            "schema_version": "phase21_model2_causal_baseline_scan_record_v1",
            "physical_index": rec["physical_index"],
            "prompt_id": rec["prompt_id"],
            "source_index": rec["source_index"],
            "logical_mappings": rec["logical_mappings"],
            "language": language,
            "scanner_eligible": eligible,
            "scanner_success": bool(semgrep["success"]),
            "scanner_status": scanner_status,
            "scanner_exit_status": semgrep["exit_code"],
            "scanner_timeout": semgrep["timeout"],
            "scanner_error": semgrep["error"],
            "findings": all_findings,
            "mapped_cwe_labels": vulnerable_cwes,
            "target_cwe_flags": target_flags,
            "any_vulnerability": bool(all_findings) if semgrep["success"] else None,
            "historical_derivation_is_vulnerable": any(x["cwe_id"] in frozen.DERIVATION_CWES for x in all_findings) if semgrep["success"] else None,
            "skipped": not eligible,
            "input_generation_sha256": generation_hash,
            "generated_text_sha256": rec["generated_text_sha256"],
        })
        print(f"SCAN {index + 1}/45 prompt_id={rec['prompt_id']} status={scanner_status} findings={len(all_findings)}", flush=True)

    failure_count = sum(not x["scanner_success"] for x in results if x["scanner_eligible"])
    skipped_count = sum(x["skipped"] for x in results)
    output = {
        "schema_version": "phase21_model2_causal_baseline_scans_v1",
        "status": "COMPLETE" if failure_count == 0 else "COMPLETE_WITH_EXPLICIT_SCANNER_FAILURES",
        "generation_input_sha256": generation_hash,
        "generation_manifest_sha256": manifest["generation_manifest"]["sha256"],
        "transfer_manifest_sha256": sha256_file(workdir / manifest_name),
        "semgrep_version": REQUIRED_SEMGREP,
        "registry_config": "p/security-audit",
        "frozen_scanner_source_sha256": manifest["frozen_scanner_source"]["sha256"],
        "fail_closed": True,
        "physical_record_count": len(results),
        "logical_record_count": sum(len(x["logical_mappings"]) for x in results),
        "scanner_success_count": sum(x["scanner_success"] for x in results),
        "scanner_failure_count": failure_count,
        "scanner_skipped_count": skipped_count,
        "records": results,
    }
    output_path = workdir / OUTPUT_NAME
    atomic_json(output_path, output)
    return_manifest = {
        "schema_version": "phase21_model2_causal_baseline_scan_return_manifest_v1",
        "status": output["status"],
        "completed_at_utc_diagnostic": datetime.now(timezone.utc).isoformat(),
        "generation_input": {"name": generation_path.name, "sha256": generation_hash},
        "scan_output": {"name": OUTPUT_NAME, "sha256": sha256_file(output_path)},
        "transfer_manifest_sha256": sha256_file(workdir / manifest_name),
        "scanner_runner_sha256": sha256_file(Path(__file__).resolve()),
        "frozen_scanner_source_sha256": manifest["frozen_scanner_source"]["sha256"],
        "semgrep_version": REQUIRED_SEMGREP,
        "registry_config": "p/security-audit",
        "fail_closed": True,
        "physical_record_count": len(results),
        "logical_record_count": output["logical_record_count"],
        "scanner_success_count": output["scanner_success_count"],
        "scanner_failure_count": failure_count,
        "scanner_skipped_count": skipped_count,
    }
    atomic_json(workdir / RETURN_NAME, return_manifest)
    print(json.dumps(return_manifest, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default=".")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    workdir = Path(args.workdir).resolve()
    result = preflight(workdir, args.manifest)
    print(json.dumps(result, indent=2))
    if not args.preflight_only:
        scan(workdir, args.manifest)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"PHASE21_CAUSAL_BASELINE_SCAN_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
