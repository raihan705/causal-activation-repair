#!/usr/bin/env python3
"""Fail-closed, resumable Phase 16 scanner handoff for Google Colab.

This wrapper imports the unchanged frozen Phase 9 scanner. It validates frozen
inputs and provenance, performs no security-metric aggregation, and scans the
13 new Phase 16 outputs sequentially.
"""

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


TRANSFER_SHA256 = "5abb65952868d1fbd91129917c2aa5fa5340587b396c46f78afcc18fea166629"
SCANNER_SHA256 = "60cedd3d6ec2d19e5e7e5ba4d709230efdd5d9478047492fefda4dd88d9afbb7"
EXPECTED_ID_HASH = "787a95327bb48f258f9259cac142c048703a56f9efe3c9f505bca344439b80d0"
EXPECTED_COUNT = 575
EXPECTED_ENTRIES = 13
RETURN_NAME = "phase16_scan_return_manifest.json"


class ValidationError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValidationError(f"Cannot read JSON {path}: {exc}") from exc


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def require(value: bool, message: str) -> None:
    if not value:
        raise ValidationError(message)


def require_hash(path: Path, expected: str, label: str) -> str:
    require(path.is_file(), f"Missing {label}: {path}")
    actual = sha256_path(path)
    require(actual == expected, f"{label} hash mismatch: expected {expected}, got {actual}")
    return actual


def resolve_file(workdir: Path, recorded: str) -> Path:
    source = Path(recorded)
    candidates = (workdir / source.name, workdir / source, source)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ValidationError(f"Required file not found for {recorded}")


def semgrep_version(required: bool) -> str:
    try:
        result = subprocess.run(
            ["semgrep", "--version"], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=30, check=False,
        )
    except Exception as exc:
        if required:
            raise ValidationError(f"Semgrep version check failed: {exc}") from exc
        return "NOT_CHECKED_PREFLIGHT_ONLY"
    if result.returncode != 0:
        if required:
            raise ValidationError(f"Semgrep version check failed: {result.stderr.strip()}")
        return "NOT_AVAILABLE_PREFLIGHT_ONLY"
    return (result.stdout or result.stderr).strip()


def direct_records(value: Any, label: str) -> list[dict[str, Any]]:
    require(isinstance(value, list), f"{label} must be a JSON list")
    require(all(isinstance(row, dict) for row in value), f"{label} contains a non-object record")
    return value


def source_records(value: Any, adapter: str, label: str) -> list[dict[str, Any]]:
    if adapter == "DIRECT_GENERATION_RECORDS":
        return direct_records(value, label)
    if adapter == "RCI_FINAL_OUTPUTS_TO_FROZEN_SCANNER_SCHEMA":
        require(isinstance(value, dict), f"{label} must be a JSON object")
        return direct_records(value.get("final_outputs"), f"{label}.final_outputs")
    raise ValidationError(f"Unsupported frozen adapter: {adapter}")


def validate_source(records: list[dict[str, Any]], entry: dict[str, Any], expected_ids: list[Any]) -> None:
    label = f"{entry['method']} seed {entry['seed']}"
    require(len(records) == EXPECTED_COUNT, f"{label} record count is not 575")
    ids = [row.get("prompt_id") for row in records]
    require(ids == expected_ids, f"{label} prompt IDs/order mismatch")
    require(len(set(ids)) == EXPECTED_COUNT, f"{label} prompt IDs are not unique")
    require(all(row.get("method") == entry["method"] for row in records), f"{label} method mismatch")
    require(all(row.get("seed") == entry["seed"] for row in records), f"{label} seed mismatch")
    require(all("generated_code" in row for row in records), f"{label} lacks generated_code")


def validate_icd(path: Path, expected_ids: list[Any], label: str) -> dict[str, int]:
    records = direct_records(read_json(path), f"{label} ICD")
    require(len(records) == EXPECTED_COUNT, f"{label} ICD count is not 575")
    ids = [row.get("prompt_id") for row in records]
    require(ids == expected_ids, f"{label} ICD prompt IDs/order mismatch")
    required = {"prompt_id", "cwe_id", "language", "findings", "vulnerable_cwes", "is_vulnerable", "skipped"}
    for index, row in enumerate(records):
        require(required.issubset(row), f"{label} ICD row {index} lacks frozen fields")
        require(isinstance(row["findings"], list), f"{label} ICD row {index} findings is not a list")
        require(isinstance(row["vulnerable_cwes"], list), f"{label} ICD row {index} vulnerable_cwes is not a list")
        require(isinstance(row["is_vulnerable"], bool), f"{label} ICD row {index} is_vulnerable is not bool")
        require(isinstance(row["skipped"], bool), f"{label} ICD row {index} skipped is not bool")
    skipped = sum(row["skipped"] for row in records)
    return {"record_count": len(records), "eligible_count": len(records) - skipped, "skipped_count": skipped}


def load_scanner(path: Path):
    spec = importlib.util.spec_from_file_location("phase16_frozen_scanner", path)
    require(spec is not None and spec.loader is not None, "Cannot import frozen scanner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(callable(getattr(module, "scan_file", None)), "Frozen scanner has no scan_file function")
    return module.scan_file


def preflight(workdir: Path, manifest_path: Path, require_semgrep: bool) -> dict[str, Any]:
    require_hash(manifest_path, TRANSFER_SHA256, "scan-transfer manifest")
    manifest = read_json(manifest_path)
    require(manifest.get("status") == "READY_NOT_STARTED", "Transfer manifest is not READY_NOT_STARTED")
    require(manifest.get("entry_count") == EXPECTED_ENTRIES, "Transfer entry_count is not 13")
    require(manifest.get("not_started_count") == EXPECTED_ENTRIES, "Transfer not_started_count is not 13")
    require(manifest.get("population_count") == EXPECTED_COUNT, "Transfer population count is not 575")
    require(manifest.get("expected_prompt_ids_sha256") == EXPECTED_ID_HASH, "Frozen ID hash mismatch")
    expected_ids = manifest.get("expected_prompt_ids")
    require(isinstance(expected_ids, list) and len(expected_ids) == EXPECTED_COUNT, "Expected ID list malformed")
    require(sha256_json(expected_ids) == EXPECTED_ID_HASH, "Expected ID list content hash mismatch")
    entries = manifest.get("entries")
    require(isinstance(entries, list) and len(entries) == EXPECTED_ENTRIES, "Transfer entries malformed")
    require([row.get("order_index") for row in entries] == list(range(EXPECTED_ENTRIES)), "Scan order mismatch")

    scanner = resolve_file(workdir, manifest["scanner_path"])
    require_hash(scanner, SCANNER_SHA256, "frozen scanner")
    wrapper = Path(__file__).resolve()
    artifacts = []
    for entry in entries:
        label = f"{entry.get('method')} seed {entry.get('seed')}"
        require(entry.get("scan_status") == "NOT_STARTED", f"{label} transfer status changed")
        require(entry.get("record_count") == EXPECTED_COUNT, f"{label} transfer count mismatch")
        require(entry.get("expected_prompt_ids_sha256") == EXPECTED_ID_HASH, f"{label} ID hash mismatch")
        require(entry.get("scanner_sha256") == SCANNER_SHA256, f"{label} scanner hash mismatch")
        source = resolve_file(workdir, entry["input_path"])
        run_manifest = resolve_file(workdir, entry["run_manifest_path"])
        require_hash(source, entry["input_sha256"], f"{label} generation")
        require_hash(run_manifest, entry["run_manifest_sha256"], f"{label} run manifest")
        records = source_records(read_json(source), entry["adapter"], label)
        validate_source(records, entry, expected_ids)
        scan_input = source
        adapter_path = None
        adapter_sha = None
        if entry["adapter"] == "RCI_FINAL_OUTPUTS_TO_FROZEN_SCANNER_SCHEMA":
            adapter_path = (workdir / "rci_1_seed42_scanner_input.json").resolve()
            atomic_json(adapter_path, records)
            adapter_sha = sha256_path(adapter_path)
            adapter_records = direct_records(read_json(adapter_path), "RCI scanner adapter")
            require(adapter_records == records, "RCI scanner adapter changed final_outputs")
            scan_input = adapter_path
        output = (workdir / Path(entry["scan_output_path"]).name).resolve()
        artifacts.append({
            "order_index": entry["order_index"], "method": entry["method"], "seed": entry["seed"],
            "information_tier": entry["information_tier"], "adapter": entry["adapter"],
            "generation_path": str(source), "generation_sha256": entry["input_sha256"],
            "run_manifest_path": str(run_manifest), "run_manifest_sha256": entry["run_manifest_sha256"],
            "scanner_input_path": str(scan_input), "scanner_adapter_path": str(adapter_path) if adapter_path else None,
            "scanner_adapter_sha256": adapter_sha, "icd_path": str(output),
        })
    return {
        "schema_version": "phase16_scan_preflight_v1", "status": "PASS",
        "captured_at_utc": utc_now(), "transfer_manifest_sha256": TRANSFER_SHA256,
        "scanner_path": str(scanner), "scanner_sha256": SCANNER_SHA256,
        "wrapper_path": str(wrapper), "wrapper_sha256": sha256_path(wrapper),
        "expected_prompt_ids": expected_ids, "expected_prompt_ids_sha256": EXPECTED_ID_HASH,
        "record_count_per_condition": EXPECTED_COUNT, "condition_count": EXPECTED_ENTRIES,
        "semgrep_version": semgrep_version(require_semgrep), "platform": platform.platform(),
        "python_version": platform.python_version(), "artifacts": artifacts,
        "security_metrics_computed": False,
    }


def new_return(preflight_result: dict[str, Any]) -> dict[str, Any]:
    entries = []
    for item in preflight_result["artifacts"]:
        entries.append({**item, "scan_status": "NOT_STARTED", "completion_order": None,
                        "icd_sha256": None, "scanner_record_count": None,
                        "eligible_count": None, "skipped_count": None,
                        "generation_hash_unchanged": None, "started_at_utc": None,
                        "completed_at_utc": None, "error": None})
    return {
        "schema_version": "phase16_scan_return_manifest_v1", "phase": 16,
        "status": "SCANNING_IN_PROGRESS", "created_at_utc": utc_now(), "updated_at_utc": utc_now(),
        "transfer_manifest_sha256": TRANSFER_SHA256, "scanner_sha256": SCANNER_SHA256,
        "wrapper_sha256": preflight_result["wrapper_sha256"],
        "expected_prompt_ids_sha256": EXPECTED_ID_HASH,
        "environment": {"platform": preflight_result["platform"],
                        "python_version": preflight_result["python_version"],
                        "semgrep_version": preflight_result["semgrep_version"]},
        "entries": entries, "complete_count": 0, "total_scanner_records": 0,
        "security_metrics_computed": False,
    }


def validate_completed(entry: dict[str, Any], current: dict[str, Any], ids: list[Any]) -> None:
    for field in ("method", "seed", "generation_sha256", "run_manifest_sha256", "adapter", "scanner_adapter_sha256"):
        require(entry.get(field) == current.get(field), f"Completed {current['method']} linkage mismatch: {field}")
    output = Path(current["icd_path"])
    require_hash(output, entry["icd_sha256"], f"completed {current['method']} ICD")
    counts = validate_icd(output, ids, current["method"])
    require(counts == {"record_count": entry["scanner_record_count"], "eligible_count": entry["eligible_count"],
                       "skipped_count": entry["skipped_count"]}, f"Completed {current['method']} counts changed")


def run_all(workdir: Path, preflight_result: dict[str, Any], resume: bool) -> Path:
    return_path = workdir / RETURN_NAME
    if return_path.exists():
        require(resume, f"{RETURN_NAME} exists; use --resume")
        manifest = read_json(return_path)
        require(manifest.get("transfer_manifest_sha256") == TRANSFER_SHA256, "Return transfer hash mismatch")
        require(manifest.get("scanner_sha256") == SCANNER_SHA256, "Return scanner hash mismatch")
        require(manifest.get("wrapper_sha256") == preflight_result["wrapper_sha256"], "Return wrapper hash mismatch")
        require(len(manifest.get("entries", [])) == EXPECTED_ENTRIES, "Return entries malformed")
    else:
        manifest = new_return(preflight_result)
        atomic_json(return_path, manifest)
    ids = preflight_result["expected_prompt_ids"]
    scan_file = load_scanner(Path(preflight_result["scanner_path"]))

    for index, current in enumerate(preflight_result["artifacts"]):
        entry = manifest["entries"][index]
        label = f"{current['method']} seed {current['seed']}"
        if entry.get("scan_status") == "COMPLETE":
            validate_completed(entry, current, ids)
            print(f"REUSED VALIDATED COMPLETE: {label}")
            continue
        require(entry.get("scan_status") == "NOT_STARTED", f"Unsafe return status for {label}")
        require_hash(Path(current["generation_path"]), current["generation_sha256"], f"{label} generation")
        require_hash(Path(current["run_manifest_path"]), current["run_manifest_sha256"], f"{label} run manifest")
        require_hash(Path(preflight_result["scanner_path"]), SCANNER_SHA256, "frozen scanner")
        if current["scanner_adapter_path"]:
            require_hash(Path(current["scanner_adapter_path"]), current["scanner_adapter_sha256"], "RCI adapter")
        output = Path(current["icd_path"])
        temporary = output.with_name(f".{output.name}.scan-tmp")
        require(not output.exists(), f"Unlinked existing ICD refuses overwrite: {output}")
        if temporary.exists():
            temporary.unlink()
        entry["started_at_utc"] = utc_now()
        atomic_json(return_path, manifest)
        print(f"SCANNING {index + 1}/{EXPECTED_ENTRIES}: {label}")
        scan_file(Path(current["scanner_input_path"]), temporary)
        require(temporary.is_file(), f"Frozen scanner did not create {label} output")
        counts = validate_icd(temporary, ids, label)
        os.replace(temporary, output)
        require_hash(Path(current["generation_path"]), current["generation_sha256"], f"post-scan {label} generation")
        entry.update({"scan_status": "COMPLETE", "completion_order": 1 + sum(
            row.get("scan_status") == "COMPLETE" for row in manifest["entries"]),
            "icd_sha256": sha256_path(output), "scanner_record_count": counts["record_count"],
            "eligible_count": counts["eligible_count"], "skipped_count": counts["skipped_count"],
            "generation_hash_unchanged": True, "completed_at_utc": utc_now(), "error": None})
        manifest["complete_count"] = sum(row.get("scan_status") == "COMPLETE" for row in manifest["entries"])
        manifest["total_scanner_records"] = sum(row.get("scanner_record_count") or 0 for row in manifest["entries"])
        manifest["updated_at_utc"] = utc_now()
        manifest["status"] = "SCANS_COMPLETE" if manifest["complete_count"] == EXPECTED_ENTRIES else "SCANNING_IN_PROGRESS"
        atomic_json(return_path, manifest)
        print(f"COMPLETE: {label} records={counts['record_count']} eligible={counts['eligible_count']} skipped={counts['skipped_count']}")
    require(manifest["complete_count"] == EXPECTED_ENTRIES, "Not all 13 scans completed")
    require(manifest["total_scanner_records"] == EXPECTED_ENTRIES * EXPECTED_COUNT, "Final scanner record total mismatch")
    print(f"ALL 13 SCANS COMPLETE: {return_path}")
    print(f"Return manifest SHA-256: {sha256_path(return_path)}")
    return return_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.preflight_only and not args.all:
        parser.error("Scanning requires --all")
    return args


def main() -> int:
    args = parse_args()
    workdir = args.workdir.expanduser().resolve()
    require(workdir.is_dir(), f"Workdir does not exist: {workdir}")
    manifest_path = args.manifest.expanduser().resolve() if args.manifest else workdir / "phase16_scan_transfer_manifest.json"
    result = preflight(workdir, manifest_path, require_semgrep=not args.preflight_only)
    atomic_json(workdir / "phase16_scan_preflight.json", result)
    print(f"PREFLIGHT PASS: {EXPECTED_ENTRIES} conditions, {EXPECTED_COUNT} records each")
    if args.preflight_only:
        return 0
    run_all(workdir, result, resume=args.resume)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        raise SystemExit(2)
