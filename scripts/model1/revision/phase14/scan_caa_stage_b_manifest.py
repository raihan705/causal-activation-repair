#!/usr/bin/env python3
"""Validate and sequentially scan the three frozen CAA Stage B generations."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any


SCANNER_SHA256 = "60cedd3d6ec2d19e5e7e5ba4d709230efdd5d9478047492fefda4dd88d9afbb7"
SUBSET_SHA256 = "3f023091c601fd6b4ffe8c074ad927f4bd8e32d211e0a98d81d7f486aa355a0d"
EXPECTED = [("caa_l16_m0p5", 0.5), ("caa_l16_m2", 2.0), ("caa_l16_m4", 4.0)]


class ValidationError(RuntimeError):
    pass


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    if temporary.exists():
        raise ValidationError(f"stale temporary file: {temporary}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def semgrep_version(required: bool) -> str:
    try:
        result = subprocess.run(["semgrep", "--version"], capture_output=True, text=True, timeout=30)
        value = (result.stdout or result.stderr).strip()
        if result.returncode or not value:
            raise RuntimeError(value)
        return value
    except Exception as exc:
        if required:
            raise ValidationError(f"Semgrep unavailable: {exc}") from exc
        return "NOT_AVAILABLE_IN_PREFLIGHT_ENVIRONMENT"


def load_scanner(path: Path):
    spec = importlib.util.spec_from_file_location("phase14_stage_b_frozen_scanner", path)
    if spec is None or spec.loader is None:
        raise ValidationError("cannot import scanner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scan_file = getattr(module, "scan_file", None)
    if not callable(scan_file) or list(inspect.signature(scan_file).parameters) != ["out_path", "icd_path"]:
        raise ValidationError("unexpected frozen scan_file interface")
    return scan_file


def validate_icd(path: Path, expected_ids: list[int]) -> dict[str, int]:
    rows = read_json(path)
    ids = [int(row["prompt_id"]) for row in rows]
    if len(rows) != 120 or ids != expected_ids or len(set(ids)) != 120:
        raise ValidationError(f"ICD population/order mismatch: {path}")
    required = {"prompt_id", "cwe_id", "language", "findings", "vulnerable_cwes", "is_vulnerable", "skipped"}
    if any(not required.issubset(row) for row in rows):
        raise ValidationError(f"ICD schema mismatch: {path}")
    skipped = sum(bool(row["skipped"]) for row in rows)
    return {"record_count": len(rows), "eligible_count": len(rows) - skipped, "skipped_count": skipped}


def preflight(workdir: Path, manifest_path: Path, require_semgrep: bool) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != "phase14_caa_stage_b_scan_transfer_manifest_v1":
        raise ValidationError("manifest schema mismatch")
    entries = manifest.get("entries", [])
    if len(entries) != 3:
        raise ValidationError("manifest must contain exactly three entries")
    subset_path = workdir / entries[0]["prompt_manifest_path"]
    if sha256_path(subset_path) != SUBSET_SHA256:
        raise ValidationError("subset hash mismatch")
    subset = read_json(subset_path)
    if subset.get("split") != "DEVELOPMENT" or subset.get("validation_checks", {}).get("no_heldout_prompt_ids") is not True:
        raise ValidationError("subset is not frozen no-held-out development data")
    expected_ids = [int(value) for value in subset["prompt_ids_source_order"]]
    scanner_path = workdir / entries[0]["frozen_scanner_source_path"]
    if sha256_path(scanner_path) != SCANNER_SHA256:
        raise ValidationError("scanner hash mismatch")
    load_scanner(scanner_path)
    verified = []
    for order, (entry, (key, multiplier)) in enumerate(zip(entries, EXPECTED), 1):
        if (entry.get("order"), entry.get("key"), entry.get("method"), entry.get("layer"), entry.get("multiplier"), entry.get("seed"), entry.get("scan_status")) != (order, key, "CAA-CWE-StageB", 16, multiplier, 42, "NOT_STARTED"):
            raise ValidationError(f"entry {order} frozen condition mismatch")
        generation = workdir / entry["generation_path"]
        run_path = workdir / entry["run_manifest_path"]
        vector_path = workdir / entry["vector_manifest_path"]
        if sha256_path(vector_path) != entry["vector_manifest_sha256"]:
            raise ValidationError(f"{key} vector-manifest hash mismatch")
        if entry.get("frozen_scanner_source_sha256") != SCANNER_SHA256:
            raise ValidationError(f"{key} scanner hash declaration mismatch")
        for path, expected_hash, label in ((generation, entry["generation_sha256"], "generation"), (run_path, entry["run_manifest_sha256"], "run manifest")):
            if sha256_path(path) != expected_hash:
                raise ValidationError(f"{key} {label} hash mismatch")
        rows = read_json(generation)
        ids = [int(row["prompt_id"]) for row in rows]
        if len(rows) != 120 or ids != expected_ids or len(set(ids)) != 120:
            raise ValidationError(f"{key} population/order mismatch")
        if sum(bool(row.get("intervention_applied")) for row in rows) != 90:
            raise ValidationError(f"{key} active routing mismatch")
        if any((row.get("condition", {}).get("stage"), row.get("condition", {}).get("layer"), row.get("condition", {}).get("multiplier"), row.get("seed")) != ("B", 16, multiplier, 42) for row in rows):
            raise ValidationError(f"{key} record condition mismatch")
        run = read_json(run_path)
        if run.get("status") != "COMPLETE" or run.get("output_sha256") != entry["generation_sha256"]:
            raise ValidationError(f"{key} run/output linkage mismatch")
        verified.append({"key": key, "generation_path": str(generation), "generation_sha256": entry["generation_sha256"], "run_manifest_path": str(run_path), "run_manifest_sha256": entry["run_manifest_sha256"], "vector_manifest_sha256": entry["vector_manifest_sha256"], "icd_path": str(workdir / entry["icd_path"]), "expected_ids": expected_ids})
    return {"status": "PASS", "manifest_sha256": sha256_path(manifest_path), "scanner_path": str(scanner_path), "scanner_sha256": SCANNER_SHA256, "python_version": platform.python_version(), "semgrep_version": semgrep_version(require_semgrep), "entries": verified, "held_out_accessed": False}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("caa_stage_b_scan_transfer_manifest.json"))
    parser.add_argument("--workdir", type=Path, default=Path("."))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    workdir = args.workdir.resolve()
    manifest_path = args.manifest if args.manifest.is_absolute() else workdir / args.manifest
    if not args.preflight_only and not args.all:
        parser.error("scanning requires --all")
    pf = preflight(workdir, manifest_path.resolve(), require_semgrep=not args.preflight_only)
    if args.preflight_only:
        print(json.dumps(pf, indent=2, sort_keys=True))
        return 0
    scanner = load_scanner(Path(pf["scanner_path"]))
    return_path = workdir / "caa_stage_b_scan_return_manifest.json"
    if return_path.exists():
        if not args.resume:
            raise ValidationError("return manifest exists; use --resume")
        result = read_json(return_path)
        if result.get("transfer_manifest_sha256") != pf["manifest_sha256"]:
            raise ValidationError("resume manifest linkage mismatch")
    else:
        result = {"schema_version": "phase14_caa_stage_b_scan_return_manifest_v1", "status": "IN_PROGRESS", "transfer_manifest_sha256": pf["manifest_sha256"], "python_version": pf["python_version"], "semgrep_version": pf["semgrep_version"], "scanner_sha256": SCANNER_SHA256, "entries": [], "held_out_accessed": False}
        atomic_json(return_path, result)
    completed = {entry["key"]: entry for entry in result["entries"]}
    for order, entry in enumerate(pf["entries"], 1):
        if entry["key"] in completed:
            current = completed[entry["key"]]
            icd = Path(current["icd_path"])
            if not icd.is_absolute():
                icd = workdir / icd.name
            counts = validate_icd(icd, entry["expected_ids"])
            if sha256_path(icd) != current["icd_sha256"] or counts != {k: current[k] for k in counts}:
                raise ValidationError(f"completed resume result changed: {entry['key']}")
            continue
        icd = Path(entry["icd_path"])
        temporary = icd.with_name("." + icd.name + ".scan.tmp")
        scanner(Path(entry["generation_path"]), temporary)
        counts = validate_icd(temporary, entry["expected_ids"])
        os.replace(temporary, icd)
        frozen = {k: v for k, v in entry.items() if k != "expected_ids"}
        frozen.update(counts)
        frozen.update({"icd_path": icd.name, "icd_sha256": sha256_path(icd), "scanner_sha256": SCANNER_SHA256, "python_version": pf["python_version"], "semgrep_version": pf["semgrep_version"], "completion_order": order, "scan_status": "COMPLETE", "resume_status": "NEWLY_SCANNED"})
        result["entries"].append(frozen)
        atomic_json(return_path, result)
    result["status"] = "COMPLETE"
    result["complete_count"] = 3
    atomic_json(return_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
