#!/usr/bin/env python3
"""Run one excluded warm-up and three fixed-input Phase 18 scanner timings."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


SCANNER_SHA = "60cedd3d6ec2d19e5e7e5ba4d709230efdd5d9478047492fefda4dd88d9afbb7"
MANIFEST_NAME = "phase18_scan_transfer_manifest.json"
RETURN_NAME = "phase18_scanner_timing.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_scan(path: Path):
    spec = importlib.util.spec_from_file_location("phase18_frozen_scanner", path)
    require(spec is not None and spec.loader is not None, "Cannot import frozen scanner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(callable(getattr(module, "scan_file", None)), "Frozen scanner lacks scan_file")
    return module.scan_file


def semgrep_version(required: bool) -> str:
    result = subprocess.run(["semgrep", "--version"], capture_output=True, text=True, timeout=30, check=False)
    if result.returncode != 0:
        if required:
            raise RuntimeError(result.stderr.strip() or "Semgrep unavailable")
        return "NOT_AVAILABLE_PREFLIGHT_ONLY"
    return (result.stdout or result.stderr).strip()


def validate_icd(path: Path, ids: list[int]) -> None:
    rows = read_json(path)
    require(isinstance(rows, list) and len(rows) == 50, "ICD count mismatch")
    require([int(row["prompt_id"]) for row in rows] == ids, "ICD prompt order mismatch")


def preflight(workdir: Path, require_semgrep: bool) -> dict[str, Any]:
    manifest_path = workdir / MANIFEST_NAME
    require(manifest_path.is_file(), "Transfer manifest missing")
    manifest = read_json(manifest_path)
    require(manifest.get("status") == "READY_NOT_STARTED" and manifest.get("record_count") == 50, "Transfer manifest mismatch")
    input_path = workdir / manifest["scanner_input"]
    scanner_path = workdir / manifest["scanner"]
    require(input_path.is_file() and sha256(input_path) == manifest["scanner_input_sha256"], "Scanner input mismatch")
    require(scanner_path.is_file() and sha256(scanner_path) == SCANNER_SHA == manifest["scanner_sha256"], "Scanner mismatch")
    ids = list(map(int, manifest["expected_prompt_ids"]))
    require([int(row["prompt_id"]) for row in read_json(input_path)] == ids, "Scanner input order mismatch")
    return {"status": "PASS", "manifest_sha256": sha256(manifest_path), "input_sha256": sha256(input_path),
            "scanner_sha256": SCANNER_SHA, "semgrep_version": semgrep_version(require_semgrep),
            "python_version": platform.python_version(), "platform": platform.platform(), "ids": ids,
            "input_path": input_path, "scanner_path": scanner_path}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, default=Path("."))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    workdir = args.workdir.resolve()
    state = preflight(workdir, require_semgrep=not args.preflight_only)
    if args.preflight_only:
        print(json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in state.items() if key != "ids"}, indent=2))
        return
    scan_file = load_scan(state["scanner_path"])
    warmup = workdir / ".phase18_scanner_warmup.json"
    if warmup.exists():
        warmup.unlink()
    scan_file(state["input_path"], warmup)
    validate_icd(warmup, state["ids"])
    warmup.unlink()

    repetitions = []
    reference_hash = None
    reference_path = workdir / "b0_repetition1_icd.json"
    for repetition in (1, 2, 3):
        temporary = workdir / f".phase18_scanner_rep{repetition}.json"
        if temporary.exists():
            temporary.unlink()
        started = time.perf_counter()
        scan_file(state["input_path"], temporary)
        elapsed = time.perf_counter() - started
        validate_icd(temporary, state["ids"])
        output_hash = sha256(temporary)
        if reference_hash is None:
            reference_hash = output_hash
            shutil.copy2(temporary, reference_path)
        else:
            require(output_hash == reference_hash, "Scanner outputs differ across identical repetitions")
        repetitions.append({"repetition": repetition, "record_count": 50, "wall_seconds": elapsed,
                            "input_sha256": state["input_sha256"], "output_sha256": output_hash,
                            "output_identity_verified": True})
        temporary.unlink()
    result = {
        "schema_version": "phase18_scanner_timing_v1", "phase": 18, "status": "COMPLETE",
        "warmup_full_scan_count": 1, "warmup_excluded": True, "measured_full_scan_count": 3,
        "fixed_input_sha256": state["input_sha256"], "output_identity_sha256": reference_hash,
        "designated_icd_path": reference_path.name, "designated_icd_sha256": sha256(reference_path),
        "scanner_sha256": state["scanner_sha256"], "semgrep_version": state["semgrep_version"],
        "python_version": state["python_version"], "platform": state["platform"],
        "measured_repetitions": repetitions, "heldout_used": False,
    }
    atomic_json(workdir / RETURN_NAME, result)
    print(json.dumps({"status": "COMPLETE", "times": [row["wall_seconds"] for row in repetitions],
                      "output_sha256": reference_hash}, indent=2))


if __name__ == "__main__":
    main()
