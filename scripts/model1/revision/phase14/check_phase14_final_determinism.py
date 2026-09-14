#!/usr/bin/env python3
"""Run deterministic Phase 14 finalization twice and require byte-identical outputs."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[3]
OUTPUTS = ROOT / "revision" / "model1" / "phase14" / "outputs"
FINALIZER = SCRIPT_DIR / "finalize_caa_stage_b.py"
TARGETS = (
    "caa_stage_b_metrics.json",
    "caa_stage_b_selection.json",
    "caa_cwe_config.json",
    "phase14_method_statuses.json",
    "phase14_final_development_summary.json",
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_once() -> dict[str, str]:
    subprocess.run([sys.executable, str(FINALIZER)], cwd=ROOT, check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return {name: sha256_path(OUTPUTS / name) for name in TARGETS}


def main() -> None:
    first = run_once()
    second = run_once()
    identical = first == second
    result = {
        "schema_version": "phase14_final_determinism_v1",
        "status": "PASS" if identical else "FAIL",
        "scope": "DETERMINISTIC_METRIC_SELECTION_CONFIG_STATUS_SUMMARY_ONLY",
        "generation_executed": False,
        "scanner_executed": False,
        "held_out_accessed": False,
        "finalizer_path": FINALIZER.relative_to(ROOT).as_posix(),
        "finalizer_sha256": sha256_path(FINALIZER),
        "first_run_hashes": first,
        "second_run_hashes": second,
        "byte_identical": identical,
    }
    output = OUTPUTS / "phase14_final_determinism.json"
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    if not identical:
        raise SystemExit("Phase 14 finalizer outputs were not byte-identical")


if __name__ == "__main__":
    main()
