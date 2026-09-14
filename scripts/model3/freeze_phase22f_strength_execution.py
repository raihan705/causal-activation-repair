#!/usr/bin/env python
"""Freeze Stage 22F execution manifest after protocol and runner exist."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import run_phase22f_strength_calibration as runner

ROOT = Path(__file__).resolve().parents[4]
PHASE = ROOT / "revision/model3/phase22"
OUT = PHASE / "outputs"
PROTOCOL = OUT / "phase22f_strength_calibration_protocol.json"
CAUSAL_PROTOCOL = OUT / "phase22e_causal_validation_protocol.json"
RUNNER = PHASE / "scripts/run_phase22f_strength_calibration.py"
SHARED = PHASE / "scripts/run_phase22e_causal_steering.py"
OUTPUT = OUT / "phase22f_strength_execution_protocol.json"


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


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError("immutable Stage 22F execution protocol already exists")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    causal = json.loads(CAUSAL_PROTOCOL.read_text(encoding="utf-8"))
    worklist = runner.build_worklist(protocol, causal)
    value = {
        "schema_version": "phase22f_strength_execution_protocol_v1", "status": "FROZEN_BEFORE_STRENGTH_GENERATION",
        "strength_protocol_sha256": sha256_file(PROTOCOL), "causal_protocol_sha256": sha256_file(CAUSAL_PROTOCOL),
        "runner_path": "revision/model3/phase22/scripts/run_phase22f_strength_calibration.py",
        "runner_sha256": sha256_file(RUNNER), "shared_helper_path": "revision/model3/phase22/scripts/run_phase22e_causal_steering.py",
        "shared_helper_sha256": sha256_file(SHARED), "worklist_count": len(worklist),
        "worklist_canonical_sha256": runner.canonical_sha256(worklist), "checkpoint_interval": 5,
        "new_strength_multipliers": [0.5, 1.5, 2.0],
        "order": protocol["execution_order"], "resume_rule": protocol["failure_handling"]["checkpoint"],
        "model": causal["model"], "sae": causal["sae"], "generation": causal["generation"],
        "scanner_run": False, "stage22g_started": False, "heldout_used": False,
    }
    atomic_json(OUTPUT, value)
    print(json.dumps({**value, "execution_protocol_sha256": sha256_file(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
