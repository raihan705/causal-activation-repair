#!/usr/bin/env python3
"""Sequentially supervise the 13 frozen Phase 16A generation conditions."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUTPUTS = ROOT / "revision/model1/phase16/outputs"
STATE = OUTPUTS / "phase16_generation_controller_state.json"
FREEZE = "revision/model1/phase15/outputs/revision_freeze_manifest.json"

ORDER = [
    ("primary", "B0", 43), ("primary", "B0", 44),
    ("primary", "B1", 43), ("primary", "B1", 44),
    ("primary", "B*", 43), ("primary", "B*", 44),
    ("primary", "B2-alpha20", 42), ("primary", "B2-alpha20", 43), ("primary", "B2-alpha20", 44),
    ("comparator", "B3-ungated", 42), ("comparator", "B1-CWE", 42),
    ("comparator", "RCI-1", 42), ("comparator", "CAA-CWE", 42),
]

def now() -> str:
    return datetime.now(timezone.utc).isoformat()

def write(value: dict) -> None:
    temporary = STATE.with_name(STATE.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(STATE)

def main() -> None:
    state = {"schema_version": "phase16_generation_controller_v1", "status": "RUNNING", "started_at_utc": now(),
             "condition_order": [{"kind": k, "method": m, "seed": s} for k, m, s in ORDER], "completed": [], "active": None, "failure": None}
    write(state)
    for index, (kind, method, seed) in enumerate(ORDER):
        runner = "revision/model1/phase16/scripts/run_primary_seed_extension.py" if kind == "primary" else "revision/model1/phase16/scripts/run_single_seed_comparators.py"
        command = [sys.executable, runner, "--freeze", FREEZE, "--method", method, "--seed", str(seed), "--resume"]
        state["active"] = {"index": index, "kind": kind, "method": method, "seed": seed, "started_at_utc": now(), "command": command}
        write(state)
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode != 0:
            state["status"] = "FAILED"; state["failure"] = {**state["active"], "returncode": result.returncode, "failed_at_utc": now()}; state["active"] = None; write(state); raise SystemExit(result.returncode)
        state["completed"].append({"index": index, "method": method, "seed": seed, "completed_at_utc": now()}); state["active"] = None; write(state)
    state["status"] = "COMPLETE"; state["completed_at_utc"] = now(); write(state)

if __name__ == "__main__":
    main()
