#!/usr/bin/env python3
"""Bind the reduced routewise-utility protocol to exact executables."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from routewise_utility_common import OUTPUTS, atomic_json, read_json, relative, require, sha256_path


SCRIPTS = Path(__file__).resolve().parent
PROTOCOL = OUTPUTS / "reduced_routewise_utility_protocol.json"
APPROVAL = OUTPUTS / "reduced_routewise_utility_execution_approval.json"
RUNNER = SCRIPTS / "run_reduced_routewise_utility.py"
FULL_RUNNER = SCRIPTS / "run_routewise_utility.py"
COMMON = SCRIPTS / "routewise_utility_common.py"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approval-basis", required=True)
    args = parser.parse_args()
    require(PROTOCOL.is_file(), "Reduced protocol is missing")
    protocol = read_json(PROTOCOL)
    require(protocol.get("status") == "FROZEN", "Reduced protocol is not frozen")
    approval = {
        "schema_version": "reduced_routewise_bstar_utility_execution_approval_v1",
        "status": "APPROVED_FOR_EXECUTION",
        "approval_date": date.today().isoformat(),
        "approval_basis": args.approval_basis,
        "protocol_path": relative(PROTOCOL),
        "protocol_sha256": sha256_path(PROTOCOL),
        "execution_hashes": {relative(path): sha256_path(path)
                             for path in (RUNNER, FULL_RUNNER, COMMON)},
        "population_counts": protocol["population"]["counts"],
        "task_count_per_route": protocol["population"]["total_tasks_per_route"],
        "routes": protocol["routes"],
        "prohibitions": [
            "No utility-outcome-informed task or route selection",
            "No omission of any frozen B* route",
            "No change to generation limits, alpha, layers, features, or evaluator",
            "No full-BigCodeBench claim from the 150-task subset",
        ],
    }
    atomic_json(APPROVAL, approval)
    print(json.dumps({"status": approval["status"], "approval_path": relative(APPROVAL),
                      "approval_sha256": sha256_path(APPROVAL)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
