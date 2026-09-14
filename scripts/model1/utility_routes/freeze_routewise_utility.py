#!/usr/bin/env python3
"""Bind the approved route-wise protocol to the exact executable hashes."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from routewise_utility_common import (
    APPROVAL, PROTOCOL, SCRIPTS, atomic_json, load_protocol, relative, sha256_path,
)


RUNNER = SCRIPTS / "run_routewise_utility.py"
COMMON = SCRIPTS / "routewise_utility_common.py"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approval-basis", required=True)
    args = parser.parse_args()
    protocol = load_protocol()
    approval = {
        "schema_version": "routewise_bstar_utility_execution_approval_v1",
        "status": "APPROVED_FOR_EXECUTION",
        "approval_date": date.today().isoformat(),
        "approval_basis": args.approval_basis,
        "protocol_path": relative(PROTOCOL),
        "protocol_sha256": sha256_path(PROTOCOL),
        "execution_hashes": {
            relative(RUNNER): sha256_path(RUNNER),
            relative(COMMON): sha256_path(COMMON),
        },
        "routes": protocol["routes"],
        "authorized_stages": [
            "fresh paired B0 utility generation",
            "nine independent single-route B* utility generations",
            "official HumanEval evaluation in isolated Linux/Colab",
            "deterministic metric and uncertainty calculation",
        ],
        "prohibitions": [
            "No modification of Phase 10 or Phase 17 artifacts",
            "No artificial CWE assignment to utility tasks",
            "No multi-feature AlwaysOn intervention",
            "No route replacement, alpha change, prompt change, or pooled B* effect claim",
        ],
    }
    atomic_json(APPROVAL, approval)
    print(json.dumps({"status": "APPROVED_FOR_EXECUTION", "approval_path": relative(APPROVAL),
                      "approval_sha256": sha256_path(APPROVAL)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

