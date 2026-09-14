#!/usr/bin/env python3
"""Select future CAA Stage A/B configurations under the frozen gates."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from phase14_common import CAA_LAYERS, CAA_MULTIPLIERS, atomic_json, read_json, require


def admissible(row: dict[str, Any]) -> bool:
    return float(row["corruption"]) <= 0.05 and float(row["validity"]) >= 0.95


def select_stage_a(rows: list[dict[str, Any]]) -> dict[str, Any]:
    require({int(row["layer"]) for row in rows} == set(CAA_LAYERS), "Stage A must contain layers 16/19/23")
    require(all(float(row["multiplier"]) == 1.0 and int(row["seed"]) == 42 for row in rows),
            "Stage A multiplier/seed mismatch")
    candidates = [row for row in rows if admissible(row)]
    if not candidates:
        return {"status": "NO_ADMISSIBLE_CONFIGURATION", "selected_layer": None,
                "stage_b_selected": False}
    selected = sorted(candidates, key=lambda row: (-float(row["corrvrr"]),
                                                    float(row["corruption"]),
                                                    -float(row["validity"]),
                                                    int(row["layer"])))[0]
    return {"status": "SELECTED", "selected_layer": int(selected["layer"]),
            "stage_b_selected": True, "selected_row": selected}


def select_stage_b(rows: list[dict[str, Any]], selected_layer: int) -> dict[str, Any]:
    require({float(row["multiplier"]) for row in rows} == set(CAA_MULTIPLIERS),
            "Stage B must contain multipliers 0.5/1/2/4")
    require(all(int(row["layer"]) == selected_layer and int(row["seed"]) == 42 for row in rows),
            "Stage B selected-layer/seed mismatch")
    candidates = [row for row in rows if admissible(row)]
    if not candidates:
        return {"status": "NO_ADMISSIBLE_CONFIGURATION", "selected_multiplier": None}
    selected = sorted(candidates, key=lambda row: (-float(row["corrvrr"]),
                                                    float(row["corruption"]),
                                                    -float(row["validity"]),
                                                    float(row["multiplier"])))[0]
    return {"status": "SELECTED", "selected_multiplier": float(selected["multiplier"]),
            "selected_row": selected}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-a", type=Path, required=True)
    parser.add_argument("--stage-b", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stage_a = select_stage_a(read_json(args.stage_a.resolve()))
    result = {"schema_version": "phase14_caa_selection_v1", "admissibility": {
        "corruption_max": 0.05, "validity_min": 0.95},
        "stage_a_tie_break": ["highest CorrVRR", "lower corruption", "higher validity", "lowest layer ID"],
        "stage_b_tie_break": ["highest CorrVRR", "lower corruption", "higher validity", "smaller multiplier"],
        "stage_a": stage_a, "stage_b": None}
    if stage_a["stage_b_selected"]:
        require(args.stage_b is not None, "Stage B evidence required after admissible Stage A")
        result["stage_b"] = select_stage_b(read_json(args.stage_b.resolve()), stage_a["selected_layer"])
        result["status"] = result["stage_b"]["status"]
    else:
        require(args.stage_b is None, "Stage B must not run when Stage A has no admissible layer")
        result["status"] = "NO_ADMISSIBLE_CONFIGURATION"
    atomic_json(args.output.resolve(), result)


if __name__ == "__main__":
    main()
