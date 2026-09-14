#!/usr/bin/env python3
"""Freeze authoritative CAA Stage B selection provenance before outcomes exist."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
PHASE14 = ROOT / "revision" / "model1" / "phase14"
OUTPUTS = PHASE14 / "outputs"
SELECTOR = PHASE14 / "scripts" / "select_caa_configuration.py"
PLAN = ROOT / "revision" / "REVISION_PLAN.md"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    outcome_patterns = ("caa_stage_b_layer16_mult0p5_seed42.json",
                        "caa_stage_b_layer16_mult2p0_seed42.json",
                        "caa_stage_b_layer16_mult4p0_seed42.json")
    existing = [name for name in outcome_patterns if (OUTPUTS / name).exists()]
    if existing:
        raise RuntimeError(f"Stage B outcomes already exist before selector freeze: {existing}")

    selector_text = SELECTOR.read_text(encoding="utf-8")
    required_fragments = (
        'float(row["corruption"]) <= 0.05',
        'float(row["validity"]) >= 0.95',
        '-float(row["corrvrr"])',
        'float(row["corruption"])',
        '-float(row["validity"])',
        'float(row["multiplier"])',
    )
    missing = [fragment for fragment in required_fragments if fragment not in selector_text]
    if missing:
        raise RuntimeError(f"Selector does not implement authoritative Stage B rule: {missing}")

    result = {
        "schema_version": "phase14_caa_stage_b_selector_freeze_v1",
        "status": "FROZEN_BEFORE_STAGE_B_OUTCOME_INSPECTION",
        "selected_layer": 16,
        "candidate_multipliers": [0.5, 1.0, 2.0, 4.0],
        "seed": 42,
        "admissibility_gates": {"corruption_rate_max_inclusive": 0.05, "validity_rate_min_inclusive": 0.95},
        "ranking_rule": ["highest CorrVRR", "lower corruption", "higher validity", "smaller multiplier"],
        "authoritative_tie_rule": "Use the smaller multiplier when performance ties.",
        "performance_tie_interpretation": "Equal CorrVRR, corruption, and validity after applying the admissibility gates.",
        "authoritative_source": {
            "path": PLAN.relative_to(ROOT).as_posix(),
            "sha256": sha256_path(PLAN),
            "reference": "Section D, Phase 14, Procedure 14.4 step 8",
        },
        "selector": {
            "path": SELECTOR.relative_to(ROOT).as_posix(),
            "sha256": sha256_path(SELECTOR),
            "reconciled_with_authoritative_rule": True,
            "scientific_selection_change_required": False,
        },
        "stage_a_selection": {
            "path": "revision/model1/phase14/outputs/caa_stage_a_selection.json",
            "sha256": sha256_path(OUTPUTS / "caa_stage_a_selection.json"),
            "selected_layer": 16,
        },
        "stage_a": {
            "selected_layer": 16,
            "compatibility_purpose": "Nested read-only projection required by the frozen CAA runner preflight; value is copied from the unchanged reviewed Stage-A selection artifact.",
        },
        "runner_compatibility": {
            "frozen_runner_expected_field": "stage_a.selected_layer",
            "reviewed_stage_a_artifact_field": "selected_layer",
            "scientific_value_changed": False,
            "reviewed_stage_a_artifact_modified": False,
        },
        "stage_b_outcome_files_present_at_freeze": existing,
        "stage_b_outcomes_inspected": False,
        "held_out_accessed": False,
        "provenance_resolution": "Section D governs. The prior warning concerned Stage A's final layer tie only; the existing Stage B selector already exactly implements Section D's smaller-multiplier performance-tie rule.",
    }
    output = OUTPUTS / "caa_stage_b_selector_freeze.json"
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)


if __name__ == "__main__":
    main()
