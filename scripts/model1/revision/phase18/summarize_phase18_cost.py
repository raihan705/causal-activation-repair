#!/usr/bin/env python3
"""Phase 18 cost summarizer with stage-correct RCI-1 failure accounting."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import summarize_phase18_cost_implementation as implementation


def structural_failure(method: str, row: dict[str, Any]) -> bool:
    if method != "RCI-1":
        return row["generation_status"] != "COMPLETED"
    stages = row.get("stage_records", [])
    return not (
        len(stages) == 3
        and [stage.get("stage_name") for stage in stages]
        == ["initial", "critique_1", "improve_1"]
        and all(stage.get("status") == "COMPLETED" for stage in stages)
    )


def aggregate(method: str, repetition: str | int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = implementation._uncorrected_aggregate(method, repetition, rows)
    result["top_level_exception_count"] = sum(
        row["generation_status"] != "COMPLETED" for row in rows
    )
    result["failure_count"] = sum(structural_failure(method, row) for row in rows)
    result["structurally_complete_count"] = sum(
        not structural_failure(method, row) for row in rows
    )
    return result


def enrich_summary() -> None:
    output = implementation.OUT / "phase18_cost_summary.json"
    summary = implementation.read_json(output)
    rci = implementation.read_json(implementation.RAW / "rci_1_timing.json")
    b3pg = implementation.read_json(implementation.RAW / "b3_pg_timing.json")

    rci_rows = rci["records"]
    skipped_omitted = sum(
        3 - len(row.get("stage_records", []))
        for row in rci_rows
        if structural_failure("RCI-1", row)
    )
    gate_steps = [
        step for row in b3pg["records"] for step in row.get("gate_log", [])
    ]
    summary["rci_failure_accounting"] = {
        "source": "stage_records",
        "top_level_status_ignored_for_structural_failure": True,
        "top_level_exception_count": sum(
            row["generation_status"] != "COMPLETED" for row in rci_rows
        ),
        "structurally_complete_count": sum(
            not structural_failure("RCI-1", row) for row in rci_rows
        ),
        "structural_failure_count": sum(
            structural_failure("RCI-1", row) for row in rci_rows
        ),
        "omitted_explicit_skipped_stage_record_count": skipped_omitted,
    }
    summary["b3pg_gate_cost"] = {
        "gate_step_count": len(gate_steps),
        "generated_tokens": sum(
            int(row["generated_tokens"]) for row in b3pg["records"]
        ),
        "model_calls": sum(
            int(row["model_call_count"]) for row in b3pg["records"]
        ),
        "mean_model_calls_per_generated_token": (
            sum(int(row["model_call_count"]) for row in b3pg["records"])
            / sum(int(row["generated_tokens"]) for row in b3pg["records"])
        ),
        "nan_kl_count": sum(step.get("kl") == "nan" for step in gate_steps),
        "alpha_zero_count": sum(float(step["alpha_used"]) == 0.0 for step in gate_steps),
        "preserved_gate_outcome": "ALL_KL_NAN_ALL_ALPHA_ZERO",
    }
    summary["instrumentation_notes"] = {
        "RCI-1": "Structural failures are derived from stage_records, not the top-level status.",
        "B1-CWE": "All active routes used the specified B1 fallback.",
        "B3-PG": "The observed NaN-KL/alpha-zero behavior is preserved without correction.",
    }
    implementation.atomic_json(output, summary)
    print({
        "status": "COMPLETE_STAGE_CORRECTED",
        "summary_sha256": implementation.sha256(output),
        "rci_structural_failures": summary["rci_failure_accounting"]["structural_failure_count"],
    })


if __name__ == "__main__":
    implementation._uncorrected_aggregate = implementation.aggregate
    implementation.aggregate = aggregate
    implementation.main()
    if "--preflight-only" not in sys.argv:
        enrich_summary()
