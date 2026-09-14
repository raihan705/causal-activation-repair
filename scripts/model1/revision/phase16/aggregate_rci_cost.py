#!/usr/bin/env python3
"""Aggregate the frozen executed RCI-1 held-out cost record."""

from __future__ import annotations

import csv
from collections import Counter

from phase16_analysis_common import OUT, load_all, sha256_path, write_csv


def main() -> int:
    load_all()
    source = OUT / "rci_1_seed42_cost.csv"
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    statuses = Counter(row["completion_status"] for row in rows)
    stages = Counter(row["stage_name"] for row in rows)
    result = [{
        "method": "RCI-1", "seed": 42, "seed_status": "SINGLE_SEED", "execution_status": "COMPLETE",
        "final_output_count": 575, "completed_final_output_count": 424,
        "failed_final_output_count": 151, "stage_row_count": len(rows),
        "model_call_count": sum(bool(row["model_call_index"]) for row in rows),
        "input_token_count": sum(int(row["stage_input_token_count"]) for row in rows),
        "generated_token_count": sum(int(row["generated_token_count"]) for row in rows),
        "elapsed_seconds": f"{sum(float(row['elapsed_seconds']) for row in rows):.6f}",
        "initial_stage_rows": stages["initial"], "critique_stage_rows": stages["critique_1"],
        "improvement_stage_rows": stages["improve_1"],
        "completed_stage_rows": statuses["COMPLETED"],
        "extraction_failure_stage_rows": statuses["EXTRACTION_FAILED_INVALID_OR_EMPTY"],
        "prerequisite_skipped_stage_rows": statuses["SKIPPED_PREREQUISITE_FAILURE"],
        "source_sha256": sha256_path(source), "interpretation": "FAITHFUL_NEGATIVE_RESULT",
    }]
    write_csv(OUT / "phase16_rci_cost.csv", list(result[0]), result)
    print("COMPLETE: phase16_rci_cost.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
