#!/usr/bin/env python3
"""Validate and freeze the three new CAA Stage B generation conditions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
OUTPUTS = ROOT / "revision" / "model1" / "phase14" / "outputs"
SUBSET = ROOT / "revision" / "model1" / "phase13" / "outputs" / "baseline_selection_subset.json"
DENOMINATORS = ROOT / "revision" / "model1" / "phase2" / "phase2_denominator_audit.json"
VECTOR_MANIFEST = OUTPUTS / "caa_vector_manifest.json"
RUNNER = ROOT / "revision" / "model1" / "phase14" / "scripts" / "run_caa_cwe_dev.py"
WRAPPER = RUNNER.with_name("caa_position_mask.py")

CONDITIONS = [
    (0.5, "caa_stage_b_layer16_mult0p5_seed42"),
    (2.0, "caa_stage_b_layer16_mult2p0_seed42"),
    (4.0, "caa_stage_b_layer16_mult4p0_seed42"),
]


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    subset = read_json(SUBSET)
    expected_ids = [int(value) for value in subset["prompt_ids_source_order"]]
    expected_records = subset["records"]
    held_out_ids = set(int(value) for value in read_json(DENOMINATORS)["heldout"]["total_prompt_ids"])
    vector_hash = sha256_path(VECTOR_MANIFEST)
    runner_hash = sha256_path(RUNNER)
    wrapper_hash = sha256_path(WRAPPER)
    rows = []
    errors: list[str] = []

    for multiplier, stem in CONDITIONS:
        generation_path = OUTPUTS / f"{stem}.json"
        checkpoint_path = OUTPUTS / f"{stem}_checkpoint.json"
        run_path = OUTPUTS / f"{stem}_run_manifest.json"
        records = read_json(generation_path)
        run = read_json(run_path)
        ids = [int(record["prompt_id"]) for record in records]
        active = sum(bool(record.get("intervention_applied")) for record in records)
        inactive = len(records) - active
        failures = sum(record.get("generation_status") != "COMPLETED" for record in records)
        strict_empty = sum(record.get("generated_code") == "" for record in records)
        whitespace = sum(bool(record.get("generated_code")) and not record["generated_code"].strip() for record in records)
        local_errors = []
        if len(records) != 120 or len(set(ids)) != 120 or ids != expected_ids:
            local_errors.append("record count/uniqueness/source-order mismatch")
        if set(ids) & held_out_ids:
            local_errors.append("held-out prompt ID present")
        if [record["source_index"] for record in records] != [record["source_index"] for record in expected_records]:
            local_errors.append("source-index order mismatch")
        if [record["prompt_text_sha256"] for record in records] != [record["prompt_text_sha256"] for record in expected_records]:
            local_errors.append("prompt-text identity mismatch")
        if active != 90 or inactive != 30:
            local_errors.append("90-active/30-no-intervention routing mismatch")
        for record in records:
            condition = record.get("condition", {})
            expected_active = record["target_cwe"] in {"CWE-120", "CWE-327", "CWE-89", "CWE-338"}
            if (record.get("method"), record.get("information_tier"), record.get("seed")) != ("CAA-CWE", "ORACLE_CWE", 42):
                local_errors.append(f"prompt {record['prompt_id']} method/tier/seed mismatch")
                break
            if (condition.get("stage"), condition.get("layer"), condition.get("multiplier")) != ("B", 16, multiplier):
                local_errors.append(f"prompt {record['prompt_id']} condition mismatch")
                break
            if bool(record.get("intervention_applied")) != expected_active:
                local_errors.append(f"prompt {record['prompt_id']} route mismatch")
                break
            if condition.get("vector_manifest_sha256") != vector_hash or condition.get("runner_sha256") != runner_hash or condition.get("wrapper_sha256") != wrapper_hash:
                local_errors.append(f"prompt {record['prompt_id']} frozen hash mismatch")
                break
        if run.get("status") != "COMPLETE" or int(run.get("completed_records", 0)) != 120:
            local_errors.append("run manifest incomplete")
        if run.get("output_sha256") != sha256_path(generation_path):
            local_errors.append("run/output hash linkage mismatch")
        errors.extend(f"M{multiplier}: {error}" for error in local_errors)
        rows.append({
            "multiplier": multiplier,
            "status": "PASS" if not local_errors else "FAIL",
            "generation_path": generation_path.relative_to(ROOT).as_posix(),
            "generation_sha256": sha256_path(generation_path),
            "checkpoint_path": checkpoint_path.relative_to(ROOT).as_posix(),
            "checkpoint_sha256": sha256_path(checkpoint_path),
            "run_manifest_path": run_path.relative_to(ROOT).as_posix(),
            "run_manifest_sha256": sha256_path(run_path),
            "record_count": len(records),
            "unique_prompt_id_count": len(set(ids)),
            "source_order_exact": ids == expected_ids,
            "held_out_overlap_count": len(set(ids) & held_out_ids),
            "intervention_active_count": active,
            "no_intervention_count": inactive,
            "generation_failure_count": failures,
            "strict_empty_count": strict_empty,
            "whitespace_only_count": whitespace,
            "errors": local_errors,
        })

    result = {
        "schema_version": "phase14_caa_stage_b_generation_validation_v1",
        "status": "PASS" if not errors else "FAIL",
        "subset_path": SUBSET.relative_to(ROOT).as_posix(),
        "subset_sha256": sha256_path(SUBSET),
        "record_count": 120,
        "selected_layer": 16,
        "seed": 42,
        "vector_manifest_sha256": vector_hash,
        "runner_sha256": runner_hash,
        "position_mask_wrapper_sha256": wrapper_hash,
        "conditions": rows,
        "held_out_accessed": False,
        "errors": errors,
    }
    output = OUTPUTS / "caa_stage_b_generation_structural_validation.json"
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    if errors:
        raise SystemExit("; ".join(errors))


if __name__ == "__main__":
    main()
