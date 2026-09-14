#!/usr/bin/env python3
"""Structurally validate Phase 16A outputs and build provenance/scan manifests.

This script never invokes the scanner and never computes security metrics.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from phase16_common import *

REUSED = [
    ("B0", 42, "REUSED", "METADATA_FREE", "outputs/phase9/baseline_test_outputs.json", "outputs/phase9/baseline_test_icd.json"),
    ("B1", 42, "REUSED", "METADATA_FREE", "outputs/phase9/zeroshot_test_outputs.json", "outputs/phase9/zeroshot_test_icd.json"),
    ("B*", 42, "REUSED", "ORACLE_CWE", "outputs/phase9/bstar_test_outputs.json", "outputs/phase9/bstar_test_icd.json"),
]
PRESERVED = ("B3-PG", 42, "PRESERVED_REFERENCE", "ORACLE_CWE", "outputs/phase9/semantic_static_test_outputs.json", "outputs/phase9/semantic_static_test_icd.json")


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def structural_counts(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "record_count": len(records),
        "unique_prompt_ids": len({int(row["prompt_id"]) for row in records}),
        "generation_failure_count": sum(row.get("generation_status") not in ("COMPLETED", "SUCCESS") for row in records),
        "strict_empty_count": sum(row.get("empty_status") == "STRICT_EMPTY" for row in records),
        "whitespace_only_count": sum(row.get("empty_status") == "WHITESPACE_ONLY" for row in records),
        "route_status_counts": dict(sorted(Counter(str(row.get("route_status")) for row in records).items())),
        "fallback_status_counts": dict(sorted(Counter(str(row.get("fallback_status")) for row in records).items())),
    }


def main() -> None:
    gate = phase15_gate()
    population = load_frozen_population()
    commands = read_json(COMMANDS)
    controller_path = OUTPUTS / "phase16_generation_controller_state.json"
    controller = read_json(controller_path)
    require(controller["status"] == "COMPLETE" and controller["failure"] is None, "controller is not complete/clean")
    require(len(controller["completed"]) == 13, "controller did not complete 13 conditions")
    freeze_time = parse_time(PHASE15_FREEZE_TIMESTAMP)
    validations = []
    new_manifest_rows = []
    scan_rows = []
    rci_summary = None
    b1_summary = None
    caa_summary = None
    for index, spec in enumerate(commands["generation_conditions"]):
        method, seed = spec["method"], int(spec["seed"])
        output = ROOT / spec["output_path"]
        run_manifest_path = ROOT / spec["run_manifest_path"]
        checkpoint_path = ROOT / spec["checkpoint_path"]
        scan_path = ROOT / spec["scanner_procedure"]["scan_output_path"]
        require(output.is_file() and run_manifest_path.is_file() and checkpoint_path.is_file(), f"missing files for {method} seed{seed}")
        require(not scan_path.exists(), f"scan exists before the specified scan stage: {method} seed{seed}")
        run_manifest = read_json(run_manifest_path)
        require(run_manifest["status"] == "COMPLETE", f"run manifest incomplete: {method} seed{seed}")
        require(run_manifest["output_sha256"] == sha256_path(output), f"output hash mismatch: {method} seed{seed}")
        require(run_manifest["condition"]["phase15_freeze_sha256"] == EXPECTED_HASHES["revision_freeze_manifest.json"], "freeze hash mismatch")
        require(run_manifest["condition"]["config_sha256"] == spec["config_sha256"], "config hash mismatch")
        require(parse_time(run_manifest["started_at_utc"]) > freeze_time and parse_time(run_manifest["completed_at_utc"]) > freeze_time, "generation predates freeze")
        body = read_json(output)
        records = body["final_outputs"] if method == "RCI-1" else body
        validate_record_order(records, population, method, seed)
        require(all(row["information_tier"] == spec["information_tier"] or (method == "B0" and row["information_tier"] == "NO_INTERVENTION_BASELINE") for row in records), f"information tier mismatch: {method}")
        require(all(row["population_sha256"] == population["population_sha256"] for row in records), f"population hash mismatch: {method}")
        counts = structural_counts(records)
        require(counts["record_count"] == EXPECTED_COUNT and counts["unique_prompt_ids"] == EXPECTED_COUNT, "coverage mismatch")
        validation = {
            "order_index": index, "method": method, "seed": seed, "status": "PASS", **counts,
            "information_tier": spec["information_tier"], "config_sha256": spec["config_sha256"],
            "output_path": relative(output), "output_sha256": sha256_path(output),
            "checkpoint_path": relative(checkpoint_path), "checkpoint_sha256": sha256_path(checkpoint_path),
            "run_manifest_path": relative(run_manifest_path), "run_manifest_sha256": sha256_path(run_manifest_path),
            "completion_resume_status": run_manifest["completion_resume_status"], "resume_event_count": len(run_manifest.get("resume_events", [])),
            "started_at_utc": run_manifest["started_at_utc"], "completed_at_utc": run_manifest["completed_at_utc"],
            "elapsed_seconds": run_manifest.get("elapsed_seconds"), "generated_token_count": run_manifest.get("generated_token_count"),
            "scan_status": "NOT_STARTED", "scan_output_absent": True,
        }
        validations.append(validation)
        new_manifest_rows.append({
            "disposition": "NEW", "method": method, "seed": seed,
            "generation_path": relative(output), "generation_sha256": sha256_path(output),
            "run_manifest_path": relative(run_manifest_path), "run_manifest_sha256": sha256_path(run_manifest_path),
            "information_tier": spec["information_tier"], "population_count": EXPECTED_COUNT,
            "provenance_status": "NEW_REVISION_GENERATION_AFTER_PHASE15_FREEZE",
            "scan_status": "NOT_STARTED",
        })
        scan_rows.append({
            "order_index": index, "method": method, "seed": seed,
            "information_tier": spec["information_tier"],
            "input_path": relative(output), "input_sha256": sha256_path(output),
            "run_manifest_path": relative(run_manifest_path), "run_manifest_sha256": sha256_path(run_manifest_path),
            "record_count": EXPECTED_COUNT, "expected_prompt_ids_sha256": population["prompt_ids_sha256"],
            "expected_prompt_order_source": "revision/model1/phase2/phase2_denominator_audit.json heldout.total_prompt_ids",
            "scanner_path": "phases/phase9/colab_scan_phase9.py", "scanner_sha256": EXPECTED_HASHES["scanner.py"],
            "scan_output_path": spec["scanner_procedure"]["scan_output_path"], "scan_status": "NOT_STARTED",
            "adapter": "RCI_FINAL_OUTPUTS_TO_FROZEN_SCANNER_SCHEMA" if method == "RCI-1" else "DIRECT_GENERATION_RECORDS",
        })
        if method == "RCI-1":
            stages = body["stage_records"]
            failure_by_stage = defaultdict(int)
            for row in stages:
                if row["completion_status"] != "COMPLETED": failure_by_stage[row["stage_name"]] += 1
            rci_summary = {
                "final_output_count": len(records), "total_stage_rows": len(stages),
                "total_model_calls": sum(row["model_call_index"] is not None for row in stages),
                "failure_counts_by_stage": dict(sorted(failure_by_stage.items())),
                "input_tokens": sum(int(row["stage_input_token_count"]) for row in stages),
                "generated_tokens": sum(int(row["generated_token_count"]) for row in stages),
                "stage_elapsed_seconds": sum(float(row["elapsed_seconds"]) for row in stages),
                "cost_path": "revision/model1/phase16/outputs/rci_1_seed42_cost.csv",
                "cost_sha256": sha256_path(OUTPUTS / "rci_1_seed42_cost.csv"),
            }
        elif method == "B1-CWE":
            b1_summary = {"route_status_counts": counts["route_status_counts"], "fallback_status_counts": counts["fallback_status_counts"], "guidance_available_true": sum(bool(row.get("guidance_available")) for row in records)}
        elif method == "CAA-CWE":
            caa_summary = {"route_status_counts": counts["route_status_counts"], "fallback_status_counts": counts["fallback_status_counts"], "intervention_applied": sum(bool(row.get("intervention_applied")) for row in records), "no_intervention": sum(not bool(row.get("intervention_applied")) for row in records)}

    reuse = read_json(OUTPUTS / "phase16_reuse_verification.json")["artifacts"]
    generation_rows = []
    for method, seed, disposition, tier, generation_text, scan_text in REUSED:
        row = next(item for item in reuse if item["method"] == method)
        generation_rows.append({"disposition": disposition, "method": method, "seed": seed, "generation_path": generation_text, "generation_sha256": row["generation_sha256"], "run_manifest_path": None, "run_manifest_sha256": None, "information_tier": tier, "population_count": EXPECTED_COUNT, "provenance_status": row["provenance_status"], "scan_status": "REUSE_EXISTING", "existing_scan_path": scan_text, "existing_scan_sha256": row["existing_scan_sha256"]})
    generation_rows.extend(new_manifest_rows)
    method, seed, disposition, tier, generation_text, scan_text = PRESERVED
    row = next(item for item in reuse if item["method"] == method)
    generation_rows.append({"disposition": disposition, "method": method, "seed": seed, "generation_path": generation_text, "generation_sha256": row["generation_sha256"], "run_manifest_path": None, "run_manifest_sha256": None, "information_tier": tier, "population_count": EXPECTED_COUNT, "provenance_status": row["provenance_status"], "scan_status": "REUSE_EXISTING", "existing_scan_path": scan_text, "existing_scan_sha256": row["existing_scan_sha256"]})

    validation_artifact = {
        "schema_version": "phase16_generation_structural_validation_v1", "status": "PASS",
        "validated_at_utc": utc_now(), "phase15_gate": gate, "condition_count": 13,
        "all_outputs_after_phase15_freeze": True, "all_scan_outputs_absent": True,
        "security_metrics_computed": False, "scanner_executed": False,
        "conditions": validations, "rci_structural_summary": rci_summary,
        "b1cwe_routing_summary": b1_summary, "caa_routing_summary": caa_summary,
    }
    atomic_json(OUTPUTS / "phase16_generation_structural_validation.json", validation_artifact)
    generation_manifest = {
        "schema_version": "phase16_generation_manifest_v1", "status": "HELDOUT_GENERATION_COMPLETE_AWAITING_SCANS",
        "phase15_freeze_sha256": EXPECTED_HASHES["revision_freeze_manifest.json"],
        "phase15_freeze_timestamp": PHASE15_FREEZE_TIMESTAMP,
        "first_heldout_access_timestamp": FIRST_ACCESS_TIMESTAMP,
        "population_count": EXPECTED_COUNT, "prompt_ids_sha256": population["prompt_ids_sha256"],
        "population_sha256": population["population_sha256"], "condition_count": len(generation_rows),
        "reused_count": 3, "new_count": 13, "preserved_submitted_count": 1,
        "conditions": generation_rows, "scanner_executed": False, "security_metrics_computed": False,
    }
    atomic_json(OUTPUTS / "phase16_generation_manifest.json", generation_manifest)
    scan_manifest = {
        "schema_version": "phase16_scan_transfer_manifest_v1", "status": "READY_NOT_STARTED",
        "phase15_freeze_sha256": EXPECTED_HASHES["revision_freeze_manifest.json"],
        "scanner_path": "phases/phase9/colab_scan_phase9.py", "scanner_sha256": EXPECTED_HASHES["scanner.py"],
        "population_count": EXPECTED_COUNT, "expected_prompt_ids": population["prompt_ids"],
        "expected_prompt_ids_sha256": population["prompt_ids_sha256"], "entry_count": len(scan_rows),
        "not_started_count": sum(row["scan_status"] == "NOT_STARTED" for row in scan_rows),
        "entries": scan_rows, "scanner_executed": False,
    }
    atomic_json(OUTPUTS / "phase16_scan_transfer_manifest.json", scan_manifest)
    checkpoint = {
        "schema_version": "phase16a_checkpoint_v1", "phase": 16,
        "phase_status": "IN_PROGRESS", "checkpoint": "PENDING", "intermediate_state": "HELDOUT_GENERATION_COMPLETE_AWAITING_SCANS",
        "generation_condition_count": 13, "generation_structural_validation": "PASS",
        "generation_manifest_sha256": sha256_path(OUTPUTS / "phase16_generation_manifest.json"),
        "scan_transfer_manifest_sha256": sha256_path(OUTPUTS / "phase16_scan_transfer_manifest.json"),
        "scan_entries_not_started": 13, "scanner_executed": False, "security_metrics_computed": False,
        "phase17_status": "NOT_STARTED/PENDING", "blockers": [],
    }
    atomic_json(OUTPUTS / "phase16a_checkpoint.json", checkpoint)
    print(json.dumps({"status": "PASS", "conditions": validations, "rci": rci_summary, "b1cwe": b1_summary, "caa": caa_summary,
                      "generation_manifest_sha256": sha256_path(OUTPUTS / "phase16_generation_manifest.json"),
                      "scan_transfer_manifest_sha256": sha256_path(OUTPUTS / "phase16_scan_transfer_manifest.json")}, indent=2))

if __name__ == "__main__":
    main()
