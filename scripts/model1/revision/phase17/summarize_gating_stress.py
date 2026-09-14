#!/usr/bin/env python3
"""Prepare the Phase 17 scan handoff and summarize active gating stress results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from phase17_common import OUTPUTS, ROOT, SCANNER, atomic_csv, atomic_json, canonical_sha256, load_prepared_manifest, read_json, require, sha256_path


GENERATION = OUTPUTS / "alwayson_security_seed42_outputs.json"
SCANNER_INPUT = OUTPUTS / "alwayson_security_seed42_scanner_input.json"
SCAN = OUTPUTS / "alwayson_security_seed42_icd.json"
TRANSFER = OUTPUTS / "phase17_scan_transfer_manifest.json"
SECURITY = OUTPUTS / "phase17_security_results.csv"
UTILITY = OUTPUTS / "phase17_utility_results.csv"
INTERVALS = OUTPUTS / "phase17_utility_intervals.csv"
SUMMARY = OUTPUTS / "phase17_gating_stress_summary.json"


def prepare_scan(manifest: dict[str, Any], preparation: dict[str, Any]) -> dict[str, Any]:
    require(GENERATION.is_file(), "Active security generation is missing")
    rows = read_json(GENERATION)
    require(isinstance(rows, list) and len(rows) == 575, "Active security generation count mismatch")
    metadata = {int(row["prompt_id"]): row for row in manifest["populations"]["heldout"]["scanner_metadata"]}
    adapted = []
    for row in rows:
        prompt_id = int(row["prompt_id"])
        require(prompt_id in metadata, f"Missing scanner metadata for prompt {prompt_id}")
        adapted.append({**row, "cwe_id": metadata[prompt_id]["target_cwe"],
                        "language": metadata[prompt_id]["language"]})
    atomic_json(SCANNER_INPUT, adapted)
    transfer = {"schema_version": "phase17_scan_transfer_manifest_v1",
                "status": "READY_NOT_STARTED", "entry_count": 1,
                "method": "B*-AlwaysOn-L19", "seed": 42,
                "information_tier": "METADATA_FREE", "record_count": 575,
                "expected_prompt_ids": [row["prompt_id"] for row in adapted],
                "expected_prompt_ids_sha256": canonical_sha256([row["prompt_id"] for row in adapted]),
                "source_generation_path": str(GENERATION.relative_to(ROOT)).replace("\\", "/"),
                "source_generation_sha256": sha256_path(GENERATION),
                "scanner_input_path": str(SCANNER_INPUT.relative_to(ROOT)).replace("\\", "/"),
                "scanner_input_sha256": sha256_path(SCANNER_INPUT),
                "scanner_path": str(SCANNER.relative_to(ROOT)).replace("\\", "/"),
                "scanner_sha256": sha256_path(SCANNER),
                "scan_output_path": str(SCAN.relative_to(ROOT)).replace("\\", "/"),
                "input_manifest_sha256": preparation["input_manifest_sha256"],
                "generation_runner_consumed_cwe_metadata": False,
                "adapter_role": "POST_GENERATION_SCANNER_METADATA_JOIN_ONLY"}
    atomic_json(TRANSFER, transfer)
    return transfer


def summarize(manifest: dict[str, Any], preparation: dict[str, Any]) -> dict[str, Any]:
    for path in (GENERATION, SCAN, UTILITY, INTERVALS):
        require(path.is_file(), f"Missing required Phase 17 result: {path}")
    generated = {int(row["prompt_id"]): row for row in read_json(GENERATION)}
    scan = {int(row["prompt_id"]): row for row in read_json(SCAN)}
    b0 = {int(row["prompt_id"]): row for row in read_json(ROOT / "outputs/phase9/baseline_test_icd.json")}
    b0_generation = {int(row["prompt_id"]): row for row in read_json(
        ROOT / "outputs/phase9/baseline_test_outputs.json")}
    universe = set(map(int, read_json(ROOT / "revision/model1/phase2/phase2_denominator_audit.json")["heldout"]["scanner_eligible_prompt_ids"]))
    require(set(generated) == set(scan) == set(b0) == set(b0_generation)
            and len(generated) == 575, "Security population mismatch")
    vulnerable = {prompt_id for prompt_id in universe if b0[prompt_id]["is_vulnerable"] is True}
    safe = universe - vulnerable
    valid = {prompt_id for prompt_id, row in generated.items() if str(row.get("generated_code", "")).strip()}
    repaired = {prompt_id for prompt_id in vulnerable if prompt_id in valid and not scan[prompt_id]["skipped"] and not scan[prompt_id]["is_vulnerable"]}
    corrupted = {prompt_id for prompt_id in safe if not scan[prompt_id]["skipped"] and scan[prompt_id]["is_vulnerable"]}
    active = {"condition": "B*-AlwaysOn-L19", "information_tier": "METADATA_FREE",
              "seed": 42, "total_prompts": 575, "scanner_universe_count": len(universe),
              "scanner_eligible_count": sum(not row["skipped"] for row in scan.values()),
              "scanner_skipped_count": sum(row["skipped"] for row in scan.values()),
              "b0_vulnerable_denominator": len(vulnerable), "repair_count": len(repaired),
              "corrvrr": f"{len(repaired)/len(vulnerable):.6f}",
              "corrected_vulnerable_count": len(vulnerable)-len(repaired),
              "raw_vulnerable_count": sum(row["is_vulnerable"] for row in scan.values()),
              "raw_vrr": f"{1-sum(row['is_vulnerable'] for row in scan.values())/len(vulnerable):.6f}",
              "b0_safe_denominator": len(safe), "corruption_count": len(corrupted),
              "corruption_rate": f"{len(corrupted)/len(safe):.6f}",
              "validity_count": len(valid), "validity_rate": f"{len(valid)/575:.6f}",
              "strict_empty_count": sum(row.get("empty_status") == "STRICT_EMPTY" for row in generated.values()),
              "whitespace_only_count": sum(row.get("empty_status") == "WHITESPACE_ONLY" for row in generated.values()),
              "generation_failure_count": sum(row.get("generation_status") != "COMPLETED" for row in generated.values()),
              "zero_hook_call_count": sum(int(row.get("hook_evidence", {}).get("hook_call_count", 0)) == 0
                                          for row in generated.values()),
              "generation_sha256": sha256_path(GENERATION), "scan_sha256": sha256_path(SCAN)}
    b0_validity = sum(bool(str(row.get("generated_code", "")).strip()) for row in b0_generation.values())
    b0_strict_empty = sum(row.get("generated_code", "") == "" for row in b0_generation.values())
    b0_whitespace = sum(row.get("generated_code", "") != "" and not row.get("generated_code", "").strip()
                        for row in b0_generation.values())
    b0_row = {"condition": "B0", "information_tier": "NO_INTERVENTION_BASELINE", "seed": 42,
              "total_prompts": 575, "scanner_universe_count": len(universe),
              "scanner_eligible_count": len(universe), "scanner_skipped_count": 575-len(universe),
              "b0_vulnerable_denominator": len(vulnerable), "repair_count": 0, "corrvrr": "0.000000",
              "corrected_vulnerable_count": len(vulnerable),
              "raw_vulnerable_count": len(vulnerable), "b0_safe_denominator": len(safe),
              "raw_vrr": "0.000000",
              "corruption_count": 0, "corruption_rate": "0.000000",
              "validity_count": b0_validity, "validity_rate": f"{b0_validity/575:.6f}",
              "strict_empty_count": b0_strict_empty, "whitespace_only_count": b0_whitespace,
              "generation_failure_count": "NOT_AVAILABLE",
              "zero_hook_call_count": "NOT_APPLICABLE",
              "generation_sha256": sha256_path(ROOT / "outputs/phase9/baseline_test_outputs.json"),
              "scan_sha256": sha256_path(ROOT / "outputs/phase9/baseline_test_icd.json")}
    atomic_csv(SECURITY, list(b0_row), [b0_row, active])
    utility_rows = list(csv.DictReader(UTILITY.open(encoding="utf-8", newline="")))
    interval_rows = list(csv.DictReader(INTERVALS.open(encoding="utf-8", newline="")))
    summary = {"schema_version": "phase17_gating_stress_summary_v1", "status": "COMPLETE",
               "inactive_gated": {"record_count": 1716, "steered_true": 0,
                                  "corrected_intervention_delta": 0,
                                  "interpretation": "EXACT_PAIRED_B0_REFERENCE_NOT_SEPARATE_GENERATION"},
               "active_security": {**active, "repaired_ids": sorted(repaired), "corrupted_ids": sorted(corrupted)},
               "active_utility": utility_rows, "utility_intervals": interval_rows,
               "input_manifest_sha256": preparation["input_manifest_sha256"],
               "artifacts": {"security_results_sha256": sha256_path(SECURITY),
                             "utility_results_sha256": sha256_path(UTILITY),
                             "utility_intervals_sha256": sha256_path(INTERVALS)}}
    atomic_json(SUMMARY, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--prepare-scan", action="store_true")
    args = parser.parse_args()
    manifest, preparation = load_prepared_manifest()
    if args.preflight_only:
        required = [GENERATION, SCAN, UTILITY, INTERVALS]
        print(json.dumps({"status": ("READY_FOR_SUMMARY" if all(path.is_file() for path in required)
                                     else "PASS_AWAITING_GENERATION_SCAN_AND_EVALUATION"),
                          "summary_run": False,
                          "present": {path.name: path.is_file() for path in required}}, indent=2))
        return 0
    if args.prepare_scan:
        print(json.dumps(prepare_scan(manifest, preparation), indent=2, sort_keys=True))
        return 0
    result = summarize(manifest, preparation)
    print(json.dumps({"status": result["status"], "summary_sha256": sha256_path(SUMMARY)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
