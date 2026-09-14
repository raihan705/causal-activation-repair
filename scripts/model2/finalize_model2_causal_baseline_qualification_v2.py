#!/usr/bin/env python
"""Validate warning-aware v2 scan and freeze the versioned causal denominator."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "revision/model2/phase21/outputs"
GEN = OUT / "model2_causal_baseline_generations.json"
GEN_MANIFEST = OUT / "model2_causal_baseline_generation_manifest.json"
AMENDMENT = OUT / "model2_causal_baseline_scanner_warning_amendment_v2.json"
SCAN = OUT / "model2_causal_baseline_scans_v2.json"
RETURN = OUT / "model2_causal_baseline_scan_return_manifest_v2.json"
QUAL = OUT / "model2_causal_baseline_qualification_v2.csv"
DENOM = OUT / "model2_causal_denominator_manifest_v2.json"
CHECKPOINT = OUT / "model2_causal_baseline_checkpoint_v2.json"
TARGETS = ["CWE-120", "CWE-327", "CWE-89"]
EXPECTED = {
    GEN: "3940c731ef49b68409fd168a0d1afadc100146240fba598ac80aa91939fe78cc",
    GEN_MANIFEST: "6f69b231115251a4a6033e0949b63c61e4ecd831e011cd63c13a1e6c8133bd1c",
    AMENDMENT: "68805604c99bc13126d2cc9b0b933f33746a53a678448d19ee77729424a29ac3",
    OUT / "model2_causal_baseline_scans.json": "e3ed20e2f53072eece94edeae9fdc008eb4142c49d83d57b57303e4e75de261b",
    OUT / "model2_causal_baseline_scan_return_manifest.json": "eaaeb02d7661a23a748b0447089186d86daaa8b9492be7d76c225b0376f84f86",
    OUT / "model2_causal_baseline_qualification.csv": "b13b54797bfcc0c88cf3b749ead8088c7b4c009390aec13182256a98632f7aac",
    OUT / "model2_causal_denominator_manifest.json": "e55e81dcbe4a8beaa78d77cb2eee3c5dca849b17fe7d19619b03ce9ff3f9d7a6",
    OUT / "model2_causal_baseline_checkpoint.json": "da1f5340c34796ad9ec6e907a4c11a8e0029f3f78cac13751ed3f6ea3866a735",
    OUT / "model2_causal_candidate_manifest_model1_fidelity.json": "be072afa3a9c58c20729bcd5deb343007f15b7c981e4fe02bb857bc17d00b401",
    OUT / "model2_causal_validation_protocol.json": "b3aac3b21f52270c231e1c077adeb0cc1ef2a2ebce1f156c08a13df2695d414e",
    OUT / "model2_causal_protocol_checkpoint.json": "099759f3b128ca8d3079d81c71fe24fd65be115ec0e3b270e0496b95357e1f33",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main() -> int:
    for path, digest in EXPECTED.items():
        require(path.is_file() and sha256_file(path) == digest, f"missing/drifted preserved artifact: {path.name}")
    for path in [SCAN, RETURN]:
        require(path.is_file(), f"missing v2 return artifact: {path.name}")
    for path in [QUAL, DENOM, CHECKPOINT]:
        require(not path.exists(), f"immutable v2 output already exists: {path.name}")

    gen_hash = sha256_file(GEN)
    scan_hash = sha256_file(SCAN)
    scan = json.loads(SCAN.read_text(encoding="utf-8"))
    returned = json.loads(RETURN.read_text(encoding="utf-8"))
    require(returned["scan_output"]["sha256"] == scan_hash, "v2 return/scan hash mismatch")
    require(returned["generation_input"]["sha256"] == gen_hash == scan["generation_input_sha256"], "v2 generation binding mismatch")
    require(scan["warning_amendment_sha256"] == sha256_file(AMENDMENT) == returned["warning_amendment_sha256"], "warning amendment binding mismatch")
    require(scan["transfer_manifest_sha256"] == returned["transfer_manifest_sha256"] == "0c3f60f9dc0f3bc050a99e2e8310e1edf020b4f431ff65cfe16bec704fdab432", "v2 transfer binding mismatch")
    require(scan["semgrep_version"] == returned["semgrep_version"] == "1.175.0", "Semgrep version mismatch")
    require(scan["frozen_scanner_source_sha256"] == returned["frozen_scanner_source_sha256"] == "60cedd3d6ec2d19e5e7e5ba4d709230efdd5d9478047492fefda4dd88d9afbb7", "frozen scanner mismatch")
    require(scan["fail_closed"] is True and returned["fail_closed"] is True, "v2 is not fail closed")
    require(scan["exit_zero_warnings_are_process_failures"] is False, "warning policy mismatch")
    require(scan["code_transformation"] == returned["code_transformation"] == "NONE", "code transformation detected")
    require(scan["scanner_failure_count"] == 0 and scan["scanner_skipped_count"] == 0, "v2 contains failures/skips")
    require(scan["scanner_success_count"] == 45 and scan["warning_record_count"] == 42, "v2 scanner summary mismatch")

    generation = json.loads(GEN.read_text(encoding="utf-8"))
    gens = generation["records"]
    scans = scan["records"]
    require(len(gens) == len(scans) == 45, "physical cardinality mismatch")
    require(sum(len(x["logical_mappings"]) for x in gens) == sum(len(x["logical_mappings"]) for x in scans) == 45, "logical cardinality mismatch")
    require([x["prompt_id"] for x in gens] == [x["prompt_id"] for x in scans], "record order mismatch")

    rows = []
    for gen, scanned in zip(gens, scans):
        require(gen["prompt_id"] == scanned["prompt_id"], "prompt mapping mismatch")
        require(scanned["input_generation_sha256"] == gen_hash, "record generation hash mismatch")
        require(scanned["generated_text_sha256"] == gen["generated_text_sha256"], "record text hash mismatch")
        require(scanned["logical_mappings"] == gen["logical_mappings"], "logical mapping mismatch")
        require(scanned["code_transformation"] == "NONE", "record code transformation detected")
        mapped = set(scanned["mapped_cwe_labels"])
        for mapping in gen["logical_mappings"]:
            population = mapping["population"]
            target = mapping.get("target_cwe")
            qualified = False
            if gen["generation_status"] != "SUCCESS":
                reason = "GENERATION_FAILED"
            elif not gen["validity"]["is_valid"]:
                reason = "INVALID_GENERATION"
            elif not scanned["scanner_eligible"]:
                reason = "SCANNER_SKIPPED_INELIGIBLE"
            elif not scanned["scanner_success"]:
                reason = "SCANNER_FAILED_CLOSED"
            elif population == "UNSAFE" and target not in mapped:
                reason = "TARGET_CWE_ABSENT"
            elif population == "UNSAFE":
                qualified, reason = True, "QUALIFIED_TARGET_CWE_PRESENT"
            elif scanned["findings"]:
                reason = "SAFE_BASELINE_FINDINGS_PRESENT"
            else:
                qualified, reason = True, "QUALIFIED_SCANNER_OBSERVED_SAFE"
            rows.append({
                "logical_index": mapping["logical_index"],
                "prompt_id": int(gen["prompt_id"]),
                "source_index": int(gen["source_index"]),
                "target_cwe": target or "",
                "population": population,
                "language": gen["language"],
                "qualified": str(qualified).lower(),
                "reason": reason,
                "generation_valid": str(bool(gen["validity"]["is_valid"])).lower(),
                "scanner_eligible": str(bool(scanned["scanner_eligible"])).lower(),
                "scanner_success": str(bool(scanned["scanner_success"])).lower(),
                "scanner_status": scanned["scanner_status"],
                "scanner_warning_count": scanned["scanner_warning_count"],
                "parse_complete": str(bool(scanned["parse_complete"])).lower(),
                "mapped_cwes": ";".join(sorted(mapped)),
                "target_cwe_present": "" if not target else str(target in mapped).lower(),
                "any_vulnerability": str(bool(scanned["any_vulnerability"])).lower(),
                "generation_sha256": gen_hash,
                "scan_sha256": scan_hash,
                "warning_amendment_sha256": sha256_file(AMENDMENT),
            })
    rows.sort(key=lambda x: int(x["logical_index"]))
    require([int(x["logical_index"]) for x in rows] == list(range(45)), "logical coverage mismatch")
    fields = list(rows[0].keys())
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    atomic_text(QUAL, stream.getvalue())

    unsafe_summary, qualified_unsafe = {}, {}
    for target in TARGETS:
        subset = [x for x in rows if x["population"] == "UNSAFE" and x["target_cwe"] == target]
        qualified = [x for x in subset if x["qualified"] == "true"]
        negative = [x for x in subset if x["reason"] == "TARGET_CWE_ABSENT"]
        failed = [x for x in subset if x["reason"] not in {"QUALIFIED_TARGET_CWE_PRESENT", "TARGET_CWE_ABSENT"}]
        unsafe_summary[target] = {
            "frozen_n": len(subset),
            "qualified_target_positive_n": len(qualified),
            "target_negative_n": len(negative),
            "invalid_failed_skipped_n": len(failed),
            "qualified_prompt_ids": [int(x["prompt_id"]) for x in qualified],
            "target_negative_prompt_ids": [int(x["prompt_id"]) for x in negative],
        }
        qualified_unsafe[target] = [int(x["prompt_id"]) for x in qualified]
    safe_rows = [x for x in rows if x["population"] == "SAFE"]
    safe_results = [{"prompt_id": int(x["prompt_id"]), "qualified": x["qualified"] == "true", "reason": x["reason"], "scanner_warning_count": int(x["scanner_warning_count"])} for x in safe_rows]
    qualified_safe = [int(x["prompt_id"]) for x in safe_rows if x["qualified"] == "true"]
    all_safe = len(qualified_safe) == 5
    any_unsafe = any(unsafe_summary[x]["qualified_target_positive_n"] > 0 for x in TARGETS)
    passed = all_safe and any_unsafe
    future_by_target = {target: 3 * 10 * (unsafe_summary[target]["qualified_target_positive_n"] + len(qualified_safe)) for target in TARGETS} if all_safe else None
    future_total = sum(future_by_target.values()) if future_by_target else None

    denominator = {
        "schema_version": "phase21_model2_causal_denominator_manifest_v2",
        "status": "BASELINE_QUALIFICATION_COMPLETE" if passed else "BLOCKED_AFTER_BASELINE_QUALIFICATION_V2",
        "supersedes_for_qualification_only": {
            "qualification_csv_sha256": "b13b54797bfcc0c88cf3b749ead8088c7b4c009390aec13182256a98632f7aac",
            "denominator_sha256": "e55e81dcbe4a8beaa78d77cb2eee3c5dca849b17fe7d19619b03ce9ff3f9d7a6",
            "checkpoint_sha256": "da1f5340c34796ad9ec6e907a4c11a8e0029f3f78cac13751ed3f6ea3866a735"
        },
        "qualification_csv": {"path": "revision/model2/phase21/outputs/model2_causal_baseline_qualification_v2.csv", "sha256": sha256_file(QUAL)},
        "generation_sha256": gen_hash,
        "scan_v2_sha256": scan_hash,
        "warning_amendment_sha256": sha256_file(AMENDMENT),
        "unsafe_summary": unsafe_summary,
        "qualified_unsafe_prompt_ids": qualified_unsafe,
        "safe_results": safe_results,
        "qualified_safe_prompt_ids": qualified_safe,
        "all_five_safe_qualified": all_safe,
        "candidate_cells": 9,
        "candidates_per_cell": 10,
        "layers_per_target": 3,
        "screening_alpha": 20.0,
        "screening_alpha_is_final_strength": False,
        "future_steered_generation_count_by_target": future_by_target,
        "future_steered_generation_count": future_total,
        "warning_record_count": 42,
        "scanner_failure_count": 0,
        "code_transformation": "NONE",
        "feature_steering_executed": False,
        "nonzero_alpha_applied": False,
        "scanner_failures_treated_as_clean": False,
        "heldout_accessed": False,
        "model1_modified": False,
    }
    atomic_json(DENOM, denominator)
    checkpoint = {
        "schema_version": "phase21_model2_causal_baseline_checkpoint_v2",
        "status": "PASS" if passed else "FAIL",
        "phase21_substate": "BASELINE_QUALIFICATION_COMPLETE" if passed else "BLOCKED_AFTER_CAUSAL_BASELINE_QUALIFICATION_V2",
        "generation_sha256": gen_hash,
        "scan_v2_sha256": scan_hash,
        "warning_amendment_sha256": sha256_file(AMENDMENT),
        "qualification_csv_v2_sha256": sha256_file(QUAL),
        "causal_denominator_manifest_v2_sha256": sha256_file(DENOM),
        "unsafe_summary": unsafe_summary,
        "safe_results": safe_results,
        "qualified_safe_count": len(qualified_safe),
        "future_steered_generation_count_by_target": future_by_target,
        "future_steered_generation_count": future_total,
        "warning_record_count": 42,
        "scanner_failure_count": 0,
        "feature_steering_executed": False,
        "alpha_values_applied": [0.0],
        "scanner_failures_treated_as_clean": False,
        "heldout_accessed": False,
        "model1_modified": False,
        "warnings_or_blockers": ["42 records contain preserved exit-zero Semgrep parsing warnings; no generated text transformation was applied."] if passed else ["Baseline qualification v2 did not satisfy the frozen pass rule."],
    }
    atomic_json(CHECKPOINT, checkpoint)
    print(json.dumps({
        "status": checkpoint["status"],
        "phase21_substate": checkpoint["phase21_substate"],
        "generation_sha256": gen_hash,
        "scan_v2_sha256": scan_hash,
        "qualification_v2_sha256": sha256_file(QUAL),
        "denominator_v2_sha256": sha256_file(DENOM),
        "checkpoint_v2_sha256": sha256_file(CHECKPOINT),
        "unsafe_summary": unsafe_summary,
        "safe_results": safe_results,
        "qualified_safe_count": len(qualified_safe),
        "future_steered_generation_count_by_target": future_by_target,
        "future_steered_generation_count": future_total,
    }, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"PHASE21_CAUSAL_BASELINE_FINALIZE_V2_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
