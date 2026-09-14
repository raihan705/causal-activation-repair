#!/usr/bin/env python
"""Verify the Stage 22E paired B0 scan return and freeze causal denominators."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
PHASE = ROOT / "revision/model3/phase22"
OUT = PHASE / "outputs"
BUNDLE = PHASE / "phase22e_baseline_colab_scan_bundle"

PROTOCOL = OUT / "phase22e_causal_validation_protocol.json"
GENERATION = OUT / "phase22e_paired_baseline_generations.json"
SCAN_INPUT = OUT / "phase22e_paired_baseline_scan_input.json"
SCAN_OUTPUT = OUT / "phase22e_paired_baseline_scans.json"
RETURN_MANIFEST = OUT / "phase22e_paired_baseline_scan_return_manifest.json"
TRANSFER_MANIFEST = BUNDLE / "phase22e_paired_baseline_scan_transfer_manifest.json"
SCANNER = ROOT / "phases/phase9/colab_scan_phase9.py"
WRAPPER = PHASE / "scripts/scan_phase22e_paired_baseline.py"

VERIFICATION = OUT / "phase22e_paired_baseline_scan_verification.json"
DENOMINATOR = OUT / "phase22e_paired_denominator_manifest.json"

EXPECTED = {
    "protocol": "e0fcf227460e7408471caabb1d695fa1905bc51073a1581776aa1772b13932f4",
    "generation": "e12e46bacb0fc3725e17e4e6258f71cc3f77001d2fe7b646b88b05408b684a26",
    "scan_input": "b34bbf485b50a6a5c641f33e512e18e7b30f9eebc2e41f58a33321e5f55d47f6",
    "scan_output": "80b7ed613c61fcbc1f0c58ca853402c4b517bb576310743d9af10da3a0c9fa7c",
    "scanner": "60cedd3d6ec2d19e5e7e5ba4d709230efdd5d9478047492fefda4dd88d9afbb7",
    "wrapper": "f6475a6993940744fd67aa14367490c1aaf9a3eec5fdfe2d9802f56d714d1cfb",
}
TARGETS = ("CWE-120", "CWE-327", "CWE-89")
DERIVATION_CWES = {"CWE-120", "CWE-125", "CWE-787", "CWE-190", "CWE-476", "CWE-89", "CWE-79", "CWE-327"}
REQUIRED_SCAN_KEYS = {"prompt_id", "cwe_id", "language", "findings", "vulnerable_cwes", "is_vulnerable", "skipped"}
REQUIRED_FINDING_KEYS = {"cwe_id", "rule_id", "severity", "location", "source"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    required = (PROTOCOL, GENERATION, SCAN_INPUT, SCAN_OUTPUT, RETURN_MANIFEST,
                TRANSFER_MANIFEST, SCANNER, WRAPPER)
    for path in required:
        require(path.is_file(), f"missing required artifact: {path}")

    hashes = {
        "protocol": sha256_file(PROTOCOL),
        "generation": sha256_file(GENERATION),
        "scan_input": sha256_file(SCAN_INPUT),
        "scan_output": sha256_file(SCAN_OUTPUT),
        "return_manifest": sha256_file(RETURN_MANIFEST),
        "transfer_manifest": sha256_file(TRANSFER_MANIFEST),
        "scanner": sha256_file(SCANNER),
        "wrapper": sha256_file(WRAPPER),
    }
    for name, expected in EXPECTED.items():
        require(hashes[name] == expected, f"{name} hash mismatch: {hashes[name]}")

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    generation = json.loads(GENERATION.read_text(encoding="utf-8"))
    scan_input = json.loads(SCAN_INPUT.read_text(encoding="utf-8"))
    scans = json.loads(SCAN_OUTPUT.read_text(encoding="utf-8"))
    returned = json.loads(RETURN_MANIFEST.read_text(encoding="utf-8"))
    transfer = json.loads(TRANSFER_MANIFEST.read_text(encoding="utf-8"))
    generated = generation["records"]
    plan = protocol["paired_baseline_prompts"]

    require(returned["status"] == "COMPLETE", "return manifest is not COMPLETE")
    require(returned["semgrep_version"] == transfer["required_semgrep_version"] == "1.175.0", "Semgrep version mismatch")
    require(returned["paired_baseline_generation_sha256"] == hashes["generation"], "returned generation provenance mismatch")
    require(returned["scan_input_sha256"] == transfer["scan_input_sha256"] == hashes["scan_input"], "scan-input provenance mismatch")
    require(returned["scan_output_sha256"] == hashes["scan_output"], "returned scan-output hash mismatch")
    require(returned["scanner_sha256"] == transfer["scanner_sha256"] == hashes["scanner"], "scanner provenance mismatch")
    require(returned["wrapper_sha256"] == transfer["wrapper_sha256"] == hashes["wrapper"], "wrapper provenance mismatch")
    require(returned["record_count"] == returned["eligible_count"] == 40, "return count/eligibility mismatch")
    require(returned["skipped_count"] == 0, "return reports scanner-skipped records")
    require(not returned["scanner_rules_changed"] and not returned["steering_run"] and not returned["heldout_used"], "return boundary violation")

    require(isinstance(generated, list) and isinstance(scan_input, list) and isinstance(scans, list), "record containers must be lists")
    require(len(generated) == len(scan_input) == len(scans) == len(plan) == 40, "record-count mismatch")
    require([x["record_id"] for x in generated] == [x["record_id"] for x in plan] == transfer["record_ids"], "frozen record order mismatch")
    require([int(x["prompt_id"]) for x in generated] == [int(x["prompt_id"]) for x in scan_input] == [int(x["prompt_id"]) for x in scans] == transfer["prompt_ids"], "prompt ID/order mismatch")
    require(len({int(x["prompt_id"]) for x in generated}) == 40, "duplicate paired prompt IDs")

    qualified_unsafe: dict[str, list[int]] = defaultdict(list)
    qualified_safe: dict[str, list[int]] = defaultdict(list)
    excluded: dict[str, list[dict[str, Any]]] = defaultdict(list)
    audit_rows: list[dict[str, Any]] = []

    for index, (planned, gen, inp, scan) in enumerate(zip(plan, generated, scan_input, scans)):
        require(REQUIRED_SCAN_KEYS.issubset(scan), f"scan schema mismatch at row {index}")
        require(planned["record_id"] == gen["record_id"] == inp["record_id"], f"record ID mismatch at row {index}")
        require(int(planned["prompt_id"]) == int(gen["prompt_id"]) == int(inp["prompt_id"]) == int(scan["prompt_id"]), f"prompt mismatch at row {index}")
        require(planned["target_cwe"] == gen["target_cwe"] == inp["cwe_id"] == scan["cwe_id"], f"target mismatch at row {index}")
        require(planned["language"].lower() == gen["language"].lower() == inp["language"].lower() == scan["language"].lower(), f"language mismatch at row {index}")
        require(gen["generated_code"] == inp["generated_code"], f"scan projection code mismatch at row {index}")
        require(gen["generation_status"] == "SUCCESS" and gen["validity"]["is_valid"], f"invalid paired generation at row {index}")
        require(not gen["sae_intervention_applied"] and not gen["causal_steering_applied"], f"baseline intervention boundary violation at row {index}")
        require(isinstance(scan["findings"], list) and isinstance(scan["vulnerable_cwes"], list), f"scan list schema mismatch at row {index}")
        require(all(REQUIRED_FINDING_KEYS.issubset(f) for f in scan["findings"]), f"finding schema mismatch at row {index}")
        finding_keys = [(f["cwe_id"], f["rule_id"]) for f in scan["findings"]]
        require(len(finding_keys) == len(set(finding_keys)), f"duplicate findings at row {index}")
        finding_cwes = {f["cwe_id"] for f in scan["findings"]}
        require(finding_cwes == set(scan["vulnerable_cwes"]), f"vulnerable_cwes derivation mismatch at row {index}")
        require(bool(scan["is_vulnerable"]) == bool(finding_cwes & DERIVATION_CWES), f"is_vulnerable derivation mismatch at row {index}")
        require(not scan["skipped"], f"scanner-skipped row {index}")

        target = gen["target_cwe"]
        population = gen["population"]
        prompt_id = int(gen["prompt_id"])
        if population == "UNSAFE_CANDIDATE":
            qualifies = target in finding_cwes
            reason = "TARGET_CWE_PRESENT" if qualifies else "TARGET_CWE_NOT_REPLICATED"
            if qualifies:
                qualified_unsafe[target].append(prompt_id)
            else:
                excluded[target].append({"record_id": gen["record_id"], "prompt_id": prompt_id, "population": population, "reason": reason, "observed_cwes": sorted(finding_cwes)})
        elif population == "SAFE_CANDIDATE":
            qualifies = len(scan["findings"]) == 0
            reason = "NO_FINDINGS" if qualifies else "ANY_BASELINE_FINDING_PRESENT"
            if qualifies:
                qualified_safe[target].append(prompt_id)
            else:
                excluded[target].append({"record_id": gen["record_id"], "prompt_id": prompt_id, "population": population, "reason": reason, "observed_cwes": sorted(finding_cwes)})
        else:
            raise RuntimeError(f"unknown population at row {index}: {population}")
        audit_rows.append({
            "record_id": gen["record_id"], "prompt_id": prompt_id, "target_cwe": target,
            "population": population, "language": gen["language"], "valid": True,
            "scanner_eligible": True, "observed_cwes": sorted(finding_cwes),
            "finding_count": len(scan["findings"]), "qualified": qualifies, "qualification_reason": reason,
        })

    candidate_counts = Counter((x["target_cwe"], x["population"]) for x in generated)
    target_summaries: dict[str, Any] = {}
    expected_workload = 0
    all_evaluable = True
    for target in TARGETS:
        unsafe_n = len(qualified_unsafe[target])
        safe_n = len(qualified_safe[target])
        evaluable = unsafe_n > 0 and safe_n > 0
        target_workload = 30 * (unsafe_n + safe_n) if evaluable else 0
        expected_workload += target_workload
        all_evaluable &= evaluable
        target_summaries[target] = {
            "unsafe_candidates": candidate_counts[(target, "UNSAFE_CANDIDATE")],
            "qualified_unsafe_count": unsafe_n,
            "qualified_unsafe_prompt_ids": qualified_unsafe[target],
            "safe_candidates": candidate_counts[(target, "SAFE_CANDIDATE")],
            "qualified_safe_count": safe_n,
            "qualified_safe_prompt_ids": qualified_safe[target],
            "excluded_count": len(excluded[target]),
            "excluded_records": excluded[target],
            "candidate_assignments": 30,
            "steered_generation_count": target_workload,
            "evaluation_status": "EVALUABLE" if evaluable else "NOT_EVALUABLE",
        }

    denominator = {
        "schema_version": "phase22e_paired_denominator_manifest_v1",
        "status": "FROZEN_BEFORE_STEERING" if all_evaluable else "FROZEN_WITH_NOT_EVALUABLE_TARGETS",
        "protocol_sha256": hashes["protocol"],
        "paired_baseline_generation_sha256": hashes["generation"],
        "scan_input_sha256": hashes["scan_input"],
        "scan_output_sha256": hashes["scan_output"],
        "scan_return_manifest_sha256": hashes["return_manifest"],
        "scanner_sha256": hashes["scanner"],
        "wrapper_sha256": hashes["wrapper"],
        "semgrep_version": returned["semgrep_version"],
        "qualification_rule": {
            "unsafe": "valid and scanner-eligible paired B0 with target CWE present",
            "safe": "valid and scanner-eligible paired B0 with zero scanner findings",
            "replacement": "NONE",
        },
        "targets": target_summaries,
        "all_targets_evaluable": all_evaluable,
        "candidate_assignments_per_target": 30,
        "expected_steered_generation_count": expected_workload,
        "maximum_frozen_protocol_count": protocol["expected_max_steered_generation_count"],
        "audit_rows": audit_rows,
        "causal_steering_run": False,
        "heldout_used": False,
    }
    atomic_json(DENOMINATOR, denominator)
    denominator_hash = sha256_file(DENOMINATOR)

    verification = {
        "schema_version": "phase22e_paired_baseline_scan_verification_v1",
        "status": "PASS",
        "checks": {
            "immutable_artifact_hashes": True,
            "return_manifest_complete": True,
            "semgrep_version_exact": True,
            "scanner_and_wrapper_exact": True,
            "record_count_order_and_schema": True,
            "row_metadata_and_code_projection": True,
            "finding_dedup_and_derived_fields": True,
            "eligible_40_skipped_0": True,
            "baseline_validity_40_of_40": True,
            "no_intervention_no_heldout": True,
            "no_replacement_denominators_frozen": True,
        },
        "artifact_hashes": hashes,
        "returned_environment": {
            "platform": returned["platform"], "python_version": returned["python_version"],
            "semgrep_version": returned["semgrep_version"], "completed_at_utc": returned["completed_at_utc"],
        },
        "record_count": 40,
        "eligible_count": 40,
        "skipped_count": 0,
        "scanner_vulnerable_count": sum(bool(x["is_vulnerable"]) for x in scans),
        "target_summaries": target_summaries,
        "all_targets_evaluable": all_evaluable,
        "expected_steered_generation_count": expected_workload,
        "denominator_manifest_sha256": denominator_hash,
        "scanner_limitation": "The frozen scanner suppresses Semgrep subprocess exceptions and the registry config is not content-pinned; exact runtime, artifact hashes, full row coverage, and skip policy are verified, but silent per-record Semgrep failure cannot be independently excluded.",
        "causal_steering_run": False,
        "heldout_used": False,
    }
    atomic_json(VERIFICATION, verification)
    print(json.dumps({
        "status": verification["status"], "scan_output_sha256": hashes["scan_output"],
        "denominator_manifest_sha256": denominator_hash,
        "all_targets_evaluable": all_evaluable,
        "expected_steered_generation_count": expected_workload,
        "targets": target_summaries,
    }, indent=2))


if __name__ == "__main__":
    main()
