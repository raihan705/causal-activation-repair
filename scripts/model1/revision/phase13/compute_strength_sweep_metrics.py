#!/usr/bin/env python3
"""Compute deterministic Phase 13 strength-sweep metrics from frozen artifacts.

This script performs no generation and no scanning.  It fails closed on every
frozen provenance, population, order, and hash requirement before serializing
the registered seven-condition metric outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Any


ALPHAS = (10, 20, 30, 40, 50, 60, 80)
ACTIVE_CWES = ("CWE-120", "CWE-327", "CWE-338", "CWE-89")
SEED = 42
TOTAL = 180
DEV_VULN_COUNT = 60
DEV_SAFE_COUNT = 120
STEERED_COUNT = 150
UNSTEERED_COUNT = 30

FROZEN_HASHES = {
    "strength_subset": "f8781a4532a0d58840a89652342fb88eeda7480cf42924ae1ee451f0b7b898c7",
    "dev_subset_hashes": "8266cd3a95e025a96cb9a7cf0252bbcbef20954ad647031bdee0c59344493010",
    "denominator_audit": "75b5ca0a029e43a49a9048d17a94713c0003aad6f144352539c6b0ecede3f619",
    "b0_outputs": "0a33b488b88a0414fc20182602b622fd029e79e86a27c63ff54fe307926a92a4",
    "b0_icd": "8af54a450b133a315b5578357e07b8201ddf2f431f8fee04874ef1f4a18974d6",
    "scan_return_manifest": "b10752f8c1b2edf2f5fc60f8caa7f914222a52ca3f7306358108a1f1834c27b5",
    "scan_transfer_manifest": "69b1f0069842a4afba0fb4696810aec238a2cb0456429f34f37fda1e88132df5",
    "bstar_config": "f4f5d4d6a697aa14f654763f31e52cd852ce87b9f284276edd4f784604ea36e6",
    "generation_summary": "9d99013f7f835f7c244847af33766967c16395553a85585ff6f5cbcc0f03911e",
    "scanner": "60cedd3d6ec2d19e5e7e5ba4d709230efdd5d9478047492fefda4dd88d9afbb7",
}

OVERALL_FIELDS = (
    "alpha", "seed", "total_prompts", "dev_vuln_denominator",
    "dev_safe_large_denominator", "steered_count", "unsteered_count",
    "scanner_eligible_count", "scanner_skipped_count", "raw_vulnerable_count",
    "corrected_vulnerable_count", "repair_count", "repair_rate", "raw_vrr",
    "corrvrr", "corruption_count", "corruption_rate", "safe_to_safe_count",
    "safe_to_safe_preservation", "validity_count", "validity_rate",
    "empty_output_count", "empty_output_rate", "whitespace_only_count",
    "whitespace_only_rate", "hook_failure_count", "generation_failure_count",
    "malformed_failure_count", "other_failure_count", "fallback_count",
)

PER_CWE_FIELDS = (
    "alpha", "seed", "cwe_id", "cwe_type", "b0_vulnerable_denominator",
    "estimability", "repair_count", "corrvrr", "corrected_vulnerable_count",
    "vulnerable_count", "safe_denominator", "corruption_count",
    "corruption_rate", "source_population_count", "validity_count",
    "validity_rate", "empty_output_count", "empty_output_rate",
    "whitespace_only_count", "whitespace_only_rate", "scanner_skipped_count",
    "hook_failure_count", "generation_failure_count", "malformed_failure_count",
    "other_failure_count", "fallback_count",
)


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValidationError(f"Cannot parse JSON {path}: {exc}") from exc


def check_hash(path: Path, expected: str, label: str) -> str:
    require(path.is_file(), f"Missing {label}: {path}")
    actual = sha256_file(path)
    require(actual == expected, f"{label} hash mismatch: expected {expected}, got {actual}")
    return actual


def rate(numerator: int, denominator: int) -> float:
    require(denominator > 0, "Rate denominator must be positive")
    return numerator / denominator


def rate_text(value: float) -> str:
    return f"{value:.6f}"


def is_valid_output(record: dict[str, Any]) -> bool:
    """Exact Phase 5 validity rule: bool(generated_code.strip())."""
    code = record.get("generated_code", "")
    require(isinstance(code, str), f"generated_code is not a string for prompt {record.get('prompt_id')}")
    return bool(code.strip())


def is_strict_empty(record: dict[str, Any]) -> bool:
    return record.get("generated_code", "") == ""


def is_whitespace_only(record: dict[str, Any]) -> bool:
    code = record.get("generated_code", "")
    return bool(code) and not code.strip()


def csv_bytes(fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def by_prompt(records: list[dict[str, Any]], label: str) -> dict[Any, dict[str, Any]]:
    require(isinstance(records, list), f"{label} must be a JSON list")
    ids = [record.get("prompt_id") for record in records]
    require(all(prompt_id is not None for prompt_id in ids), f"{label} contains missing prompt_id")
    require(len(ids) == len(set(ids)), f"{label} contains duplicate prompt IDs")
    return {record["prompt_id"]: record for record in records}


def failure_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    known_nonfailure = {"SUCCESS", "WHITESPACE_ONLY"}
    hook = sum(record.get("generation_status") == "HOOK_EXCEPTION" for record in records)
    generation = sum(record.get("generation_status") == "GENERATION_EXCEPTION" for record in records)
    malformed = sum(record.get("generation_status") == "MALFORMED_GENERATION" for record in records)
    other = sum(
        record.get("generation_status") not in known_nonfailure | {
            "HOOK_EXCEPTION", "GENERATION_EXCEPTION", "MALFORMED_GENERATION"
        }
        for record in records
    )
    fallback = sum(record.get("fallback_status") != "NONE" for record in records)
    return {
        "hook_failure_count": hook,
        "generation_failure_count": generation,
        "malformed_failure_count": malformed,
        "other_failure_count": other,
        "fallback_count": fallback,
    }


def validate_inputs(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "strength_subset": args.strength_subset.resolve(),
        "dev_subset_hashes": args.dev_subset_hashes.resolve(),
        "denominator_audit": args.denominator_audit.resolve(),
        "b0_outputs": args.b0_outputs.resolve(),
        "b0_icd": args.b0_icd.resolve(),
        "scan_return_manifest": args.scan_return_manifest.resolve(),
        "scan_transfer_manifest": args.scan_transfer_manifest.resolve(),
        "bstar_config": args.bstar_config.resolve(),
        "generation_summary": args.generation_summary.resolve(),
    }
    for key, path in paths.items():
        check_hash(path, FROZEN_HASHES[key], key)

    subset = load_json(paths["strength_subset"])
    subset_hashes = load_json(paths["dev_subset_hashes"])
    denominator = load_json(paths["denominator_audit"])
    b0_outputs = load_json(paths["b0_outputs"])
    b0_icd = load_json(paths["b0_icd"])
    scan_return = load_json(paths["scan_return_manifest"])
    scan_transfer = load_json(paths["scan_transfer_manifest"])
    bstar = load_json(paths["bstar_config"])
    generation_summary = load_json(paths["generation_summary"])

    require(subset.get("split") == "DEVELOPMENT", "Strength subset is not DEVELOPMENT")
    require(subset.get("validation_status") == "PASS", "Strength subset validation is not PASS")
    require(subset.get("counts") == {
        "dev_vuln": DEV_VULN_COUNT,
        "dev_safe_large": DEV_SAFE_COUNT,
        "strength_subset": TOTAL,
    }, "Strength subset counts differ from frozen 60/120/180")
    records = subset.get("records")
    require(isinstance(records, list) and len(records) == TOTAL, "Strength subset record count is not 180")
    ids = [record.get("prompt_id") for record in records]
    require(len(set(ids)) == TOTAL, "Strength subset IDs are not unique")
    dev_vuln_ids = [r["prompt_id"] for r in records if r.get("population_type") == "DEV_VULN"]
    dev_safe_ids = [r["prompt_id"] for r in records if r.get("population_type") == "DEV_SAFE_LARGE"]
    require(len(dev_vuln_ids) == DEV_VULN_COUNT, "DEV_VULN count is not 60")
    require(len(dev_safe_ids) == DEV_SAFE_COUNT, "DEV_SAFE_LARGE count is not 120")
    safe_cwes = {r.get("cwe_id") for r in records if r.get("population_type") == "DEV_SAFE_LARGE"}
    require(safe_cwes == set(ACTIVE_CWES), f"DEV_SAFE_LARGE CWE set mismatch: {sorted(safe_cwes)}")
    require(all(r.get("scanner_eligible") is True for r in records), "Subset contains scanner-ineligible record")
    require(all(r.get("b0_is_vulnerable") is True for r in records if r.get("population_type") == "DEV_VULN"), "DEV_VULN contains non-vulnerable B0 record")
    require(all(r.get("b0_is_vulnerable") is False for r in records if r.get("population_type") == "DEV_SAFE_LARGE"), "DEV_SAFE_LARGE contains vulnerable B0 record")

    require(subset_hashes.get("validation_status") == "PASS", "dev_subset_hashes validation is not PASS")
    require(subset_hashes.get("n_vuln") == DEV_VULN_COUNT, "dev_subset_hashes n_vuln mismatch")
    require(subset_hashes.get("manifest_files", {}).get("strength_subset_manifest", {}).get("sha256") == FROZEN_HASHES["strength_subset"], "dev_subset_hashes subset link mismatch")
    require(subset_hashes.get("nesting_validation", {}).get("dev_safe_small_count") == DEV_VULN_COUNT, "DEV_SAFE_SMALL count is not 60")
    require(subset_hashes.get("nesting_validation", {}).get("dev_safe_large_count") == DEV_SAFE_COUNT, "DEV_SAFE_LARGE nesting count mismatch")
    require(subset_hashes.get("nesting_validation", {}).get("dev_safe_small_strict_subset_dev_safe_large") is True, "DEV_SAFE_SMALL nesting is not verified")
    require(denominator.get("verification_status") == "PASS", "Phase 2 denominator audit is not PASS")
    require(not denominator.get("blockers"), "Phase 2 denominator audit has blockers")

    b0_output_map = by_prompt(b0_outputs, "B0 outputs")
    b0_icd_map = by_prompt(b0_icd, "B0 ICD")
    require(set(ids).issubset(b0_output_map), "Subset ID absent from B0 outputs")
    require(set(ids).issubset(b0_icd_map), "Subset ID absent from B0 ICD")
    for record in records:
        prompt_id = record["prompt_id"]
        b0_scan = b0_icd_map[prompt_id]
        require(b0_scan.get("skipped") is False, f"Subset prompt {prompt_id} is B0 scanner-skipped")
        require(bool(b0_scan.get("is_vulnerable")) == bool(record["b0_is_vulnerable"]), f"Subset/B0 vulnerability mismatch for {prompt_id}")
        output_hash = sha256_bytes(b0_output_map[prompt_id].get("generated_code", "").encode("utf-8"))
        require(output_hash == record.get("b0_output_sha256"), f"Subset/B0 output hash mismatch for {prompt_id}")

    require(scan_return.get("phase") == 13 and scan_return.get("status") == "COMPLETE", "Scan return manifest is not Phase 13 COMPLETE")
    require(scan_return.get("scan_order") == list(ALPHAS), "Scan return order mismatch")
    require(scan_return.get("seed") == SEED, "Scan return seed mismatch")
    require(scan_return.get("strength_subset_sha256") == FROZEN_HASHES["strength_subset"], "Scan return subset hash mismatch")
    require(scan_return.get("transfer_manifest_sha256") == FROZEN_HASHES["scan_transfer_manifest"], "Scan return transfer hash mismatch")
    require(scan_return.get("scanner_source_sha256") == FROZEN_HASHES["scanner"], "Scan return scanner hash mismatch")
    require(scan_return.get("historical_semgrep_version") == "NOT_REPRODUCIBLE", "Historical Semgrep version status changed")
    require(scan_return.get("historical_semgrep_registry_state") == "NOT_REPRODUCIBLE", "Historical Semgrep registry status changed")
    environment = scan_return.get("execution_environment", {})
    require(environment.get("platform", "").startswith("Linux"), "Current scan platform is not Linux")
    require(environment.get("python_version") == "3.12.13", "Current scan Python version mismatch")
    require(environment.get("semgrep_version") == "1.172.0", "Current scan Semgrep version mismatch")

    transfer_entries = scan_transfer.get("entries")
    scan_entries = scan_return.get("entries")
    require(isinstance(transfer_entries, list) and [e.get("alpha") for e in transfer_entries] == list(ALPHAS), "Transfer entries/order mismatch")
    require(isinstance(scan_entries, list) and [e.get("alpha") for e in scan_entries] == list(ALPHAS), "Scan entries/order mismatch")
    require([e.get("completion_order") for e in scan_entries] == list(range(1, 8)), "Scan completion order mismatch")
    require(all(e.get("scan_status") == "COMPLETE" for e in scan_entries), "A scan entry is not COMPLETE")
    require(all(e.get("resume_status") == "EXECUTED_FRESH" for e in scan_entries), "A scan entry was not EXECUTED_FRESH")
    require(generation_summary.get("seed") == SEED, "Generation summary seed mismatch")
    summary_conditions = generation_summary.get("conditions")
    require(isinstance(summary_conditions, list) and [c.get("alpha") for c in summary_conditions] == list(ALPHAS), "Generation summary alpha order mismatch")
    summary_by_alpha = {condition["alpha"]: condition for condition in summary_conditions}
    transfer_by_alpha = {entry["alpha"]: entry for entry in transfer_entries}
    scan_by_alpha = {entry["alpha"]: entry for entry in scan_entries}

    generation_records: dict[int, list[dict[str, Any]]] = {}
    icd_records: dict[int, list[dict[str, Any]]] = {}
    generation_paths: dict[int, Path] = {}
    icd_paths: dict[int, Path] = {}
    for alpha in ALPHAS:
        transfer_entry = transfer_by_alpha[alpha]
        scan_entry = scan_by_alpha[alpha]
        generation_path = paths["strength_subset"].parent / Path(transfer_entry["final_generation_path"]).name
        icd_path = paths["strength_subset"].parent / Path(scan_entry["scanner_output_path"]).name
        require(generation_path.is_file(), f"Missing alpha {alpha} generation")
        require(icd_path.is_file(), f"Missing alpha {alpha} ICD")
        generation_hash = sha256_file(generation_path)
        icd_hash = sha256_file(icd_path)
        require(generation_hash == transfer_entry.get("final_generation_sha256"), f"Alpha {alpha} generation hash mismatch")
        require(generation_hash == scan_entry.get("generation_sha256") == scan_entry.get("pre_scan_generation_sha256") == scan_entry.get("post_scan_generation_sha256"), f"Alpha {alpha} generation linkage mismatch")
        require(scan_entry.get("generation_hash_unchanged") is True, f"Alpha {alpha} generation unchanged flag is false")
        require(icd_hash == scan_entry.get("scanner_output_sha256"), f"Alpha {alpha} ICD hash mismatch")
        generation = load_json(generation_path)
        icd = load_json(icd_path)
        require(isinstance(generation, list) and len(generation) == TOTAL, f"Alpha {alpha} generation count mismatch")
        require(isinstance(icd, list) and len(icd) == TOTAL, f"Alpha {alpha} ICD count mismatch")
        require([r.get("prompt_id") for r in generation] == ids, f"Alpha {alpha} generation ID/order mismatch")
        require([r.get("prompt_id") for r in icd] == ids, f"Alpha {alpha} ICD ID/order mismatch")
        require(len({r["prompt_id"] for r in generation}) == TOTAL, f"Alpha {alpha} generation duplicate IDs")
        require(len({r["prompt_id"] for r in icd}) == TOTAL, f"Alpha {alpha} ICD duplicate IDs")
        require(all(r.get("run_seed") == SEED and r.get("condition_alpha") == alpha for r in generation), f"Alpha {alpha} generation seed/alpha mismatch")
        steered = sum(r.get("steered") is True for r in generation)
        unsteered = sum(r.get("steered") is False for r in generation)
        require((steered, unsteered) == (STEERED_COUNT, UNSTEERED_COUNT), f"Alpha {alpha} route counts mismatch")
        eligible = sum(r.get("skipped") is False for r in icd)
        skipped = sum(r.get("skipped") is True for r in icd)
        summary = summary_by_alpha[alpha]
        require(summary.get("output_sha256") == generation_hash, f"Alpha {alpha} generation-summary output hash mismatch")
        require((summary.get("records"), summary.get("unique_records"), summary.get("steered"), summary.get("unsteered")) == (TOTAL, TOTAL, STEERED_COUNT, UNSTEERED_COUNT), f"Alpha {alpha} generation-summary structural counts mismatch")
        whitespace = sum(is_whitespace_only(r) for r in generation)
        strict_empty = sum(is_strict_empty(r) for r in generation)
        require(summary.get("whitespace_only_outputs") == whitespace, f"Alpha {alpha} whitespace summary mismatch")
        require(summary.get("empty_outputs") == strict_empty, f"Alpha {alpha} empty summary mismatch")
        require((scan_entry.get("scanner_eligible_count"), scan_entry.get("scanner_skipped_count")) == (eligible, skipped), f"Alpha {alpha} scanner count linkage mismatch")
        require(skipped == whitespace + strict_empty, f"Alpha {alpha} scanner skips do not reconcile with invalid outputs")
        generation_records[alpha] = generation
        icd_records[alpha] = icd
        generation_paths[alpha] = generation_path
        icd_paths[alpha] = icd_path

    require(bstar.get("bstar") == "B2_a40" and float(bstar.get("alpha")) == 40.0, "Frozen B* identity/alpha mismatch")

    return {
        "paths": paths,
        "subset": subset,
        "subset_records": records,
        "ids": ids,
        "dev_vuln_ids": dev_vuln_ids,
        "dev_safe_ids": dev_safe_ids,
        "b0_icd_map": b0_icd_map,
        "scan_return": scan_return,
        "generation_summary": generation_summary,
        "summary_by_alpha": summary_by_alpha,
        "generation_records": generation_records,
        "icd_records": icd_records,
        "generation_paths": generation_paths,
        "icd_paths": icd_paths,
    }


def compute_artifacts(args: argparse.Namespace) -> tuple[dict[str, bytes], dict[str, Any]]:
    validated = validate_inputs(args)
    subset_records = validated["subset_records"]
    dev_vuln_ids = validated["dev_vuln_ids"]
    dev_safe_ids = validated["dev_safe_ids"]
    b0_icd_map = validated["b0_icd_map"]
    subset_by_id = {record["prompt_id"]: record for record in subset_records}

    overall_rows: list[dict[str, Any]] = []
    per_cwe_rows: list[dict[str, Any]] = []
    numeric_overall: dict[int, dict[str, Any]] = {}

    for alpha in ALPHAS:
        generation = validated["generation_records"][alpha]
        icd = validated["icd_records"][alpha]
        generation_map = {record["prompt_id"]: record for record in generation}
        icd_map = {record["prompt_id"]: record for record in icd}
        valid = {prompt_id: is_valid_output(generation_map[prompt_id]) for prompt_id in validated["ids"]}
        strict_empty = {prompt_id: is_strict_empty(generation_map[prompt_id]) for prompt_id in validated["ids"]}
        whitespace = {prompt_id: is_whitespace_only(generation_map[prompt_id]) for prompt_id in validated["ids"]}
        failures = failure_counts(generation)

        scanner_eligible = sum(icd_map[prompt_id].get("skipped") is False for prompt_id in validated["ids"])
        scanner_skipped = TOTAL - scanner_eligible
        raw_vulnerable = sum(icd_map[prompt_id].get("is_vulnerable") is True for prompt_id in validated["ids"])
        repaired_ids = {
            prompt_id for prompt_id in dev_vuln_ids
            if valid[prompt_id] and icd_map[prompt_id].get("is_vulnerable") is False
        }
        repair_count = len(repaired_ids)
        corrected_vulnerable = DEV_VULN_COUNT - repair_count
        corrupted_ids = {
            prompt_id for prompt_id in dev_safe_ids
            if icd_map[prompt_id].get("is_vulnerable") is True
        }
        corruption_count = len(corrupted_ids)
        validity_count = sum(valid.values())
        empty_count = sum(strict_empty.values())
        whitespace_count = sum(whitespace.values())
        steered_count = sum(record.get("steered") is True for record in generation)
        unsteered_count = sum(record.get("steered") is False for record in generation)

        values = {
            "alpha": alpha,
            "seed": SEED,
            "total_prompts": TOTAL,
            "dev_vuln_denominator": DEV_VULN_COUNT,
            "dev_safe_large_denominator": DEV_SAFE_COUNT,
            "steered_count": steered_count,
            "unsteered_count": unsteered_count,
            "scanner_eligible_count": scanner_eligible,
            "scanner_skipped_count": scanner_skipped,
            "raw_vulnerable_count": raw_vulnerable,
            "corrected_vulnerable_count": corrected_vulnerable,
            "repair_count": repair_count,
            "repair_rate": rate(repair_count, DEV_VULN_COUNT),
            "raw_vrr": 1 - rate(raw_vulnerable, DEV_VULN_COUNT),
            "corrvrr": rate(repair_count, DEV_VULN_COUNT),
            "corruption_count": corruption_count,
            "corruption_rate": rate(corruption_count, DEV_SAFE_COUNT),
            "safe_to_safe_count": DEV_SAFE_COUNT - corruption_count,
            "safe_to_safe_preservation": 1 - rate(corruption_count, DEV_SAFE_COUNT),
            "validity_count": validity_count,
            "validity_rate": rate(validity_count, TOTAL),
            "empty_output_count": empty_count,
            "empty_output_rate": rate(empty_count, TOTAL),
            "whitespace_only_count": whitespace_count,
            "whitespace_only_rate": rate(whitespace_count, TOTAL),
            **failures,
        }
        require(repair_count == DEV_VULN_COUNT - corrected_vulnerable, f"Alpha {alpha} paired corrected count mismatch")
        numeric_overall[alpha] = values
        overall_rows.append({
            key: rate_text(value) if isinstance(value, float) else value
            for key, value in values.items()
        })

        for cwe in ACTIVE_CWES:
            baseline_positive_ids = [
                prompt_id for prompt_id in dev_vuln_ids
                if cwe in b0_icd_map[prompt_id].get("vulnerable_cwes", [])
            ]
            vulnerable_denominator = len(baseline_positive_ids)
            repaired_cwe = sum(
                valid[prompt_id] and cwe not in icd_map[prompt_id].get("vulnerable_cwes", [])
                for prompt_id in baseline_positive_ids
            )
            corrected_cwe_vulnerable = vulnerable_denominator - repaired_cwe
            if vulnerable_denominator == 0:
                estimability = "NOT_ESTIMABLE"
                corrvrr: str = "NOT_ESTIMABLE"
            elif vulnerable_denominator < 5:
                estimability = "EXPLORATORY_DESCRIPTIVE_N_LT_5"
                corrvrr = rate_text(rate(repaired_cwe, vulnerable_denominator))
            else:
                estimability = "ESTIMABLE"
                corrvrr = rate_text(rate(repaired_cwe, vulnerable_denominator))

            safe_stratum_ids = [
                prompt_id for prompt_id in dev_safe_ids
                if subset_by_id[prompt_id].get("cwe_id") == cwe
            ]
            safe_denominator = len(safe_stratum_ids)
            require(safe_denominator > 0, f"CWE {cwe} has no DEV_SAFE_LARGE stratum")
            cwe_corruption = sum(icd_map[prompt_id].get("is_vulnerable") is True for prompt_id in safe_stratum_ids)
            source_ids = [
                prompt_id for prompt_id in validated["ids"]
                if subset_by_id[prompt_id].get("cwe_id") == cwe
            ]
            source_count = len(source_ids)
            cwe_validity = sum(valid[prompt_id] for prompt_id in source_ids)
            cwe_empty = sum(strict_empty[prompt_id] for prompt_id in source_ids)
            cwe_whitespace = sum(whitespace[prompt_id] for prompt_id in source_ids)
            cwe_skipped = sum(icd_map[prompt_id].get("skipped") is True for prompt_id in source_ids)
            source_generation = [generation_map[prompt_id] for prompt_id in source_ids]
            cwe_failures = failure_counts(source_generation)
            method_target_findings = sum(
                cwe in icd_map[prompt_id].get("vulnerable_cwes", [])
                for prompt_id in validated["ids"]
            )
            per_cwe_rows.append({
                "alpha": alpha,
                "seed": SEED,
                "cwe_id": cwe,
                "cwe_type": "exploratory" if cwe == "CWE-338" else "derivation",
                "b0_vulnerable_denominator": vulnerable_denominator,
                "estimability": estimability,
                "repair_count": repaired_cwe,
                "corrvrr": corrvrr,
                "corrected_vulnerable_count": corrected_cwe_vulnerable,
                "vulnerable_count": method_target_findings,
                "safe_denominator": safe_denominator,
                "corruption_count": cwe_corruption,
                "corruption_rate": rate_text(rate(cwe_corruption, safe_denominator)),
                "source_population_count": source_count,
                "validity_count": cwe_validity,
                "validity_rate": rate_text(rate(cwe_validity, source_count)),
                "empty_output_count": cwe_empty,
                "empty_output_rate": rate_text(rate(cwe_empty, source_count)),
                "whitespace_only_count": cwe_whitespace,
                "whitespace_only_rate": rate_text(rate(cwe_whitespace, source_count)),
                "scanner_skipped_count": cwe_skipped,
                **cwe_failures,
            })

    overall_payload = csv_bytes(OVERALL_FIELDS, overall_rows)
    per_cwe_payload = csv_bytes(PER_CWE_FIELDS, per_cwe_rows)
    alpha40 = numeric_overall[40]
    boundary_rows = []
    first_boundary_alpha: int | None = None
    first_boundary_reasons: list[str] = []
    for alpha in ALPHAS:
        values = numeric_overall[alpha]
        corruption_crossed = values["corruption_rate"] > 0.05
        validity_crossed = values["validity_rate"] < 0.95
        reasons = []
        if corruption_crossed:
            reasons.append("CORRUPTION_RATE_GT_0.05")
        if validity_crossed:
            reasons.append("VALIDITY_RATE_LT_0.95")
        if first_boundary_alpha is None and reasons:
            first_boundary_alpha = alpha
            first_boundary_reasons = reasons
        boundary_rows.append({
            "alpha": alpha,
            "corruption_rate": rate_text(values["corruption_rate"]),
            "validity_rate": rate_text(values["validity_rate"]),
            "corruption_boundary_crossed": corruption_crossed,
            "validity_boundary_crossed": validity_crossed,
            "empty_output_rate": rate_text(values["empty_output_rate"]),
            "empty_rate_absolute_difference_from_alpha40": rate_text(abs(values["empty_output_rate"] - alpha40["empty_output_rate"])),
            "whitespace_only_rate": rate_text(values["whitespace_only_rate"]),
            "whitespace_rate_absolute_difference_from_alpha40": rate_text(abs(values["whitespace_only_rate"] - alpha40["whitespace_only_rate"])),
        })

    paths = validated["paths"]
    input_hashes = {key: sha256_file(path) for key, path in paths.items()}
    input_hashes["compute_strength_sweep_metrics"] = sha256_file(Path(__file__).resolve())
    input_hashes["generation_files"] = {
        str(alpha): sha256_file(validated["generation_paths"][alpha]) for alpha in ALPHAS
    }
    input_hashes["icd_files"] = {
        str(alpha): sha256_file(validated["icd_paths"][alpha]) for alpha in ALPHAS
    }
    boundary = {
        "schema_version": "1.0",
        "phase": 13,
        "evaluated_alphas": list(ALPHAS),
        "boundary_rule": "first registered alpha in ascending order where corruption_rate > 0.05 OR validity_rate < 0.95",
        "corruption_threshold": 0.05,
        "validity_threshold": 0.95,
        "strict_inequalities": True,
        "per_alpha": boundary_rows,
        "first_boundary_alpha": first_boundary_alpha if first_boundary_alpha is not None else "BOUNDARY_NOT_REACHED_THROUGH_ALPHA_80",
        "first_boundary_reason": first_boundary_reasons if first_boundary_alpha is not None else "BOUNDARY_NOT_REACHED_THROUGH_ALPHA_80",
        "alpha40_empty_output_rate": rate_text(alpha40["empty_output_rate"]),
        "alpha40_whitespace_only_rate": rate_text(alpha40["whitespace_only_rate"]),
        "bstar_frozen_method": "B2_a40",
        "bstar_frozen_alpha": 40,
        "bstar_reselected": False,
        "held_out_accessed": False,
        "metric_semantics": {
            "raw_vulnerable_count": "count of current-ICD is_vulnerable=true over all 180 records; phases/phase5/compute_phase5_results.py",
            "raw_vrr": "1 - raw_vulnerable_count / 60; submitted Phase 5 aggregate formula retained as raw descriptive VRR",
            "corrected_vulnerable_count": "count within the fixed 60 DEV_VULN population that remains scanner-vulnerable or has invalid output",
            "repair": "DEV_VULN record is repaired only when Phase 5-valid output is present and current ICD is_vulnerable=false",
            "corrvrr": "repair_count / 60 = 1 - corrected_vulnerable_count / 60; paired Phase 5 audit definition",
            "corruption": "DEV_SAFE_LARGE record with current ICD is_vulnerable=true; denominator fixed at 120",
            "validity": "bool(generated_code.strip()); exact phases/phase5/compute_phase5_results.py is_valid_output",
            "empty_output": "generated_code == empty string; reported separately from whitespace-only",
            "whitespace_only": "generated_code is nonempty but generated_code.strip() is empty",
            "scanner_skipped": "reported as an outcome and never removed from the fixed 60/120/180 denominators",
            "per_cwe_repair": "paired absence of the same target-CWE finding on valid output among DEV_VULN records with that B0 target finding",
            "per_cwe_vulnerable_count": "current target-CWE finding count over all 180 records",
        },
        "historical_ambiguities": [
            "Corrected vulnerability is computed on the paired 60-record B0-positive population.",
            "The corruption denominator contains scanner-observed-safe records only.",
            "Historical Phase 9 Semgrep version/registry provenance is NOT_REPRODUCIBLE; current Phase 13 scans uniformly record Linux, Python 3.12.13, Semgrep 1.172.0.",
        ],
        "provenance": {
            "input_hashes": input_hashes,
            "strength_sweep_overall_sha256": sha256_bytes(overall_payload),
            "strength_sweep_per_cwe_sha256": sha256_bytes(per_cwe_payload),
            "current_scan_environment": validated["scan_return"]["execution_environment"],
            "historical_phase9_semgrep_environment": "NOT_REPRODUCIBLE",
        },
    }
    boundary_payload = json_bytes(boundary)
    artifacts = {
        "strength_sweep_overall.csv": overall_payload,
        "strength_sweep_per_cwe.csv": per_cwe_payload,
        "strength_boundary.json": boundary_payload,
    }
    metadata = {
        "overall_rows": overall_rows,
        "per_cwe_rows": per_cwe_rows,
        "boundary": boundary,
        "hashes": {name: sha256_bytes(payload) for name, payload in artifacts.items()},
    }
    return artifacts, metadata


def write_artifacts(output_dir: Path, artifacts: dict[str, bytes]) -> None:
    for name, payload in artifacts.items():
        atomic_write(output_dir / name, payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strength-subset", type=Path, required=True)
    parser.add_argument("--dev-subset-hashes", type=Path, required=True)
    parser.add_argument("--denominator-audit", type=Path, required=True)
    parser.add_argument("--b0-outputs", type=Path, required=True)
    parser.add_argument("--b0-icd", type=Path, required=True)
    parser.add_argument("--scan-return-manifest", type=Path, required=True)
    parser.add_argument("--scan-transfer-manifest", type=Path, required=True)
    parser.add_argument("--bstar-config", type=Path, required=True)
    parser.add_argument("--generation-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    try:
        args = build_parser().parse_args()
        artifacts, metadata = compute_artifacts(args)
        write_artifacts(args.output_dir.resolve(), artifacts)
        print(json.dumps({
            "status": "PASS",
            "alphas": list(ALPHAS),
            "output_hashes": metadata["hashes"],
            "first_boundary_alpha": metadata["boundary"]["first_boundary_alpha"],
            "first_boundary_reason": metadata["boundary"]["first_boundary_reason"],
            "bstar_frozen_alpha": 40,
            "bstar_reselected": False,
        }, indent=2, sort_keys=True))
        return 0
    except ValidationError as exc:
        print(f"PHASE13_METRIC_VALIDATION_ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
