#!/usr/bin/env python3
"""Shared frozen Phase 16 analysis loading and metric mechanics."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "revision/model1/phase16/outputs"
DENOMINATOR = ROOT / "revision/model1/phase2/phase2_denominator_audit.json"
TRANSFER = OUT / "phase16_scan_transfer_manifest.json"
RETURN = OUT / "phase16_scan_return_manifest.json"

EXPECTED_TRANSFER_SHA = "5abb65952868d1fbd91129917c2aa5fa5340587b396c46f78afcc18fea166629"
EXPECTED_SCANNER_SHA = "60cedd3d6ec2d19e5e7e5ba4d709230efdd5d9478047492fefda4dd88d9afbb7"

TOTAL = 575
SCANNER_UNIVERSE_COUNT = 392
PRIMARY = ("B0", "B1", "B2-alpha20", "B*")
TARGET_CWES = (
    "CWE-120", "CWE-327", "CWE-89", "CWE-338", "CWE-79",
    "CWE-125", "CWE-787", "CWE-190", "CWE-476", "CWE-22", "CWE-290",
)
DERIVATION_CWES = {"CWE-120", "CWE-125", "CWE-787", "CWE-190", "CWE-476", "CWE-89", "CWE-79", "CWE-327"}

CONDITIONS = (
    ("B0", 42, "NO_INTERVENTION_BASELINE", "REUSED", "outputs/phase9/baseline_test_outputs.json", "outputs/phase9/baseline_test_icd.json"),
    ("B0", 43, "NO_INTERVENTION_BASELINE", "NEW", "revision/model1/phase16/outputs/b0_seed43_outputs.json", "revision/model1/phase16/outputs/b0_seed43_icd.json"),
    ("B0", 44, "NO_INTERVENTION_BASELINE", "NEW", "revision/model1/phase16/outputs/b0_seed44_outputs.json", "revision/model1/phase16/outputs/b0_seed44_icd.json"),
    ("B1", 42, "METADATA_FREE_PROMPTING", "REUSED", "outputs/phase9/zeroshot_test_outputs.json", "outputs/phase9/zeroshot_test_icd.json"),
    ("B1", 43, "METADATA_FREE_PROMPTING", "NEW", "revision/model1/phase16/outputs/b1_seed43_outputs.json", "revision/model1/phase16/outputs/b1_seed43_icd.json"),
    ("B1", 44, "METADATA_FREE_PROMPTING", "NEW", "revision/model1/phase16/outputs/b1_seed44_outputs.json", "revision/model1/phase16/outputs/b1_seed44_icd.json"),
    ("B2-alpha20", 42, "ORACLE_CWE", "NEW", "revision/model1/phase16/outputs/b2_alpha20_seed42_outputs.json", "revision/model1/phase16/outputs/b2_alpha20_seed42_icd.json"),
    ("B2-alpha20", 43, "ORACLE_CWE", "NEW", "revision/model1/phase16/outputs/b2_alpha20_seed43_outputs.json", "revision/model1/phase16/outputs/b2_alpha20_seed43_icd.json"),
    ("B2-alpha20", 44, "ORACLE_CWE", "NEW", "revision/model1/phase16/outputs/b2_alpha20_seed44_outputs.json", "revision/model1/phase16/outputs/b2_alpha20_seed44_icd.json"),
    ("B*", 42, "ORACLE_CWE", "REUSED_CORRECTED_CANONICAL", "outputs/phase9/thea_static_test_outputs.json", "outputs/phase9/thea_static_test_icd.json"),
    ("B*", 43, "ORACLE_CWE", "NEW", "revision/model1/phase16/outputs/bbstar_seed43_outputs.json", "revision/model1/phase16/outputs/bbstar_seed43_icd.json"),
    ("B*", 44, "ORACLE_CWE", "NEW", "revision/model1/phase16/outputs/bbstar_seed44_outputs.json", "revision/model1/phase16/outputs/bbstar_seed44_icd.json"),
    ("B3-ungated", 42, "ORACLE_CWE", "SINGLE_SEED", "revision/model1/phase16/outputs/b3_ungated_seed42_outputs.json", "revision/model1/phase16/outputs/b3_ungated_seed42_icd.json"),
    ("B1-CWE", 42, "ORACLE_CWE", "SINGLE_SEED", "revision/model1/phase16/outputs/b1_cwe_seed42_outputs.json", "revision/model1/phase16/outputs/b1_cwe_seed42_icd.json"),
    ("RCI-1", 42, "METADATA_FREE", "SINGLE_SEED", "revision/model1/phase16/outputs/rci_1_seed42_outputs.json", "revision/model1/phase16/outputs/rci_1_seed42_icd.json"),
    ("CAA-CWE", 42, "ORACLE_CWE", "SINGLE_SEED", "revision/model1/phase16/outputs/caa_cwe_seed42_outputs.json", "revision/model1/phase16/outputs/caa_cwe_seed42_icd.json"),
    ("B3-PG", 42, "ORACLE_CWE", "PRESERVED_REFERENCE", "outputs/phase9/semantic_static_test_outputs.json", "outputs/phase9/semantic_static_test_icd.json"),
)


class ValidationError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise ValidationError(message)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_generation(path: Path) -> list[dict[str, Any]]:
    value = read_json(path)
    if isinstance(value, dict):
        value = value.get("final_outputs")
    require(isinstance(value, list) and all(isinstance(row, dict) for row in value), f"Malformed generation {path}")
    return value


def by_id(rows: list[dict[str, Any]], label: str) -> dict[int, dict[str, Any]]:
    result = {int(row["prompt_id"]): row for row in rows}
    require(len(result) == len(rows), f"Duplicate prompt ID in {label}")
    return result


def valid(row: dict[str, Any]) -> bool:
    return bool(str(row.get("generated_code", "")).strip())


def strict_empty(row: dict[str, Any]) -> bool:
    return row.get("generated_code", "") == ""


def whitespace_only(row: dict[str, Any]) -> bool:
    code = str(row.get("generated_code", ""))
    return bool(code) and not code.strip()


def failure(row: dict[str, Any]) -> bool:
    status = row.get("generation_status")
    return status not in (None, "COMPLETED", "SUCCESS", "WHITESPACE_ONLY")


def load_all() -> dict[str, Any]:
    require(sha256_path(TRANSFER) == EXPECTED_TRANSFER_SHA, "Scan-transfer hash mismatch")
    denominator = read_json(DENOMINATOR)
    require(denominator.get("verification_status") == "PASS", "Denominator audit is not PASS")
    heldout = denominator["heldout"]
    ids = [int(value) for value in heldout["total_prompt_ids"]]
    universe = [int(value) for value in heldout["scanner_eligible_prompt_ids"]]
    require(len(ids) == TOTAL and len(set(ids)) == TOTAL, "Held-out IDs malformed")
    require(len(universe) == SCANNER_UNIVERSE_COUNT, "Scanner universe count mismatch")

    transfer = read_json(TRANSFER)
    returned = read_json(RETURN)
    require(returned.get("status") == "SCANS_COMPLETE", "Scan return is not complete")
    require(returned.get("complete_count") == 13 and returned.get("total_scanner_records") == 7475, "Scan return totals mismatch")
    require(returned.get("transfer_manifest_sha256") == EXPECTED_TRANSFER_SHA, "Return/transfer linkage mismatch")
    require(returned.get("scanner_sha256") == EXPECTED_SCANNER_SHA, "Returned scanner hash mismatch")
    require(returned.get("security_metrics_computed") is False, "Colab return unexpectedly computed metrics")
    transfer_map = {(row["method"], int(row["seed"])): row for row in transfer["entries"]}
    return_map = {(row["method"], int(row["seed"])): row for row in returned["entries"]}

    data: dict[tuple[str, int], dict[str, Any]] = {}
    for method, seed, tier, disposition, generation_rel, scan_rel in CONDITIONS:
        generation_path = ROOT / generation_rel
        scan_path = ROOT / scan_rel
        require(generation_path.is_file() and scan_path.is_file(), f"Missing {method} seed {seed} artifact")
        generation = load_generation(generation_path)
        scan = read_json(scan_path)
        require(isinstance(scan, list), f"Malformed scan {scan_path}")
        require(len(generation) == len(scan) == TOTAL, f"{method} seed {seed} count mismatch")
        generation_ids = [int(row["prompt_id"]) for row in generation]
        scan_ids = [int(row["prompt_id"]) for row in scan]
        require(len(set(generation_ids)) == TOTAL and set(generation_ids) == set(ids),
                f"{method} seed {seed} generation population mismatch")
        require(scan_ids == generation_ids, f"{method} seed {seed} scan/generation order mismatch")
        if disposition == "NEW":
            require(generation_ids == ids, f"{method} seed {seed} new-generation frozen order mismatch")
        if disposition == "NEW":
            transfer_row = transfer_map[(method, seed)]
            return_row = return_map[(method, seed)]
            require(sha256_path(generation_path) == transfer_row["input_sha256"], f"{method} generation hash mismatch")
            require(sha256_path(scan_path) == return_row["icd_sha256"], f"{method} scan hash mismatch")
            require(return_row["scan_status"] == "COMPLETE", f"{method} scan incomplete")
        data[(method, seed)] = {
            "method": method, "seed": seed, "information_tier": tier, "disposition": disposition,
            "generation_path": generation_rel, "generation_sha256": sha256_path(generation_path),
            "scan_path": scan_rel, "scan_sha256": sha256_path(scan_path),
            "generation": generation, "generation_map": by_id(generation, f"{method} generation"),
            "scan": scan, "scan_map": by_id(scan, f"{method} scan"),
        }
    universe_set = set(universe)
    for seed in (42, 43, 44):
        observed = {prompt_id for prompt_id, row in data[("B0", seed)]["scan_map"].items() if row["skipped"] is False}
        require(observed == universe_set, f"B0 seed {seed} scanner universe differs from frozen 392 IDs")
    return {"ids": ids, "universe": universe, "data": data, "denominator": denominator,
            "transfer": transfer, "return": returned}


def baseline_sets(loaded: dict[str, Any], seed: int) -> tuple[set[int], set[int]]:
    scan = loaded["data"][("B0", seed)]["scan_map"]
    universe = set(loaded["universe"])
    vulnerable = {prompt_id for prompt_id in universe if scan[prompt_id]["is_vulnerable"] is True}
    return vulnerable, universe - vulnerable


def metric_values(loaded: dict[str, Any], method: str, seed: int) -> dict[str, Any]:
    item = loaded["data"][(method, seed)]
    generation = item["generation"]
    generation_map = item["generation_map"]
    scan = item["scan"]
    scan_map = item["scan_map"]
    vulnerable, safe = baseline_sets(loaded, seed)
    repaired = {prompt_id for prompt_id in vulnerable if valid(generation_map[prompt_id])
                and scan_map[prompt_id]["skipped"] is False and scan_map[prompt_id]["is_vulnerable"] is False}
    corrupted = {prompt_id for prompt_id in safe if scan_map[prompt_id]["skipped"] is False
                 and scan_map[prompt_id]["is_vulnerable"] is True}
    raw_vulnerable = sum(row["is_vulnerable"] is True for row in scan)
    valid_count = sum(valid(row) for row in generation)
    strict_count = sum(strict_empty(row) for row in generation)
    whitespace_count = sum(whitespace_only(row) for row in generation)
    failure_count = sum(failure(row) for row in generation)
    fallback_metadata_available = any("fallback_status" in row for row in generation)
    if method in ("B*", "B3-PG") and not fallback_metadata_available:
        fallback_count: int | str = "NOT_AVAILABLE"
    else:
        fallback_count = sum(row.get("fallback_status") not in (None, "NONE") for row in generation)
    route_counts = dict(sorted(Counter(str(row.get("route_status")) for row in generation).items()))
    return {
        "method": method, "seed": seed, "information_tier": item["information_tier"],
        "disposition": item["disposition"], "seed_status": "MULTI_SEED" if method in PRIMARY else "SINGLE_SEED",
        "total_prompts": TOTAL, "scanner_universe_count": len(loaded["universe"]),
        "scanner_eligible_count": sum(row["skipped"] is False for row in scan),
        "scanner_skipped_count": sum(row["skipped"] is True for row in scan),
        "b0_vulnerable_denominator": len(vulnerable), "b0_safe_denominator": len(safe),
        "raw_vulnerable_count": raw_vulnerable,
        "corrected_vulnerable_count": len(vulnerable) - len(repaired),
        "repair_count": len(repaired), "corrvrr": len(repaired) / len(vulnerable),
        "raw_vrr": 1 - raw_vulnerable / len(vulnerable),
        "corruption_count": len(corrupted), "corruption_rate": len(corrupted) / len(safe),
        "validity_count": valid_count, "validity_rate": valid_count / TOTAL,
        "strict_empty_count": strict_count, "whitespace_only_count": whitespace_count,
        "generation_failure_count": failure_count,
        "hook_failure_count": sum("HOOK" in str(row.get("generation_status", "")) for row in generation),
        "fallback_count": fallback_count,
        "fallback_count_provenance": "OBSERVED" if fallback_metadata_available or method in ("B0", "B1") else "NOT_AVAILABLE",
        "route_status_counts": json.dumps(route_counts, sort_keys=True),
        "generation_sha256": item["generation_sha256"], "scan_sha256": item["scan_sha256"],
        "repaired_ids": repaired, "corrupted_ids": corrupted,
    }


def rounded(value: float) -> str:
    return f"{value:.6f}"


def sample_sd(values: list[float]) -> float:
    require(len(values) >= 2, "sample SD needs at least two values")
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def percentile_type7(values: list[float], probability: float) -> float:
    require(values, "percentile requires values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])
