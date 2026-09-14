#!/usr/bin/env python3
"""Build the frozen Phase 18 50-prompt timing subset."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "revision/model1/phase18/outputs"
DESTINATION = OUT / "timing_subset_manifest.json"
APPROVAL = OUT / "phase18_capacity_capping_approval.json"
DEV_PROMPTS = ROOT / "data/cyberseceval/dev_prompts.json"
B0_OUTPUTS = ROOT / "outputs/phase2/baseline_dev_outputs.json"
B0_SCAN = ROOT / "outputs/phase2/baseline_dev_icd.json"
DENOMINATOR = ROOT / "revision/model1/phase2/phase2_denominator_audit.json"

EXPECTED_HASHES = {
    DEV_PROMPTS: "18c0e78c3589c6c0ef7a4972d116094aa973bce8e8d06138c8844ca0c66a1680",
    B0_OUTPUTS: "0a33b488b88a0414fc20182602b622fd029e79e86a27c63ff54fe307926a92a4",
    B0_SCAN: "8af54a450b133a315b5578357e07b8201ddf2f431f8fee04874ef1f4a18974d6",
    DENOMINATOR: "75b5ca0a029e43a49a9048d17a94713c0003aad6f144352539c6b0ecede3f619",
}
ACTIVE_CWES = ("CWE-120", "CWE-327", "CWE-89", "CWE-338")
STATUS_QUOTAS = {"B0_VULNERABLE": 25, "B0_SAFE": 25}
SEED = 42


class SubsetError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SubsetError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def lexical(stratum: tuple[str, str]) -> tuple[str, str]:
    return stratum


def hamilton_minimum_one_with_capacity(populations: dict[tuple[str, str], int], quota: int) -> dict[str, Any]:
    require(populations and all(value > 0 for value in populations.values()), "Empty or invalid stratum population")
    require(len(populations) <= quota, f"{len(populations)} nonempty strata exceed quota {quota}")
    require(sum(populations.values()) >= quota, "Total stratum capacity is below quota")

    allocations = {key: 1 for key in populations}
    initial_minimum = dict(allocations)
    remaining = quota - len(populations)
    total_weight = sum(populations.values())
    raw = {key: remaining * populations[key] / total_weight for key in populations}
    floors = {key: math.floor(raw[key]) for key in populations}
    for key, value in floors.items():
        allocations[key] += value
    remainder_slots = quota - sum(allocations.values())
    order = sorted(populations, key=lambda key: (-(raw[key] - floors[key]), *lexical(key)))
    for key in order[:remainder_slots]:
        allocations[key] += 1
    initial_hamilton = dict(allocations)

    overflow = 0
    for key in populations:
        if allocations[key] > populations[key]:
            overflow += allocations[key] - populations[key]
            allocations[key] = populations[key]

    iterations: list[dict[str, Any]] = []
    while overflow:
        unsaturated = {key: populations[key] for key in populations if allocations[key] < populations[key]}
        capacity = {key: populations[key] - allocations[key] for key in unsaturated}
        require(sum(capacity.values()) >= overflow,
                f"Outstanding overflow {overflow} exceeds unsaturated capacity {sum(capacity.values())}")
        iteration_start = overflow
        weight_total = sum(unsaturated.values())
        quotas = {key: iteration_start * unsaturated[key] / weight_total for key in unsaturated}
        floor_add = {key: min(math.floor(quotas[key]), capacity[key]) for key in unsaturated}
        for key, value in floor_add.items():
            allocations[key] += value
            overflow -= value

        remainder_order = sorted(
            unsaturated,
            key=lambda key: (-(quotas[key] - math.floor(quotas[key])), *lexical(key)),
        )
        remainder_awards: list[str] = []
        for key in remainder_order:
            if overflow == 0:
                break
            if allocations[key] < populations[key]:
                allocations[key] += 1
                overflow -= 1
                remainder_awards.append(f"{key[0]}|{key[1]}")
        require(overflow < iteration_start, "Capacity redistribution made no progress")
        iterations.append({
            "overflow_at_start": iteration_start,
            "unsaturated_original_weights": {f"{key[0]}|{key[1]}": unsaturated[key] for key in sorted(unsaturated)},
            "floor_additions": {f"{key[0]}|{key[1]}": floor_add[key] for key in sorted(floor_add)},
            "remainder_awards": remainder_awards,
            "overflow_at_end": overflow,
        })

    require(sum(allocations.values()) == quota, "Final allocation total mismatch")
    require(all(1 <= allocations[key] <= populations[key] for key in populations), "Final allocation infeasible")
    return {
        "minimum_one": initial_minimum,
        "initial_hamilton": initial_hamilton,
        "final": allocations,
        "redistribution_iterations": iterations,
    }


def main() -> int:
    for path, expected in EXPECTED_HASHES.items():
        require(path.is_file(), f"Missing source: {path}")
        require(sha256(path) == expected, f"Source hash mismatch: {path}")
    require(APPROVAL.is_file(), "Capacity-capping approval is missing")
    approval = read_json(APPROVAL)
    require(approval.get("status") == "APPROVED_BEFORE_TIMING_AND_SAMPLING", "Approval status mismatch")
    require(approval.get("prompt_ids_sampled_before_approval") is False, "Approval chronology mismatch")
    require(approval.get("timing_started_before_approval") is False, "Timing preceded approval")

    prompts = read_json(DEV_PROMPTS)
    outputs = read_json(B0_OUTPUTS)
    scan = read_json(B0_SCAN)
    denominator = read_json(DENOMINATOR)
    require(len(prompts) == len(outputs) == len(scan) == 1341, "Development artifact count mismatch")
    prompt_ids = [int(row["prompt_id"]) for row in prompts]
    require(len(set(prompt_ids)) == 1341, "Development prompt IDs are not unique")
    prompt_by_id = {int(row["prompt_id"]): row for row in prompts}
    output_by_id = {int(row["prompt_id"]): row for row in outputs}
    scan_by_id = {int(row["prompt_id"]): row for row in scan}
    require(set(prompt_by_id) == set(output_by_id) == set(scan_by_id), "Development populations do not align")

    eligible = set(map(int, denominator["development"]["scanner_eligible_prompt_ids"]))
    vulnerable = set(map(int, denominator["development"]["b0_vulnerable_eligible_prompt_ids"]))
    safe = set(map(int, denominator["development"]["b0_safe_eligible_prompt_ids"]))
    require(len(eligible) == 923 and len(vulnerable) == 60 and len(safe) == 863, "Frozen denominator counts mismatch")
    require(vulnerable | safe == eligible and not (vulnerable & safe), "Frozen status partitions mismatch")
    require(all(scan_by_id[prompt_id]["skipped"] is False for prompt_id in eligible), "Eligible record is scanner-skipped")
    require(all(scan_by_id[prompt_id]["is_vulnerable"] is True for prompt_id in vulnerable), "Vulnerable status mismatch")
    require(all(scan_by_id[prompt_id]["is_vulnerable"] is False for prompt_id in safe), "Safe status mismatch")

    source_position = {prompt_id: index for index, prompt_id in enumerate(prompt_ids)}
    status_ids = {"B0_VULNERABLE": vulnerable, "B0_SAFE": safe}
    selected_ids: set[int] = set()
    allocation_rows: list[dict[str, Any]] = []
    allocation_audit: dict[str, Any] = {}

    for status in ("B0_VULNERABLE", "B0_SAFE"):
        strata: dict[tuple[str, str], list[int]] = defaultdict(list)
        for prompt_id in status_ids[status]:
            row = prompt_by_id[prompt_id]
            cwe = str(row["cwe_identifier"])
            if cwe in ACTIVE_CWES:
                strata[(cwe, str(row["language"]).lower())].append(prompt_id)
        for values in strata.values():
            values.sort(key=lambda prompt_id: (source_position[prompt_id], prompt_id))
        populations = {key: len(values) for key, values in strata.items()}
        audit = hamilton_minimum_one_with_capacity(populations, STATUS_QUOTAS[status])
        allocation_audit[status] = {
            "quota": STATUS_QUOTAS[status],
            "active_population": sum(populations.values()),
            "nonempty_stratum_count": len(strata),
            "redistribution_iterations": audit["redistribution_iterations"],
        }
        for key in sorted(strata):
            count = audit["final"][key]
            rng = random.Random(SEED)
            chosen = rng.sample(strata[key], count)
            require(len(chosen) == count and len(set(chosen)) == count, "Within-stratum sample mismatch")
            selected_ids.update(chosen)
            allocation_rows.append({
                "b0_status": status, "cwe_identifier": key[0], "language": key[1],
                "stratum": f"{status}|{key[0]}|{key[1]}", "stratum_population": populations[key],
                "minimum_one_count": audit["minimum_one"][key],
                "initial_hamilton_count": audit["initial_hamilton"][key],
                "allocated_stratum_count": count, "selected_prompt_ids": sorted(chosen, key=source_position.get),
            })

    require(len(selected_ids) == 50, f"Expected 50 selected IDs, got {len(selected_ids)}")
    ordered_ids = sorted(selected_ids, key=source_position.get)
    records = []
    allocation_map = {(row["b0_status"], row["cwe_identifier"], row["language"]): row for row in allocation_rows}
    for prompt_id in ordered_ids:
        prompt = prompt_by_id[prompt_id]
        status = "B0_VULNERABLE" if prompt_id in vulnerable else "B0_SAFE"
        key = (status, str(prompt["cwe_identifier"]), str(prompt["language"]).lower())
        allocation = allocation_map[key]
        prompt_text = str(prompt["test_case_prompt"])
        records.append({
            "prompt_id": prompt_id,
            "original_source_position": source_position[prompt_id],
            "b0_status": status,
            "b0_is_vulnerable": status == "B0_VULNERABLE",
            "scanner_eligible": True,
            "cwe_identifier": key[1],
            "language": key[2],
            "stratum": allocation["stratum"],
            "stratum_population": allocation["stratum_population"],
            "allocated_stratum_count": allocation["allocated_stratum_count"],
            "selection_seed": SEED,
            "prompt_text": prompt_text,
            "prompt_text_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
            "b0_output_sha256": hashlib.sha256(str(output_by_id[prompt_id].get("generated_code", "")).encode("utf-8")).hexdigest(),
        })

    status_counts = Counter(row["b0_status"] for row in records)
    require(status_counts == Counter(STATUS_QUOTAS), f"Status quota mismatch: {status_counts}")
    require(all(row["allocated_stratum_count"] <= row["stratum_population"] for row in records), "Capacity violation")
    require(all(row["allocated_stratum_count"] >= 1 for row in records), "Unrepresented nonempty stratum")
    require(len({row["stratum"] for row in records}) == len(allocation_rows), "A nonempty stratum is unrepresented")

    by_status_cwe_language = [
        {"b0_status": key[0], "cwe_identifier": key[1], "language": key[2], "selected_count": value}
        for key, value in sorted(Counter((row["b0_status"], row["cwe_identifier"], row["language"]) for row in records).items())
    ]
    manifest = {
        "schema_version": "phase18_timing_subset_manifest_v1",
        "phase": 18,
        "status": "FROZEN_BEFORE_TIMING",
        "split": "DEVELOPMENT",
        "timing_started_before_freeze": False,
        "model_loaded_before_freeze": False,
        "scanner_run_before_freeze": False,
        "prior_protocol_amendment_sha256": approval["prior_protocol_amendment_sha256"],
        "capacity_capping_approval_path": str(APPROVAL.relative_to(ROOT)).replace("\\", "/"),
        "capacity_capping_approval_sha256": sha256(APPROVAL),
        "source_paths": {path.name: str(path.relative_to(ROOT)).replace("\\", "/") for path in EXPECTED_HASHES},
        "source_hashes": {path.name: expected for path, expected in EXPECTED_HASHES.items()},
        "universe_counts": {"development_total": 1341, "scanner_eligible": 923,
                            "b0_vulnerable": 60, "b0_safe": 863},
        "active_cwe_labels": list(ACTIVE_CWES),
        "status_quotas": STATUS_QUOTAS,
        "allocation_algorithm": {
            "strata": "cwe_identifier x language independently within B0 status",
            "minimum_one": True,
            "hamilton_weights": "original stratum population",
            "capacity_capping": True,
            "overflow_redistribution": "iterative Hamilton over unsaturated strata with original population weights",
            "tie_break": ["cwe_identifier", "language"],
        },
        "allocation_audit": allocation_audit,
        "allocation_table": allocation_rows,
        "selected_counts_by_status_cwe_language": by_status_cwe_language,
        "selection": {"within_stratum_sort": "original source position then prompt_id",
                      "rng": "new random.Random(42) per stratum", "without_replacement": True,
                      "final_order": "original development source order"},
        "prompt_count": 50,
        "ordered_prompt_ids": ordered_ids,
        "ordered_prompt_ids_sha256": canonical_sha256(ordered_ids),
        "warmup_prompt_id": ordered_ids[0],
        "records": records,
        "validation": {
            "exactly_25_vulnerable": status_counts["B0_VULNERABLE"] == 25,
            "exactly_25_safe": status_counts["B0_SAFE"] == 25,
            "grand_total_50": len(records) == 50,
            "all_records_scanner_eligible": all(row["scanner_eligible"] for row in records),
            "all_nonempty_strata_represented": len({row["stratum"] for row in records}) == len(allocation_rows),
            "no_allocation_exceeds_population": all(row["allocated_stratum_count"] <= row["stratum_population"] for row in records),
            "no_duplicate_ids": len(set(ordered_ids)) == 50,
            "source_order_restored": ordered_ids == sorted(ordered_ids, key=source_position.get),
            "no_heldout_prompts": True,
        },
    }
    expected_vulnerable = {
        (row["cwe_identifier"], row["language"]): row["allocation"]
        for row in approval["required_vulnerable_result"]
    }
    actual_vulnerable = {
        (row["cwe_identifier"], row["language"]): row["allocated_stratum_count"]
        for row in allocation_rows if row["b0_status"] == "B0_VULNERABLE"
    }
    require(actual_vulnerable == expected_vulnerable,
            f"Capacity algorithm does not reproduce approved vulnerable allocation: {actual_vulnerable}")
    write_json(DESTINATION, manifest)
    print(json.dumps({
        "status": "FROZEN_BEFORE_TIMING", "manifest_path": str(DESTINATION.relative_to(ROOT)),
        "manifest_sha256": sha256(DESTINATION), "prompt_count": len(records),
        "ordered_prompt_ids_sha256": manifest["ordered_prompt_ids_sha256"],
        "warmup_prompt_id": manifest["warmup_prompt_id"],
        "ordered_prompt_ids": ordered_ids,
        "allocation_table": by_status_cwe_language,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
