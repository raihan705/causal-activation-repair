#!/usr/bin/env python3
"""Build and freeze the nested Phase 13/14 development subsets."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
OUTPUT_DIR = ROOT / "revision/model1/phase13/outputs"

PATHS = {
    "denominator_audit": "revision/model1/phase2/phase2_denominator_audit.json",
    "development_prompts": "data/cyberseceval/dev_prompts.json",
    "development_b0_outputs": "outputs/phase2/baseline_dev_outputs.json",
    "development_b0_icd": "outputs/phase2/baseline_dev_icd.json",
}

EXPECTED_AUDIT_SHA256 = "75b5ca0a029e43a49a9048d17a94713c0003aad6f144352539c6b0ecede3f619"
SELECTION_SEED = 42
FROZEN_MODEL1_SAFE_CWES = {"CWE-120", "CWE-327", "CWE-89", "CWE-338"}


def fail(message: str) -> None:
    raise RuntimeError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="")
    os.replace(temporary, path)


def index_unique(records: list[dict[str, Any]], name: str) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for record in records:
        prompt_id = int(record["prompt_id"])
        require(prompt_id not in result, f"duplicate prompt ID {prompt_id} in {name}")
        result[prompt_id] = record
    return result


def source_artifact_hash(audit: dict[str, Any], relative_path: str) -> str:
    matches = [item for item in audit["source_artifacts"] if item.get("path") == relative_path]
    require(len(matches) == 1, f"denominator audit has no unique source entry for {relative_path}")
    return matches[0]["sha256"]


def hamilton_alloc(capacities: dict[str, int], target: int) -> dict[str, int]:
    """Proportional largest-remainder allocation with deterministic tie breaks."""
    total = sum(capacities.values())
    require(0 <= target <= total, f"allocation target {target} exceeds pool {total}")
    quotas = {key: target * value / total for key, value in capacities.items()}
    allocation = {key: math.floor(quota) for key, quota in quotas.items()}
    remaining = target - sum(allocation.values())
    order = sorted(capacities, key=lambda key: (-(quotas[key] - allocation[key]), key))
    for key in order[:remaining]:
        allocation[key] += 1
    require(sum(allocation.values()) == target, "Hamilton allocation did not reach its target")
    require(all(allocation[key] <= capacities[key] for key in capacities), "allocation exceeds a stratum")
    return allocation


def main() -> None:
    audit_path = ROOT / PATHS["denominator_audit"]
    require(sha256_path(audit_path) == EXPECTED_AUDIT_SHA256, "Phase 2 denominator-audit hash mismatch")
    audit = read_json(audit_path)
    require(audit.get("verification_status") == "PASS", "Phase 2 denominator audit is not PASS")
    require(audit.get("blockers") == [], "Phase 2 denominator audit has blockers")

    inputs: dict[str, dict[str, Any]] = {}
    loaded: dict[str, Any] = {"denominator_audit": audit}
    for name, relative in PATHS.items():
        path = ROOT / relative
        require(path.is_file(), f"missing required source {relative}")
        actual_hash = sha256_path(path)
        expected_hash = EXPECTED_AUDIT_SHA256 if name == "denominator_audit" else source_artifact_hash(audit, relative)
        require(actual_hash == expected_hash, f"source hash mismatch for {relative}")
        inputs[name] = {"path": relative, "bytes": path.stat().st_size, "sha256": actual_hash}
        if name != "denominator_audit":
            loaded[name] = read_json(path)

    prompts: list[dict[str, Any]] = loaded["development_prompts"]
    b0_outputs: list[dict[str, Any]] = loaded["development_b0_outputs"]
    b0_icd: list[dict[str, Any]] = loaded["development_b0_icd"]
    prompt_index = index_unique(prompts, "development prompts")
    output_index = index_unique(b0_outputs, "development B0 outputs")
    icd_index = index_unique(b0_icd, "development B0 ICD")
    source_order = [int(row["prompt_id"]) for row in prompts]
    require(len(source_order) == int(audit["development"]["total_count"]), "development source count differs from verified audit")
    require(set(source_order) == set(output_index) == set(icd_index), "development source/output/ICD populations differ")
    require([int(row["prompt_id"]) for row in b0_outputs] == source_order, "B0 output order differs from development source")
    require([int(row["prompt_id"]) for row in b0_icd] == source_order, "B0 ICD order differs from development source")

    for source in prompts:
        prompt_id = int(source["prompt_id"])
        output = output_index[prompt_id]
        scan = icd_index[prompt_id]
        require(output.get("prompt_text") == source.get("test_case_prompt"), f"B0 prompt rendering differs for {prompt_id}")
        require(output.get("language") == source.get("language"), f"B0 language differs for {prompt_id}")
        require(output.get("cwe_id") == source.get("cwe_identifier"), f"B0 CWE differs for {prompt_id}")
        require(scan.get("language") == source.get("language", "").lower(), f"B0 ICD language differs for {prompt_id}")
        require(scan.get("cwe_id") == source.get("cwe_identifier"), f"B0 ICD CWE differs for {prompt_id}")

    development = audit["development"]
    verified_vuln = {int(value) for value in development["b0_vulnerable_eligible_prompt_ids"]}
    verified_safe = {int(value) for value in development["b0_safe_eligible_prompt_ids"]}
    verified_eligible = {int(value) for value in development["scanner_eligible_prompt_ids"]}
    heldout_ids = {int(value) for value in audit["heldout"]["total_prompt_ids"]}
    n_vuln = int(development["b0_vulnerable_eligible_count"])
    recorded_n_vuln = int(audit["recorded_counts_section_2_10"]["development_b0_vulnerable_eligible"])
    require(n_vuln == len(verified_vuln), "verified vulnerable count/list mismatch")
    require(n_vuln == recorded_n_vuln, f"unexpected verified N_vuln={n_vuln}; recorded audit value is {recorded_n_vuln}")
    require(verified_vuln <= verified_eligible, "verified vulnerable IDs include scanner-ineligible prompts")
    require(verified_safe <= verified_eligible, "verified safe IDs include scanner-ineligible prompts")
    require(verified_vuln.isdisjoint(verified_safe), "verified vulnerable and safe sets overlap")
    require(set(source_order).isdisjoint(heldout_ids), "development and held-out IDs overlap")

    for prompt_id in verified_vuln:
        scan = icd_index[prompt_id]
        require(scan.get("skipped") is False and scan.get("is_vulnerable") is True, f"DEV_VULN predicate fails for {prompt_id}")
    for prompt_id in verified_safe:
        scan = icd_index[prompt_id]
        require(scan.get("skipped") is False and scan.get("is_vulnerable") is False, f"verified safe predicate fails for {prompt_id}")

    active_safe_pool = [
        prompt_id for prompt_id in source_order
        if prompt_id in verified_safe
        and prompt_index[prompt_id]["cwe_identifier"] in FROZEN_MODEL1_SAFE_CWES
    ]
    require(
        len(active_safe_pool) >= 2 * n_vuln,
        "frozen four-CWE scanner-supported safe pool is insufficient",
    )

    strata: dict[str, list[int]] = defaultdict(list)
    for prompt_id in active_safe_pool:
        source = prompt_index[prompt_id]
        stratum = f"{source['cwe_identifier']}|{source['language']}"
        strata[stratum].append(prompt_id)
    capacities = {key: len(value) for key, value in sorted(strata.items())}
    cwe_capacities = {
        cwe: sum(len(values) for key, values in strata.items() if key.startswith(f"{cwe}|"))
        for cwe in sorted(FROZEN_MODEL1_SAFE_CWES)
    }
    require(set(cwe_capacities) == FROZEN_MODEL1_SAFE_CWES, "four-CWE capacity inventory differs")
    require(sum(cwe_capacities.values()) == len(active_safe_pool), "safe-pool capacity inventory differs")
    large_allocation = hamilton_alloc(capacities, 2 * n_vuln)

    rng = random.Random(SELECTION_SEED)
    shuffled_by_stratum: dict[str, list[int]] = {}
    large_by_stratum: dict[str, list[int]] = {}
    for key in sorted(strata):
        shuffled = list(strata[key])
        rng.shuffle(shuffled)
        shuffled_by_stratum[key] = shuffled
        large_by_stratum[key] = shuffled[:large_allocation[key]]

    small_capacities = {key: len(value) for key, value in large_by_stratum.items()}
    small_allocation = hamilton_alloc(small_capacities, n_vuln)
    small_by_stratum = {
        key: large_by_stratum[key][:small_allocation[key]] for key in sorted(large_by_stratum)
    }
    safe_large_set = {prompt_id for values in large_by_stratum.values() for prompt_id in values}
    safe_small_set = {prompt_id for values in small_by_stratum.values() for prompt_id in values}
    require(len(safe_large_set) == 2 * n_vuln, "DEV_SAFE_LARGE count differs")
    require(len(safe_small_set) == n_vuln, "DEV_SAFE_SMALL count differs")
    require(safe_small_set < safe_large_set, "DEV_SAFE_SMALL is not a strict subset of DEV_SAFE_LARGE")
    require(verified_vuln.isdisjoint(safe_large_set), "DEV_VULN overlaps DEV_SAFE_LARGE")
    require(safe_large_set <= verified_safe, "DEV_SAFE_LARGE includes a non-safe ID")
    require(safe_large_set.isdisjoint(heldout_ids), "safe subset contains held-out IDs")
    require(
        all(prompt_index[prompt_id]["cwe_identifier"] in FROZEN_MODEL1_SAFE_CWES for prompt_id in safe_large_set),
        "safe subset includes a CWE outside the frozen Model-1 categories",
    )

    dev_vuln_order = [prompt_id for prompt_id in source_order if prompt_id in verified_vuln]
    safe_large_order = [prompt_id for prompt_id in source_order if prompt_id in safe_large_set]
    safe_small_order = [prompt_id for prompt_id in source_order if prompt_id in safe_small_set]
    strength_ids = [prompt_id for prompt_id in source_order if prompt_id in verified_vuln | safe_large_set]
    baseline_ids = [prompt_id for prompt_id in source_order if prompt_id in verified_vuln | safe_small_set]
    require(len(dev_vuln_order) == n_vuln and len(strength_ids) == 3 * n_vuln and len(baseline_ids) == 2 * n_vuln, "subset union counts differ")
    require(len(set(strength_ids)) == len(strength_ids) and len(set(baseline_ids)) == len(baseline_ids), "subset duplicate IDs")

    stratum_rows = []
    for key in sorted(strata):
        cwe, language = key.split("|", 1)
        stratum_rows.append({
            "stratum": key,
            "cwe_id": cwe,
            "language": language,
            "eligible_safe_pool_count": capacities[key],
            "dev_safe_large_count": large_allocation[key],
            "dev_safe_small_count": small_allocation[key],
            "dev_safe_large_prompt_ids_source_order": [prompt_id for prompt_id in source_order if prompt_id in set(large_by_stratum[key])],
            "dev_safe_small_prompt_ids_source_order": [prompt_id for prompt_id in source_order if prompt_id in set(small_by_stratum[key])],
        })

    def record(prompt_id: int, safe_population: str | None) -> dict[str, Any]:
        source = prompt_index[prompt_id]
        scan = icd_index[prompt_id]
        output = output_index[prompt_id]
        is_vulnerable = prompt_id in verified_vuln
        stratum = f"{source['cwe_identifier']}|{source['language']}"
        return {
            "source_index": source_order.index(prompt_id),
            "prompt_id": prompt_id,
            "population_type": "DEV_VULN" if is_vulnerable else safe_population,
            "cwe_id": source["cwe_identifier"],
            "language": source["language"],
            "stratum": stratum,
            "b0_is_vulnerable": bool(scan["is_vulnerable"]),
            "scanner_eligible": not bool(scan["skipped"]),
            "selection_seed": None if is_vulnerable else SELECTION_SEED,
            "prompt_text_sha256": hashlib.sha256(source["test_case_prompt"].encode("utf-8")).hexdigest(),
            "b0_output_sha256": hashlib.sha256(output["generated_code"].encode("utf-8")).hexdigest(),
            "source_prompt": source,
        }

    construction = {
        "n_vuln_derived_from_audit": n_vuln,
        "safe_large_ratio": 2,
        "safe_small_ratio": 1,
        "safe_sampling_seed": SELECTION_SEED,
        "safe_strata_definition": "frozen Model-1 scanner-supported CWE label x programming language",
        "active_scanner_supported_cwes": sorted(FROZEN_MODEL1_SAFE_CWES),
        "active_safe_candidate_pool_count": len(active_safe_pool),
        "active_safe_candidate_pool_count_by_cwe": cwe_capacities,
        "active_safe_candidate_pool_count_by_stratum": capacities,
        "category_authority": "revision/REVISION_PLAN.md Section 2.1",
        "allocation_method": "Hamilton proportional largest remainder; lexical stratum tie-break",
        "sampling_method": "one seeded shuffle per lexical stratum from one Random(42) stream; DEV_SAFE_SMALL is a prefix selection within frozen DEV_SAFE_LARGE, never independently sampled",
        "source_order_policy": "all emitted prompt IDs and records follow development source order",
    }
    checks = {
        "denominator_audit_hash_verified": True,
        "all_ids_exist_in_development": True,
        "all_dev_vuln_scanner_eligible_and_b0_vulnerable": True,
        "all_safe_scanner_eligible_and_b0_safe": True,
        "all_safe_source_cwes_in_frozen_model1_set": True,
        "safe_source_cwe_22_absent": True,
        "safe_source_cwe_79_absent": True,
        "no_duplicate_ids": True,
        "dev_vuln_safe_disjoint": True,
        "dev_safe_small_strict_subset_of_dev_safe_large": True,
        "no_heldout_prompt_ids": True,
        "source_metadata_and_b0_prompt_rendering_preserved": True,
        "recorded_n_vuln_matches_recomputed": True,
    }

    strength_manifest = {
        "schema_version": "1.0",
        "phase": 13,
        "manifest_name": "PHASE13_STRENGTH_SUBSET",
        "split": "DEVELOPMENT",
        "source_files": inputs,
        "construction": construction,
        "counts": {"dev_vuln": n_vuln, "dev_safe_large": len(safe_large_order), "strength_subset": len(strength_ids)},
        "prompt_ids_source_order": strength_ids,
        "dev_vuln_prompt_ids_source_order": dev_vuln_order,
        "dev_safe_large_prompt_ids_source_order": safe_large_order,
        "records": [record(prompt_id, "DEV_SAFE_LARGE") for prompt_id in strength_ids],
        "safe_strata": stratum_rows,
        "nesting": {"dev_safe_small_is_strict_subset_of_dev_safe_large": True, "baseline_selection_manifest": "revision/model1/phase13/outputs/baseline_selection_subset.json"},
        "validation_checks": checks,
        "validation_status": "PASS",
    }
    strength_manifest["manifest_content_sha256"] = canonical_hash(strength_manifest)

    baseline_manifest = {
        "schema_version": "1.0",
        "phase": 13,
        "consumer_phase": 14,
        "manifest_name": "PHASE14_BASELINE_SELECTION_SUBSET",
        "split": "DEVELOPMENT",
        "source_files": inputs,
        "construction": construction,
        "counts": {"dev_vuln": n_vuln, "dev_safe_small": len(safe_small_order), "baseline_selection_subset": len(baseline_ids)},
        "prompt_ids_source_order": baseline_ids,
        "dev_vuln_prompt_ids_source_order": dev_vuln_order,
        "dev_safe_small_prompt_ids_source_order": safe_small_order,
        "records": [record(prompt_id, "DEV_SAFE_SMALL") for prompt_id in baseline_ids],
        "safe_strata": stratum_rows,
        "nesting": {"dev_safe_small_is_strict_subset_of_dev_safe_large": True, "dev_safe_large_prompt_ids_source_order": safe_large_order, "phase14_must_reuse_exactly_without_resampling": True},
        "validation_checks": checks,
        "validation_status": "PASS",
    }
    baseline_manifest["manifest_content_sha256"] = canonical_hash(baseline_manifest)

    strength_path = OUTPUT_DIR / "strength_subset_manifest.json"
    baseline_path = OUTPUT_DIR / "baseline_selection_subset.json"
    hashes_path = OUTPUT_DIR / "dev_subset_hashes.json"
    atomic_json(strength_path, strength_manifest)
    atomic_json(baseline_path, baseline_manifest)
    hash_record = {
        "schema_version": "1.0",
        "phase": 13,
        "sampling_seed": SELECTION_SEED,
        "n_vuln": n_vuln,
        "component_hashes": {
            "dev_vuln_prompt_ids_source_order": canonical_hash(dev_vuln_order),
            "dev_safe_large_prompt_ids_source_order": canonical_hash(safe_large_order),
            "dev_safe_small_prompt_ids_source_order": canonical_hash(safe_small_order),
            "strength_subset_prompt_ids_source_order": canonical_hash(strength_ids),
            "baseline_selection_prompt_ids_source_order": canonical_hash(baseline_ids),
        },
        "manifest_files": {
            "strength_subset_manifest": {"path": "revision/model1/phase13/outputs/strength_subset_manifest.json", "bytes": strength_path.stat().st_size, "sha256": sha256_path(strength_path), "record_count": len(strength_ids)},
            "baseline_selection_subset": {"path": "revision/model1/phase13/outputs/baseline_selection_subset.json", "bytes": baseline_path.stat().st_size, "sha256": sha256_path(baseline_path), "record_count": len(baseline_ids)},
        },
        "nesting_validation": {"dev_safe_small_strict_subset_dev_safe_large": True, "dev_safe_small_count": len(safe_small_set), "dev_safe_large_count": len(safe_large_set), "overlap_count": len(safe_small_set & safe_large_set)},
        "source_hashes": inputs,
        "validation_status": "PASS",
    }
    atomic_json(hashes_path, hash_record)

    print("validation_status=PASS")
    print(f"N_vuln={n_vuln}")
    print(f"DEV_VULN={len(dev_vuln_order)}")
    print(f"DEV_SAFE_LARGE={len(safe_large_order)}")
    print(f"DEV_SAFE_SMALL={len(safe_small_order)}")
    print(f"corrected_safe_candidate_pool={len(active_safe_pool)}")
    for cwe, count in cwe_capacities.items():
        print(f"safe_candidate_pool[{cwe}]={count}")
    for stratum, count in capacities.items():
        print(f"safe_candidate_pool[{stratum}]={count}")
    print(f"strength_subset={len(strength_ids)}")
    print(f"baseline_selection_subset={len(baseline_ids)}")
    print("nesting=STRICT_SUBSET_PASS")
    print(f"strength_manifest_sha256={sha256_path(strength_path)}")
    print(f"baseline_manifest_sha256={sha256_path(baseline_path)}")
    print(f"hash_record_sha256={sha256_path(hashes_path)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1)
