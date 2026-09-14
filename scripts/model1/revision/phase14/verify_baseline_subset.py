#!/usr/bin/env python3
"""Deterministically verify the frozen Phase 14 development subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


EXPECTED_HASHES = {
    "baseline_subset": "3f023091c601fd6b4ffe8c074ad927f4bd8e32d211e0a98d81d7f486aa355a0d",
    "strength_subset": "f8781a4532a0d58840a89652342fb88eeda7480cf42924ae1ee451f0b7b898c7",
    "dev_subset_hashes": "8266cd3a95e025a96cb9a7cf0252bbcbef20954ad647031bdee0c59344493010",
    "denominator_audit": "75b5ca0a029e43a49a9048d17a94713c0003aad6f144352539c6b0ecede3f619",
    "development_prompts": "18c0e78c3589c6c0ef7a4972d116094aa973bce8e8d06138c8844ca0c66a1680",
    "b0_outputs": "0a33b488b88a0414fc20182602b622fd029e79e86a27c63ff54fe307926a92a4",
    "b0_icd": "8af54a450b133a315b5578357e07b8201ddf2f431f8fee04874ef1f4a18974d6",
    "unsafe_b0_ids": "c5d18593a582e98569d546ab5f716626fd7f13918fd2bbd05979fd4298ff6350",
    "safe_b0_ids": "ed175446df1684181412ee0003233eeaa367dfddb036975d776a35076994f8b3",
    "phase13_checkpoint": "146ec655360be84c2edbda6ec4137d0bec834e6df86247f2a4db09bc8f9ee643",
}
ACTIVE_SAFE_CWES = {"CWE-120", "CWE-327", "CWE-338", "CWE-89"}


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path: Path, value: Any) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
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


def verify(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "baseline_subset": args.baseline_subset.resolve(),
        "strength_subset": args.strength_subset.resolve(),
        "dev_subset_hashes": args.dev_subset_hashes.resolve(),
        "denominator_audit": args.denominator_audit.resolve(),
        "development_prompts": args.development_prompts.resolve(),
        "b0_outputs": args.b0_outputs.resolve(),
        "b0_icd": args.b0_icd.resolve(),
        "unsafe_b0_ids": args.unsafe_b0_ids.resolve(),
        "safe_b0_ids": args.safe_b0_ids.resolve(),
        "phase13_checkpoint": args.phase13_checkpoint.resolve(),
    }
    source_hashes = {}
    for key, path in paths.items():
        require(path.is_file(), f"Missing required input: {path}")
        actual = sha256_file(path)
        require(actual == EXPECTED_HASHES[key], f"{key} hash mismatch: {actual}")
        source_hashes[key] = {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}

    baseline = load_json(paths["baseline_subset"])
    strength = load_json(paths["strength_subset"])
    hashes = load_json(paths["dev_subset_hashes"])
    audit = load_json(paths["denominator_audit"])
    dev_prompts = load_json(paths["development_prompts"])
    b0_outputs = load_json(paths["b0_outputs"])
    b0_icd = load_json(paths["b0_icd"])
    unsafe_submitted = load_json(paths["unsafe_b0_ids"])
    safe_submitted = load_json(paths["safe_b0_ids"])
    phase13_checkpoint = load_json(paths["phase13_checkpoint"])

    require(phase13_checkpoint.get("checkpoint_status") == "PASS", "Phase 13 checkpoint is not PASS")
    require(phase13_checkpoint.get("held_out_accessed") is False, "Phase 13 checkpoint reports held-out access")
    require(audit.get("verification_status") == "PASS" and not audit.get("blockers"), "Phase 2 denominator audit is not PASS")
    require(baseline.get("validation_status") == "PASS" and baseline.get("split") == "DEVELOPMENT", "Baseline subset is not verified DEVELOPMENT")
    require(strength.get("validation_status") == "PASS" and strength.get("split") == "DEVELOPMENT", "Strength subset is not verified DEVELOPMENT")
    require(hashes.get("validation_status") == "PASS", "dev_subset_hashes validation is not PASS")
    require(hashes.get("manifest_files", {}).get("baseline_selection_subset", {}).get("sha256") == EXPECTED_HASHES["baseline_subset"], "dev_subset_hashes baseline link mismatch")
    require(hashes.get("manifest_files", {}).get("strength_subset_manifest", {}).get("sha256") == EXPECTED_HASHES["strength_subset"], "dev_subset_hashes strength link mismatch")

    verified_n_vuln = audit["development"]["b0_vulnerable_eligible_count"]
    expected_total = verified_n_vuln * 2
    require(verified_n_vuln == 60, "Verified current DEV_VULN count differs from recorded value 60")
    require(len(baseline.get("records", [])) == expected_total == 120, "Baseline subset size mismatch")
    require(baseline.get("counts") == {
        "dev_vuln": verified_n_vuln,
        "dev_safe_small": verified_n_vuln,
        "baseline_selection_subset": expected_total,
    }, "Baseline subset recorded counts mismatch")

    records = baseline["records"]
    ids = [record.get("prompt_id") for record in records]
    require(ids == baseline.get("prompt_ids_source_order"), "Baseline subset record order differs from frozen prompt order")
    require(len(ids) == len(set(ids)), "Baseline subset contains duplicate IDs")
    vuln_ids = [record["prompt_id"] for record in records if record.get("population_type") == "DEV_VULN"]
    safe_ids = [record["prompt_id"] for record in records if record.get("population_type") == "DEV_SAFE_SMALL"]
    require(vuln_ids == baseline.get("dev_vuln_prompt_ids_source_order"), "DEV_VULN component order mismatch")
    require(safe_ids == baseline.get("dev_safe_small_prompt_ids_source_order"), "DEV_SAFE_SMALL component order mismatch")
    require(len(vuln_ids) == verified_n_vuln and len(safe_ids) == verified_n_vuln, "Component counts mismatch")
    require(set(vuln_ids).isdisjoint(safe_ids), "DEV_VULN and DEV_SAFE_SMALL overlap")

    audit_vuln_ids = audit["development"]["b0_vulnerable_eligible_prompt_ids"]
    audit_safe_ids = audit["development"]["b0_safe_eligible_prompt_ids"]
    audit_eligible_ids = audit["development"]["scanner_eligible_prompt_ids"]
    heldout_ids = audit["heldout"]["total_prompt_ids"]
    require(set(vuln_ids) == set(audit_vuln_ids), "DEV_VULN does not contain every verified vulnerable eligible ID")
    require(set(safe_ids).issubset(audit_safe_ids), "DEV_SAFE_SMALL includes non-safe-eligible ID")
    require(set(ids).issubset(audit_eligible_ids), "Subset includes scanner-ineligible ID")
    require(set(ids).isdisjoint(heldout_ids), "Subset overlaps held-out IDs")
    require(set(vuln_ids) == set(unsafe_submitted), "DEV_VULN differs from submitted B0 unsafe partition")
    require(set(safe_ids).issubset(safe_submitted), "DEV_SAFE_SMALL differs from submitted B0 safe complement")

    strength_safe_ids = strength.get("dev_safe_large_prompt_ids_source_order")
    require(set(safe_ids) < set(strength_safe_ids), "DEV_SAFE_SMALL is not a strict subset of DEV_SAFE_LARGE")
    require(baseline.get("nesting", {}).get("dev_safe_large_prompt_ids_source_order") == strength_safe_ids, "Baseline nesting record differs from strength subset")
    require(hashes.get("component_hashes", {}).get("dev_vuln_prompt_ids_source_order") == canonical_hash(vuln_ids), "DEV_VULN component hash mismatch")
    require(hashes.get("component_hashes", {}).get("dev_safe_small_prompt_ids_source_order") == canonical_hash(safe_ids), "DEV_SAFE_SMALL component hash mismatch")
    require(hashes.get("component_hashes", {}).get("baseline_selection_prompt_ids_source_order") == canonical_hash(ids), "Baseline prompt-order hash mismatch")

    dev_map = {record["prompt_id"]: record for record in dev_prompts}
    output_map = {record["prompt_id"]: record for record in b0_outputs}
    icd_map = {record["prompt_id"]: record for record in b0_icd}
    require(len(dev_map) == 1341 and set(ids).issubset(dev_map), "Development source membership mismatch")
    for record in records:
        prompt_id = record["prompt_id"]
        source = dev_map[prompt_id]
        scan = icd_map[prompt_id]
        output = output_map[prompt_id]
        require(scan.get("skipped") is False, f"Prompt {prompt_id} is scanner skipped")
        expected_vulnerable = record["population_type"] == "DEV_VULN"
        require(scan.get("is_vulnerable") is expected_vulnerable, f"Prompt {prompt_id} B0 classification mismatch")
        require(record.get("scanner_eligible") is True and record.get("b0_is_vulnerable") is expected_vulnerable, f"Prompt {prompt_id} manifest classification mismatch")
        require(record.get("prompt_text_sha256") == sha256_text(source["test_case_prompt"]), f"Prompt {prompt_id} text hash mismatch")
        require(record.get("b0_output_sha256") == sha256_text(output["generated_code"]), f"Prompt {prompt_id} B0 output hash mismatch")
        require(record.get("cwe_id") == source.get("cwe_identifier") and record.get("language") == source.get("language"), f"Prompt {prompt_id} source metadata mismatch")

    safe_cwes = {record.get("cwe_id") for record in records if record.get("population_type") == "DEV_SAFE_SMALL"}
    require(safe_cwes == ACTIVE_SAFE_CWES, f"DEV_SAFE_SMALL CWE set mismatch: {sorted(safe_cwes)}")
    checks = {
        "phase13_dependency_pass": True,
        "active_manifest_hash_match": True,
        "subset_size_from_verified_denominator": True,
        "development_only_membership": True,
        "all_scanner_eligible": True,
        "dev_vuln_complete_exact": True,
        "dev_safe_small_frozen_exact": True,
        "b0_classification_matches_phase2": True,
        "no_duplicates": True,
        "no_heldout_overlap_using_phase2_audit_id_inventory": True,
        "exact_phase13_prompt_order_preserved": True,
        "dev_safe_small_strictly_nested": True,
        "no_resampling": True,
        "prompt_and_b0_output_provenance_match": True,
    }
    return {
        "schema_version": "1.0",
        "phase": 14,
        "artifact": "baseline_subset_verification",
        "verification_status": "PASS",
        "baseline_selection_subset_path": str(paths["baseline_subset"]),
        "baseline_selection_subset_sha256": source_hashes["baseline_subset"]["sha256"],
        "verified_denominator_source": str(paths["denominator_audit"]),
        "verified_n_vuln": verified_n_vuln,
        "subset_count": len(ids),
        "dev_vuln_count": len(vuln_ids),
        "dev_safe_small_count": len(safe_ids),
        "dev_safe_large_count": len(strength_safe_ids),
        "active_safe_cwes": sorted(safe_cwes),
        "prompt_ids_source_order": ids,
        "dev_vuln_prompt_ids_source_order": vuln_ids,
        "dev_safe_small_prompt_ids_source_order": safe_ids,
        "checks": checks,
        "source_hashes": source_hashes,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "generation_executed": False,
        "scanner_executed": False,
        "held_out_prompt_or_result_file_accessed": False,
        "heldout_overlap_check_source": "Phase 2 denominator-audit prompt-ID inventory only",
        "blockers": [],
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--baseline-subset", type=Path, required=True)
    result.add_argument("--strength-subset", type=Path, required=True)
    result.add_argument("--dev-subset-hashes", type=Path, required=True)
    result.add_argument("--denominator-audit", type=Path, required=True)
    result.add_argument("--development-prompts", type=Path, required=True)
    result.add_argument("--b0-outputs", type=Path, required=True)
    result.add_argument("--b0-icd", type=Path, required=True)
    result.add_argument("--unsafe-b0-ids", type=Path, required=True)
    result.add_argument("--safe-b0-ids", type=Path, required=True)
    result.add_argument("--phase13-checkpoint", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        result = verify(args)
        atomic_write(args.output.resolve(), result)
        print(json.dumps({
            "verification_status": result["verification_status"],
            "subset_count": result["subset_count"],
            "dev_vuln_count": result["dev_vuln_count"],
            "dev_safe_small_count": result["dev_safe_small_count"],
            "output_sha256": sha256_file(args.output.resolve()),
        }, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "schema_version": "1.0", "phase": 14,
            "artifact": "baseline_subset_verification",
            "verification_status": "FAIL", "blockers": [f"{type(exc).__name__}: {exc}"],
            "generation_executed": False, "scanner_executed": False,
            "held_out_prompt_or_result_file_accessed": False,
        }
        atomic_write(args.output.resolve(), failure)
        print(f"PHASE14_SUBSET_VERIFICATION_ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
