#!/usr/bin/env python3
"""Verify Phase 16A gate, population, and reused/preserved artifacts."""

from phase16_common import *

REUSED = [
    ("B0", 42, "REUSE_CANONICAL", "outputs/phase9/baseline_test_outputs.json", "outputs/phase9/baseline_test_icd.json"),
    ("B1", 42, "REUSE_CANONICAL", "outputs/phase9/zeroshot_test_outputs.json", "outputs/phase9/zeroshot_test_icd.json"),
    ("B*", 42, "REUSE_CANONICAL", "outputs/phase9/bstar_test_outputs.json", "outputs/phase9/bstar_test_icd.json"),
    ("B3-PG", 42, "PRESERVE_SUBMITTED", "outputs/phase9/semantic_static_test_outputs.json", "outputs/phase9/semantic_static_test_icd.json"),
]

def main() -> None:
    gate = phase15_gate()
    population = load_frozen_population()
    frozen = read_json(FREEZE)["source_artifacts"]
    population_record = {
        "schema_version": "phase16_heldout_population_verification_v1", "status": "PASS",
        "phase15_freeze_timestamp": PHASE15_FREEZE_TIMESTAMP,
        "first_access_timestamp": FIRST_ACCESS_TIMESTAMP,
        "source_path": relative(TEST_PROMPTS), "source_sha256": population["source_file_sha256"],
        "record_count": EXPECTED_COUNT, "unique_prompt_ids": EXPECTED_COUNT,
        "frozen_order_source": "revision/model1/phase2/phase2_denominator_audit.json heldout.total_prompt_ids",
        "frozen_order_sha256": population["prompt_ids_sha256"],
        "ordered_population_sha256": population["population_sha256"],
        "source_storage_order_matches_frozen_order": population["source_file_storage_order_matches_frozen_order"],
        "development_id_overlap": 0, "filtered": False, "resampled": False,
        "phase15_gate": gate,
    }
    atomic_json(OUTPUTS / "heldout_population_verification.json", population_record)
    rows = []
    for method, seed, disposition, generation_text, scan_text in REUSED:
        generation = ROOT / generation_text
        scan = ROOT / scan_text
        require(generation_text in frozen and scan_text in frozen, f"{method} reuse path absent from Phase15 freeze")
        require(sha256_path(generation) == frozen[generation_text]["sha256"], f"{method} generation hash mismatch")
        require(sha256_path(scan) == frozen[scan_text]["sha256"], f"{method} scan hash mismatch")
        body = read_json(generation)
        records = body["final_outputs"] if isinstance(body, dict) and "final_outputs" in body else body
        require(len(records) == EXPECTED_COUNT and len({int(row["prompt_id"]) for row in records}) == EXPECTED_COUNT, f"{method} reuse population mismatch")
        rows.append({
            "method": method, "seed": seed, "disposition": disposition,
            "generation_path": generation_text, "generation_sha256": sha256_path(generation),
            "existing_scan_path": scan_text, "existing_scan_sha256": sha256_path(scan),
            "record_count": len(records), "unique_prompt_ids": len({int(row["prompt_id"]) for row in records}),
            "provenance_status": "CANONICAL_PHASE15_FROZEN_REUSE" if disposition == "REUSE_CANONICAL" else "SUBMITTED_NEGATIVE_RESULT_WITH_FROZEN_HISTORICAL_LIMITATIONS",
            "verification_result": "PASS", "regeneration_required": False,
        })
    atomic_json(OUTPUTS / "phase16_reuse_verification.json", {
        "schema_version": "phase16_reuse_verification_v1", "status": "PASS",
        "verified_at_utc": utc_now(), "artifacts": rows,
    })
    print(json.dumps({"gate": gate["status"], "population": population_record, "reused": rows}, indent=2))

if __name__ == "__main__":
    main()
