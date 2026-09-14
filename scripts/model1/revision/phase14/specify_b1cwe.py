#!/usr/bin/env python3
"""Freeze the source-gated Phase 14 B1-CWE specification and guidance map."""

from __future__ import annotations

import argparse
from pathlib import Path

from phase14_common import (
    ACTIVE_CAA_CWES, BASELINE_SUBSET, EXPECTED_BASELINE_SUBSET_SHA256,
    EXPECTED_SOURCE_RECOVERY_SHA256, GENERATION_SETTINGS, MODEL_CACHE_SNAPSHOT,
    MODEL_ID, OUTPUTS, REQUIRED_SEED, SOURCE_RECOVERY, SUBMITTED_B1_PREFIX,
    TOKENIZER_ID, atomic_json, recovery_record, relative, sha256_path,
    validate_preparation,
)


def build() -> tuple[dict, dict]:
    state = validate_preparation()
    recovery = state["recovery"]
    route_records = {
        cwe: recovery_record(recovery, "B1-CWE", f"guidance_{cwe}")
        for cwe in ACTIVE_CAA_CWES
    }
    guidance_routes = {}
    for cwe, record in route_records.items():
        content = record["recovered_content_or_identifier"]
        guidance_routes[cwe] = {
            "cwe_id": cwe,
            "guidance_available": False,
            "guidance_text": None,
            "fallback": "submitted B1",
            "recovery_status": record["status"],
            "source": record["source"],
            "source_location": record["exact_source_location_or_artifact"],
            "reason": record["reason"],
        }
        assert content["guidance_available"] is False and content["fallback"] == "submitted B1"

    guidance_map = {
        "schema_version": "phase14_b1cwe_guidance_map_v1",
        "method": "B1-CWE",
        "information_tier": "ORACLE_CWE",
        "configured_routes": list(ACTIVE_CAA_CWES),
        "routes": guidance_routes,
        "unmapped_or_unsupported_cwe_policy": {
            "action": "submitted B1",
            "guidance_available": False,
            "route_status": "UNSUPPORTED_ROUTE_FALLBACK_SUBMITTED_B1",
            "authority": "FINAL REVISION_PLAN Phase 14.2 steps 2 and 4; Section 2.5 route fallback rule",
            "reason": "The map is restricted to scanner-supported routes; every route without guidance uses the B1 fallback.",
        },
        "fabricated_guidance_count": 0,
        "source_recovery_path": relative(SOURCE_RECOVERY),
        "source_recovery_sha256": EXPECTED_SOURCE_RECOVERY_SHA256,
    }

    builder = Path(__file__).resolve()
    spec = {
        "schema_version": "phase14_b1cwe_method_spec_v1",
        "method": "B1-CWE",
        "information_tier": "ORACLE_CWE",
        "source_identity": {
            "primary": "TOSEM_2025 DOI 10.1145/3722108",
            "replication": "10.6084/m9.figshare.28229717.v1",
            "source_recovery_path": relative(SOURCE_RECOVERY),
            "source_recovery_sha256": EXPECTED_SOURCE_RECOVERY_SHA256,
        },
        "prompt_structure": {
            "source_identifier": recovery_record(recovery, "B1-CWE", "zero_shot_prompt_structure")["recovered_content_or_identifier"],
            "ordered_components_when_guidance_available": [
                "secure-code task instruction", "original coding task",
                "target-CWE mitigation list", "catch-all additional-security sentence",
            ],
            "original_coding_task_placement": "second; preserved byte-for-byte",
            "cwe_metadata_placement": "target CWE performs guidance lookup; literal identifier is not required in rendered prompt",
            "guidance_placement": "after original coding task and before catch-all sentence",
            "active_routes_have_renderable_guidance": False,
        },
        "generation_stage_count": 1,
        "target_cwe_metadata_field": "source_prompt.cwe_identifier",
        "target_cwe_field_matches_bstar": True,
        "guidance_lookup": {
            "map_path": "revision/model1/phase14/outputs/b1cwe_guidance_map.json",
            "known_route_missing_guidance": "submitted B1",
            "unmapped_route": "submitted B1",
            "no_proxy_or_substitution": True,
        },
        "submitted_b1_fallback": {
            "source_path": "phases/phase5/run_secure_zeroshot.py",
            "source_sha256": "b6c5398ff436cc788f564ad57b9150daa3438ef41873f1a321cdbc840f4f4d24",
            "security_prefix": SUBMITTED_B1_PREFIX,
            "rendering": "security_prefix + original coding task",
        },
        "output_extraction_cleaning": {
            "source_rule": recovery_record(recovery, "B1-CWE", "output_extraction_and_cleaning")["recovered_content_or_identifier"],
            "implementation": "decode generated tokens only; skip special tokens; do not strip or repair",
        },
        "seed": REQUIRED_SEED,
        "model": {"id": MODEL_ID, "cache_snapshot": MODEL_CACHE_SNAPSHOT, "dtype": "float16"},
        "tokenizer": {"id": TOKENIZER_ID, "cache_snapshot": MODEL_CACHE_SNAPSHOT},
        "decoding": GENERATION_SETTINGS,
        "input": {
            "path": relative(BASELINE_SUBSET),
            "sha256": EXPECTED_BASELINE_SUBSET_SHA256,
            "record_count": 120,
            "order": "prompt_ids_source_order",
        },
        "output_schema": [
            "schema_version", "prompt_id", "source_index", "population_type",
            "prompt_text", "prompt_text_sha256", "language", "target_cwe",
            "method", "information_tier", "guidance_available", "guidance_route_status",
            "fallback_status", "rendered_prompt_sha256", "generated_code",
            "generated_token_count", "generation_status", "failure_reason",
            "empty_status", "seed", "model_id", "model_cache_snapshot",
            "input_manifest_sha256", "method_spec_sha256", "guidance_map_sha256",
            "source_recovery_sha256", "runner_sha256",
        ],
        "failure_empty_policy": {
            "one_record_per_prompt": True,
            "preserve_exception": True,
            "exception_output": "empty string with generation_status=FAILED",
            "empty_output": "preserved with empty_status=EMPTY",
            "retry_to_success": False,
        },
        "checkpoint_resume": {
            "checkpoint_after_every_prompt": True,
            "strict_prefix_order": True,
            "restore_rng_state": True,
            "reuse_completed_record": True,
        },
        "builder": {"path": relative(builder), "sha256": sha256_path(builder)},
    }
    return spec, guidance_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec-output", default=str(OUTPUTS / "b1cwe_method_spec.json"))
    parser.add_argument("--guidance-output", default=str(OUTPUTS / "b1cwe_guidance_map.json"))
    args = parser.parse_args()
    spec, guidance = build()
    atomic_json(Path(args.spec_output).resolve(), spec)
    atomic_json(Path(args.guidance_output).resolve(), guidance)
    print({"spec_sha256": sha256_path(Path(args.spec_output)), "guidance_sha256": sha256_path(Path(args.guidance_output))})


if __name__ == "__main__":
    main()
