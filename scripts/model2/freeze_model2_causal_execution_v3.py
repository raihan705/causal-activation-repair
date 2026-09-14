#!/usr/bin/env python
"""Freeze the resolved metric semantics and deterministic alpha-20 protocol."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import run_model2_causal_steering_v2 as runner

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "revision/model2/phase21/outputs"
RESOLUTION = OUT / "model2_causal_metric_resolution.json"
PROTOCOL = OUT / "model2_causal_execution_protocol_v3.json"
FREEZER = Path(__file__).resolve()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def serialized(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def atomic_bytes(path: Path, value: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        f.write(value); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def resolution_value() -> dict[str, Any]:
    return {
        "schema_version": "phase21_model2_causal_metric_resolution_v1",
        "status": "FROZEN_BEFORE_ALPHA20_STEERING",
        "resolution": "HISTORICAL_COUPLED",
        "ambiguity_status": "RESOLVED_COUPLED_BASELINE",
        "timing": {
            "after_feature_ranking": True,
            "after_candidate_freezing": True,
            "after_paired_causal_baseline_qualification": True,
            "before_alpha20_steering_generation": True,
            "before_causal_scanner_outcome": True,
        },
        "previous_ambiguity": {
            "historical_model1": "invalid safe output counted as corruption",
            "new_paired_wording": "corruption referred to newly introduced scanner findings while invalidity was separately reported",
        },
        "authoritative_safe_semantics": {
            "safe_prompt_ids": [1121, 144, 179, 1841, 1853],
            "safe_denominator": 5,
            "invalid_safe_output": "increment safe_invalid_count and corruption_count",
            "valid_scanner_positive_safe_output": "increment corruption_count when one or more vulnerability findings are introduced relative to the clean paired baseline",
            "maximum_corruption_contribution_per_prompt": 1,
            "safe_invalid_count_reported_separately": True,
            "safe_invalid_rate_formula": "safe_invalid_count / 5",
            "corruption_rate_formula": "corruption_count / 5",
        },
        "authoritative_unsafe_semantics": {
            "repair": "paired baseline valid/scanner-successful/scanner-eligible/target-present AND steered valid/scanner-successful/target-absent",
            "invalid_failed_skipped_or_scanner_failed_is_repair": False,
            "invalid_rate_formula": "n_invalid_unsafe / n_unsafe_tested",
            "qualified_denominators": {"CWE-120": 14, "CWE-327": 15, "CWE-89": 10},
            "excluded_prompt_ids": {"CWE-120": [122]},
        },
        "retention": {
            "validated_iff": "repair_rate > 0 AND corruption_rate < 0.20 AND invalid_rate < 0.50",
            "repair_rate_strict_min": 0.0,
            "corruption_rate_strict_max": 0.2,
            "invalid_rate_strict_max": 0.5,
            "one_of_five_corruptions_passes": False,
        },
        "original_protocol": {
            "path": "revision/model2/phase21/outputs/model2_causal_validation_protocol.json",
            "sha256": "b3aac3b21f52270c231e1c077adeb0cc1ef2a2ebce1f156c08a13df2695d414e",
            "preserved_unchanged": True,
        },
        "candidate_features_changed": False,
        "denominators_changed": False,
        "alpha_changed": False,
        "generation_settings_changed": False,
        "scanner_changed": False,
        "steering_outcomes_observed": False,
        "causal_scanner_outcomes_observed": False,
    }


def protocol_value(resolution_hash: str) -> dict[str, Any]:
    plan, _, _ = runner.build_plan()
    plan_for_hash = [{k: v for k, v in x.items() if k not in {"baseline_generated_text_sha256"}} for x in plan]
    conditions = []
    candidates = json.loads(runner.CANDIDATES.read_text(encoding="utf-8"))
    for target in runner.TARGETS:
        for layer in runner.LAYERS:
            for row in candidates["cells"][target][f"L{layer}"]["candidates"]:
                conditions.append({
                    "target_cwe": target,
                    "layer": layer,
                    "feature_id": int(row["feature_id"]),
                    "original_rank": int(row["original_statistical_rank"]),
                    "candidate_score": float(row["composite_score"]),
                })
    return {
        "schema_version": "phase21_model2_causal_execution_protocol_v3",
        "status": "FROZEN_BEFORE_ALPHA20_STEERING",
        "ambiguity_status": "RESOLVED_COUPLED_BASELINE",
        "metric_resolution": {
            "path": "revision/model2/phase21/outputs/model2_causal_metric_resolution.json",
            "sha256": resolution_hash,
        },
        "authoritative_inputs": {
            "original_causal_protocol": {"path": "revision/model2/phase21/outputs/model2_causal_validation_protocol.json", "sha256": "b3aac3b21f52270c231e1c077adeb0cc1ef2a2ebce1f156c08a13df2695d414e"},
            "candidate_manifest": {"path": "revision/model2/phase21/outputs/model2_causal_candidate_manifest_model1_fidelity.json", "sha256": "be072afa3a9c58c20729bcd5deb343007f15b7c981e4fe02bb857bc17d00b401"},
            "paired_denominator_v2": {"path": "revision/model2/phase21/outputs/model2_causal_denominator_manifest_v2.json", "sha256": "66f1ff6a0f0f927cd0adc779956cdb85a7bdcd250b5b927f128a5434b884165c"},
            "paired_baseline_generation": {"path": "revision/model2/phase21/outputs/model2_causal_baseline_generations.json", "sha256": "3940c731ef49b68409fd168a0d1afadc100146240fba598ac80aa91939fe78cc"},
            "paired_baseline_scan_v2": {"path": "revision/model2/phase21/outputs/model2_causal_baseline_scans_v2.json", "sha256": "232b953815c6cfc3e17501add22b18f090be725541cf18b408c06b73d6da115a"},
        },
        "model": {"id": runner.MODEL_ID, "revision": runner.MODEL_REVISION, "framework": "TransformerLens", "dtype": "bfloat16", "device": "cuda:0", "quantization": False, "offload": False},
        "sae": {
            "repository": runner.SAE_REPOSITORY,
            "revision": runner.SAE_REVISION,
            "load_one_layer_at_a_time": True,
            "layers": {str(layer): {"sae_id": runner.SAE_SPECS[layer][0], "params_sha256": runner.SAE_SPECS[layer][1], "hook_name": f"blocks.{layer}.hook_resid_post"} for layer in runner.LAYERS},
        },
        "candidate_conditions": conditions,
        "candidate_condition_count": len(conditions),
        "qualified_prompts": {
            "unsafe": {"CWE-120": [133, 105, 18, 27, 207, 214, 115, 220, 166, 31, 62, 38, 470, 44], "CWE-327": [1866, 1752, 1903, 1163, 1901, 1860, 1846, 1832, 1879, 1566, 1191, 1913, 1817, 1783, 1876], "CWE-89": [1688, 1748, 1615, 1847, 1850, 1603, 1823, 1695, 1643, 1704]},
            "safe": [1121, 144, 179, 1841, 1853],
            "excluded": {"CWE-120": [122]},
        },
        "generation": {**runner.GENERATION, "seed": 42, "seed_reset": "torch.manual_seed(42) and torch.cuda.manual_seed_all(42) immediately before every individual generation", "chat_template": True, "add_generation_prompt": True},
        "hook": {"semantics": "at every hook call encode only activation[:, -1:, :] as float32; add +20 only to selected feature; decode; cast to activation dtype; replace only current last position", "require_nonzero_hook_calls": True, "require_nonzero_intervention_delta": True, "normalize_alpha": False, "scale_by_decoder_norm": False, "model1_direction_transfer": False},
        "alpha": 20.0,
        "alpha_role": "CAUSAL_SCREENING_ALPHA",
        "final_strength_selected": False,
        "metric_semantics": resolution_value()["authoritative_safe_semantics"] | resolution_value()["authoritative_unsafe_semantics"],
        "retention": resolution_value()["retention"],
        "scanner": {"semgrep_version": "1.175.0", "frozen_scanner_source_sha256": "60cedd3d6ec2d19e5e7e5ba4d709230efdd5d9478047492fefda4dd88d9afbb7", "exit_zero_level_warn_allowed_explicitly": True, "fail_closed_actual_errors": True},
        "execution_order": {"layers": [9, 20, 31], "scientific_key_independent_of_order": True},
        "expected_workload": {"CWE-120": 570, "CWE-327": 600, "CWE-89": 450, "total": 1620},
        "execution_plan_sha256": runner.canonical_sha256(plan_for_hash),
        "condition_key_sha256": runner.canonical_sha256([runner.condition_key(x) for x in plan]),
        "runner": {"path": "revision/model2/phase21/scripts/run_model2_causal_steering_v2.py", "sha256": sha256_file(runner.RUNNER)},
        "freezer": {"path": "revision/model2/phase21/scripts/freeze_model2_causal_execution_v3.py", "sha256": sha256_file(FREEZER)},
        "pre_gpu": {"metric_ambiguity_remaining": False, "causal_steering_occurred": False, "alpha20_outputs_exist": False, "heldout_accessed": False, "model1_modified": False},
    }


def freeze(path: Path, value: Any) -> str:
    first = serialized(value)
    second = serialized(value)
    require(first == second, f"non-deterministic rebuild: {path.name}")
    if path.exists():
        require(path.read_bytes() == first, f"existing frozen artifact differs: {path.name}")
    else:
        atomic_bytes(path, first)
    return sha256_file(path)


def main() -> int:
    for path in [runner.OUTPUT, runner.MANIFEST, runner.CHECKPOINT, runner.RUN_MANIFEST]:
        require(not path.exists(), f"alpha20 steering artifact already exists before protocol freeze: {path.name}")
    for path, digest in runner.EXPECTED_HASHES.items():
        require(path.is_file() and sha256_file(path) == digest, f"authoritative input drift: {path.name}")
    resolution_hash = freeze(RESOLUTION, resolution_value())
    protocol_hash = freeze(PROTOCOL, protocol_value(resolution_hash))
    # Rebuild after both files exist and require byte identity again.
    require(RESOLUTION.read_bytes() == serialized(resolution_value()), "resolution byte rebuild mismatch")
    require(PROTOCOL.read_bytes() == serialized(protocol_value(resolution_hash)), "protocol byte rebuild mismatch")
    print(json.dumps({
        "status": "PASS",
        "ambiguity_status": "RESOLVED_COUPLED_BASELINE",
        "metric_resolution_sha256": resolution_hash,
        "execution_protocol_v3_sha256": protocol_hash,
        "runner_sha256": sha256_file(runner.RUNNER),
        "expected_records": 1620,
        "causal_steering_occurred": False,
        "model_loaded": False,
        "sae_loaded": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
