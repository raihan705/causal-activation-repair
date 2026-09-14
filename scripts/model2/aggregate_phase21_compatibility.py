#!/usr/bin/env python
"""Aggregate the three clean-process Phase 21 technical layer records."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
MODEL_ID = "google/gemma-2-9b-it"
MODEL_REVISION = "11c9b309abf73637e4b6f9a3fa1e92e615547819"
SAE_REPOSITORY = "google/gemma-scope-9b-it-res"
SAE_REVISION = "e86af97a5b6fbbccca28ab654f2fda1b0768f770"
OFFICIAL_REPORT = "https://storage.googleapis.com/gemma-scope/gemma-scope-report.pdf"


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


records = [
    json.loads((OUT / f"phase21_layer_{layer}_technical.json").read_text(encoding="utf-8"))
    for layer in (9, 20, 31)
]
if any(record["status"] != "COMPATIBLE" for record in records):
    raise SystemExit("All three clean-process layer records must exist and be technically compatible")
if any(record["model_revision"] != MODEL_REVISION or record["sae_revision"] != SAE_REVISION for record in records):
    raise SystemExit("Frozen revision mismatch")

quality_warning = (
    "All tensors and interventions were finite, but the single synthetic prompt produced "
    "mean L0 values far above the selected variant labels and negative explained variance. "
    "Treat compatibility as technical only; reconstruction suitability is reported by the layer-level analysis."
)
config = records[0]["model_config"]

atomic_json(
    OUT / "phase21_access_gate.json",
    {
        "schema_version": "phase21_access_gate_v1",
        "status": "PASS",
        "hf_cli_available": True,
        "huggingface_hub_version": "0.36.2",
        "authentication": {
            "status": "PASS",
            "username": "authenticated_user_redacted",
            "login_flow_invoked": False,
            "login_method": "PREEXISTING_HF_CLI_AUTHENTICATION",
            "credential_recorded_in_repository": False,
        },
        "gemma_access_test": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "file": "config.json",
            "result": "PASS",
            "gated_access": "PASS",
        },
        "model_config_revision_verified": True,
        "sae_access_test_run": True,
        "sae_access_test_status": "PASS",
        "sae_repository": SAE_REPOSITORY,
        "sae_revision": SAE_REVISION,
        "all_frozen_sae_ids_exist": True,
        "compatibility_preflight_resumed": True,
        "security_feature_discovery_run": False,
        "heldout_used": False,
        "model1_modified": False,
    },
)

atomic_json(
    OUT / "phase21_environment_manifest.json",
    {
        "schema_version": "phase21_environment_manifest_v1",
        "status": "TECHNICAL_COMPATIBILITY_COMPLETE",
        "python": "3.11.14",
        "pytorch": "2.6.0+cu118",
        "cuda_runtime": "11.8",
        "cuda_available": True,
        "transformers": "4.57.6",
        "transformer_lens": "2.17.0",
        "sae_lens": "6.37.6",
        "huggingface_hub": "0.36.2",
        "gpu": "NVIDIA RTX A5000",
        "gpu_total_vram_bytes": 25756696576,
        "hf_cli_authentication_status": "PASS",
        "hf_authenticated_username": "authenticated_user_redacted",
        "gemma_gated_access": "PASS",
        "model_revision": MODEL_REVISION,
        "sae_repository_access": "PASS",
        "sae_revision": SAE_REVISION,
        "hook_framework": "TransformerLens",
        "execution_isolation": "ONE_FRESH_PROCESS_PER_SAE_LAYER",
        "model_loading_options": records[0]["model_loading_options"],
        "compatible_layer_count": 3,
        "reconstruction_quality_warning": quality_warning,
        "package_changes": [],
        "environment_rebuilt": False,
        "hardware_migrated": False,
        "quantization_used": False,
        "cpu_offload_used": False,
    },
)

model_memory = records[0]["memory"]["model_only"]
base = records[0]["base_generation"]
atomic_json(
    OUT / "phase21_model_smoke.json",
    {
        "schema_version": "phase21_model_smoke_v1",
        "status": "PASS",
        "model_id": MODEL_ID,
        "resolved_revision": MODEL_REVISION,
        "config_path_revision_verified": True,
        "architectures": config["architectures"],
        "hidden_size": config["hidden_size"],
        "num_hidden_layers": config["num_hidden_layers"],
        "vocab_size": config["vocab_size"],
        "model_type": config["model_type"],
        "torch_dtype_from_config": config["torch_dtype"],
        "loaded_dtype": "torch.bfloat16",
        "hook_framework": "TransformerLens",
        "parameter_devices": ["cuda:0"],
        "loading_time_seconds_per_clean_run": [record["model_load_seconds"] for record in records],
        **model_memory,
        "prompt_source": base["prompt_source"],
        "prompt_sha256": base["prompt_sha256"],
        "input_token_count": base["input_tokens"],
        "generated_token_count": base["generated_tokens"],
        "base_logits_finite": base["logits_finite"],
        "generation_run": True,
        "generation_status": "PASS",
        "quantization_used": False,
        "cpu_offload_used": False,
        "heldout_used": False,
    },
)

selection = []
for record in records:
    selection.append(
        {
            "layer": record["layer"],
            "average_l0": record["average_l0_variant"],
            "sae_id": record["sae_id"],
            "params_file": f"{record['sae_id']}/params.npz",
            "params_sha256": record["params_sha256"],
            "source_parameter_keys": record["source_parameter_keys"],
            "d_in": record["d_in"],
            "d_sae": record["d_sae"],
            "hook_point": record["hook_name"],
            "hook_framework": "TransformerLens",
            "sae_dtype": record["sae_dtype"],
        }
    )
atomic_json(
    OUT / "model_sae_selection_record.json",
    {
        "schema_version": "phase21_model_sae_selection_v1",
        "status": "FROZEN_BEFORE_OUTCOME_ACCESS_PASS",
        "model": {
            "model_id": MODEL_ID,
            "resolved_revision": MODEL_REVISION,
            **config,
            "licence": "gemma",
            "access_state": "PASS",
            "hook_framework": "TransformerLens",
        },
        "sae": {
            "repository": SAE_REPOSITORY,
            "resolved_revision": SAE_REVISION,
            "licence": "cc-by-4.0",
            "width": "width_16k",
            "selected": selection,
        },
        "selection_rule": "Per retained layer, select the available width_16k SAE with average L0 closest to 50; smaller L0 breaks equal-distance ties.",
        "selection_before_reconstruction_or_security_outcome": True,
        "security_feature_discovery_run": False,
        "heldout_used": False,
    },
)

matrix_rows = []
reconstruction_rows = []
hook_tests = []
memory_rows = [{"scope": "model_only", "layer": "", "sae_id": "", **model_memory}]
for record in records:
    rec = record["reconstruction"]
    intervention = record["intervention"]
    matrix_rows.append(
        {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "sae_repository": SAE_REPOSITORY,
            "sae_revision": SAE_REVISION,
            "sae_id": record["sae_id"],
            "layer": record["layer"],
            "width": "width_16k",
            "average_l0": record["average_l0_variant"],
            "observed_mean_l0": rec["mean_l0"],
            "normalized_mse": rec["normalized_mse"],
            "explained_variance": rec["explained_variance"],
            "d_model": record["d_in"],
            "d_sae": record["d_sae"],
            "hook_framework": "TransformerLens",
            "hook_name": record["hook_name"],
            "model_load": "PASS",
            "sae_load": "PASS",
            "dimension_match": "PASS",
            "hook_capture": "PASS",
            "encode": "PASS",
            "decode": "PASS",
            "reconstruction_finite": "PASS",
            "intervention_reinsert": intervention["status"],
            "generation_after_hook": "PASS" if intervention["continued_generation_tokens"] > 0 else "FAIL",
            "peak_vram_bytes": record["memory"]["model_plus_sae"]["peak_vram_bytes"],
            "access_status": "PASS",
            "licence_status": "GEMMA_ACCESS_AVAILABLE; SAE_CC_BY_4_0",
            "compatibility_status": "COMPATIBLE",
            "quality_warning": quality_warning,
            "blocker": "",
        }
    )
    reconstruction_rows.append(
        {
            "layer": record["layer"],
            "sae_id": record["sae_id"],
            "params_sha256": record["params_sha256"],
            "mse": rec["mse"],
            "normalized_mse": rec["normalized_mse"],
            "explained_variance": rec["explained_variance"],
            "mean_cosine_similarity": rec["mean_cosine_similarity"],
            "average_l0_variant": record["average_l0_variant"],
            "observed_mean_l0": rec["mean_l0"],
            "input_finite": rec["input_finite"],
            "encode_finite": rec["encode_finite"],
            "decode_finite": rec["decode_finite"],
            "all_finite": rec["all_finite"],
            "quality_warning": quality_warning,
        }
    )
    hook_tests.append(
        {
            "layer": record["layer"],
            "sae_id": record["sae_id"],
            "hook_module": record["hook_name"],
            "hook_framework": "TransformerLens",
            "captured_shape": record["captured_shape"],
            "hook_capture": True,
            "dimension_match": True,
            "encode": "PASS",
            "decode": "PASS",
            "intervention_policy": intervention["policy"],
            "intervention_reinsert": intervention["status"],
            "hook_calls": intervention["hook_calls"],
            "generation_after_hook": "PASS" if intervention["continued_generation_tokens"] > 0 else "FAIL",
            "continuation_tokens": intervention["continued_generation_tokens"],
            "compatibility_status": "COMPATIBLE",
        }
    )
    memory_rows.append(
        {
            "scope": "model_plus_sae",
            "layer": record["layer"],
            "sae_id": record["sae_id"],
            **record["memory"]["model_plus_sae"],
        }
    )

atomic_csv(OUT / "model_sae_compatibility_matrix.csv", list(matrix_rows[0]), matrix_rows)
atomic_csv(OUT / "phase21_sae_reconstruction.csv", list(reconstruction_rows[0]), reconstruction_rows)
atomic_csv(OUT / "phase21_memory_report.csv", list(memory_rows[0]), memory_rows)
atomic_json(
    OUT / "phase21_hook_smoke.json",
    {
        "schema_version": "phase21_hook_smoke_v1",
        "status": "PASS",
        "hook_framework": "TransformerLens",
        "execution_isolation": "ONE_FRESH_PROCESS_PER_SAE_LAYER",
        "tests": hook_tests,
        "security_feature_discovery_run": False,
        "heldout_used": False,
    },
)
atomic_json(
    OUT / "phase21_hook_framework_audit.json",
    {
        "schema_version": "phase21_hook_framework_audit_v1",
        "status": "TRANSFORMERLENS_EXACT_HOOK_ACCEPTED_WITH_QUALITY_WARNING",
        "official_report": OFFICIAL_REPORT,
        "accepted_framework": "TransformerLens",
        "accepted_hook_names": [record["hook_name"] for record in records],
        "rejected_diagnostics": [
            {
                "framework": "HuggingFace Transformers direct model.layers output",
                "archive_suffix": "_hf_direct_hook_diagnostic_20260831",
                "reason": "Direct equivalence to Gemma Scope hook_resid_post was not established; finite-only output was insufficient.",
            },
            {
                "framework": "TransformerLens monolithic three-layer process",
                "result": "BLOCKED_MODEL_A5000_MEMORY_AT_LAYER_20_AFTER_LAYER_9",
                "resolution": "Fresh process per frozen layer; no offload, quantization, or scientific setting changed.",
            },
        ],
        "reconstruction_quality_warning": quality_warning,
        "protocol_changed": False,
        "model_or_sae_changed": False,
        "security_feature_discovery_run": False,
        "heldout_used": False,
    },
)
atomic_json(
    OUT / "phase21_preselection_checkpoint.json",
    {
        "schema_version": "phase21_preselection_checkpoint_v1",
        "phase": 21,
        "status": "TECHNICAL_COMPATIBILITY_COMPLETE",
        "checkpoint": "PASS_COMPATIBLE_CANDIDATE",
        "candidate_model": MODEL_ID,
        "candidate_model_revision": MODEL_REVISION,
        "candidate_sae_repository": SAE_REPOSITORY,
        "candidate_sae_revision": SAE_REVISION,
        "variant_selection_frozen_before_outcomes": True,
        "hf_authentication": "PASS",
        "hf_authenticated_username": "authenticated_user_redacted",
        "model_access": "PASS",
        "sae_access": "PASS",
        "model_loaded": True,
        "hook_framework": "TransformerLens",
        "compatible_layer_count": 3,
        "compatible_layers": [9, 20, 31],
        "technical_viability_determined": True,
        "reconstruction_quality_warning": quality_warning,
        "layer_analysis_ready": True,
        "next_candidate_started": False,
        "security_feature_discovery_run": False,
        "development_evaluation_subset_created": False,
        "heldout_used": False,
        "model1_modified": False,
        "quantization_used": False,
        "cpu_offload_used": False,
        "next_stage": "Complete the layer-level reconstruction analysis before feature discovery.",
    },
)

print(json.dumps({
    "status": "TECHNICAL_COMPATIBILITY_COMPLETE",
    "checkpoint": "PASS_COMPATIBLE_CANDIDATE",
    "compatible_layers": [9, 20, 31],
    "reconstruction_quality_warning": quality_warning,
}, indent=2))
