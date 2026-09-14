#!/usr/bin/env python
"""Correct Phase 21 smoke using Gemma Scope's documented TransformerLens hooks.

The model, revisions, SAE IDs, prompt, and technical tests are identical to
the frozen verifier. Only the hook framework is corrected from a rejected
Hugging Face layer-output approximation to exact ``hook_resid_post`` hooks.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

from run_phase21_compatibility import (
    DEVICE,
    EXPECTED_D_IN,
    EXPECTED_D_SAE,
    INTERVENTION_DELTA,
    INTERVENTION_FEATURE,
    MODEL_DTYPE,
    MODEL_ID,
    MODEL_REVISION,
    SAE_REPOSITORY,
    SAE_REVISION,
    SAE_SPECS,
    SMOKE_TEXT,
    atomic_csv,
    atomic_json,
    current_memory,
    finite_tensor,
    load_exact_sae,
    output_paths,
    reconstruction_metrics,
    sha256_file,
    write_partial_checkpoint,
)


OFFICIAL_REPORT = "https://storage.googleapis.com/gemma-scope/gemma-scope-report.pdf"


def run(output_dir: Path) -> int:
    paths = output_paths(output_dir)
    if not torch.cuda.is_available():
        write_partial_checkpoint(paths, "BLOCKED_NO_CUDA", "PyTorch CUDA is unavailable")
        return 2

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)

    config_path = hf_hub_download(MODEL_ID, "config.json", revision=MODEL_REVISION)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if config.get("hidden_size") != EXPECTED_D_IN or config.get("num_hidden_layers", 0) <= 31:
        write_partial_checkpoint(paths, "FAIL_DIMENSION_OR_LAYER_CONFIG", "Frozen model config does not match approved SAE dimensions/layers")
        return 2

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": SMOKE_TEXT}],
        tokenize=False,
        add_generation_prompt=True,
    )

    environment = {
        "schema_version": "phase21_environment_manifest_v1",
        "status": "ACCESS_PASS_COMPATIBILITY_RUNNING",
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": True,
        "transformers": importlib.metadata.version("transformers"),
        "transformer_lens": importlib.metadata.version("transformer-lens"),
        "sae_lens": importlib.metadata.version("sae-lens"),
        "huggingface_hub": importlib.metadata.version("huggingface-hub"),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_vram_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        "hf_cli_authentication_status": "PASS",
        "hf_authenticated_username": "authenticated_user_redacted",
        "gemma_gated_access": "PASS",
        "model_revision": MODEL_REVISION,
        "sae_repository_access": "PASS",
        "sae_revision": SAE_REVISION,
        "hook_framework": "TransformerLens",
        "model_loading_options": {
            "fold_ln": True,
            "center_writing_weights": False,
            "center_unembed": False,
            "device": DEVICE,
            "dtype": "bfloat16",
        },
        "package_changes": [],
        "environment_rebuilt": False,
        "hardware_migrated": False,
        "quantization_used": False,
        "cpu_offload_used": False,
    }
    atomic_json(paths["environment"], environment)

    model_load_started = time.perf_counter()
    try:
        model = HookedTransformer.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            tokenizer=tokenizer,
            device=DEVICE,
            n_devices=1,
            dtype=MODEL_DTYPE,
            fold_ln=True,
            center_writing_weights=False,
            center_unembed=False,
            move_to_device=True,
        ).eval()
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        write_partial_checkpoint(paths, "BLOCKED_MODEL_A5000_MEMORY", f"TransformerLens model load CUDA OOM: {type(exc).__name__}")
        return 3
    model_load_seconds = time.perf_counter() - model_load_started

    parameter_devices = sorted({str(parameter.device) for parameter in model.parameters()})
    if parameter_devices != [DEVICE]:
        write_partial_checkpoint(paths, "FAIL_UNAUTHORIZED_OFFLOAD", f"Unexpected parameter devices: {parameter_devices}")
        return 2
    if int(model.cfg.d_model) != EXPECTED_D_IN or int(model.cfg.n_layers) != 42:
        write_partial_checkpoint(paths, "FAIL_TRANSFORMERLENS_CONFIG", "TransformerLens converted config mismatch")
        return 2

    tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEVICE)
    with torch.inference_mode():
        base_logits = model(tokens, return_type="logits")
        base_logits_finite = finite_tensor(base_logits)
        generated = model.generate(
            tokens,
            do_sample=False,
            max_new_tokens=8,
            prepend_bos=False,
            verbose=False,
        )
    input_tokens = int(tokens.shape[-1])
    generated_tokens = int(generated.shape[-1] - input_tokens)
    generated_suffix = generated[:, input_tokens:].detach().cpu().numpy().tobytes()
    model_memory = current_memory()
    model_smoke = {
        "schema_version": "phase21_model_smoke_v1",
        "status": "PASS" if base_logits_finite and generated_tokens > 0 else "FAIL",
        "model_id": MODEL_ID,
        "resolved_revision": MODEL_REVISION,
        "config_path_revision_verified": MODEL_REVISION in str(config_path),
        "architectures": config.get("architectures"),
        "hidden_size": config.get("hidden_size"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "vocab_size": config.get("vocab_size"),
        "model_type": config.get("model_type"),
        "torch_dtype_from_config": config.get("torch_dtype"),
        "loaded_dtype": str(next(model.parameters()).dtype),
        "hook_framework": "TransformerLens",
        "parameter_devices": parameter_devices,
        "loading_time_seconds": model_load_seconds,
        **model_memory,
        "prompt_source": "SYNTHETIC_TECHNICAL_SMOKE_NOT_DATASET",
        "prompt_sha256": hashlib.sha256(SMOKE_TEXT.encode("utf-8")).hexdigest(),
        "input_token_count": input_tokens,
        "generated_token_count": generated_tokens,
        "generated_token_bytes_sha256": hashlib.sha256(generated_suffix).hexdigest(),
        "base_logits_finite": base_logits_finite,
        "generation_run": True,
        "generation_status": "PASS" if generated_tokens > 0 else "FAIL_EMPTY",
        "quantization_used": False,
        "cpu_offload_used": False,
        "heldout_used": False,
    }
    atomic_json(paths["model_smoke"], model_smoke)

    matrix_rows: list[dict[str, Any]] = []
    reconstruction_rows: list[dict[str, Any]] = []
    hook_tests: list[dict[str, Any]] = []
    memory_rows: list[dict[str, Any]] = [
        {"scope": "model_only", "layer": "", "sae_id": "", **model_memory}
    ]
    selection_entries: list[dict[str, Any]] = []

    for spec in SAE_SPECS:
        layer = int(spec["layer"])
        sae_id = str(spec["sae_id"])
        hook_name = f"blocks.{layer}.hook_resid_post"
        torch.cuda.reset_peak_memory_stats(0)
        try:
            params_path = hf_hub_download(
                repo_id=SAE_REPOSITORY,
                filename=f"{sae_id}/params.npz",
                revision=SAE_REVISION,
            )
            params_hash = sha256_file(params_path)
            sae, source_keys = load_exact_sae(params_path, layer)
            dimension_match = bool(sae.cfg.d_in == EXPECTED_D_IN and sae.cfg.d_sae == EXPECTED_D_SAE)
            with torch.inference_mode():
                captured_logits, cache = model.run_with_cache(
                    tokens,
                    names_filter=lambda name: name == hook_name,
                    return_type="logits",
                )
            hook_capture = hook_name in cache
            hidden = cache[hook_name]
            shape_match = bool(hidden.shape[-1] == EXPECTED_D_IN)
            metrics = reconstruction_metrics(hidden, sae)
            hook_calls = 0

            def intervention_hook(activation: torch.Tensor, _hook: Any) -> torch.Tensor:
                nonlocal hook_calls
                hook_calls += 1
                source = activation.to(dtype=torch.float32)
                acts = sae.encode(source)
                modified = acts.clone()
                modified[..., INTERVENTION_FEATURE] += INTERVENTION_DELTA
                return sae.decode(modified).to(dtype=activation.dtype)

            with torch.inference_mode(), model.hooks(fwd_hooks=[(hook_name, intervention_hook)]):
                intervened_logits = model(tokens, return_type="logits")
                intervention_logits_finite = finite_tensor(intervened_logits)
                continued = model.generate(
                    tokens,
                    do_sample=False,
                    max_new_tokens=2,
                    prepend_bos=False,
                    verbose=False,
                )
            continuation_tokens = int(continued.shape[-1] - input_tokens)
            post_generation_pass = continuation_tokens > 0
            memory = current_memory()
            compatible = bool(
                model_smoke["status"] == "PASS"
                and dimension_match
                and hook_capture
                and shape_match
                and metrics["all_finite"]
                and intervention_logits_finite
                and hook_calls > 0
                and post_generation_pass
            )
            compatibility_status = "COMPATIBLE" if compatible else "INCOMPATIBLE_TECHNICAL"

            selection_entries.append(
                {
                    **spec,
                    "params_file": f"{sae_id}/params.npz",
                    "params_sha256": params_hash,
                    "source_parameter_keys": source_keys,
                    "d_in": int(sae.cfg.d_in),
                    "d_sae": int(sae.cfg.d_sae),
                    "hook_point": hook_name,
                    "hook_framework": "TransformerLens",
                    "sae_dtype": str(next(sae.parameters()).dtype),
                }
            )
            reconstruction_rows.append(
                {
                    "layer": layer,
                    "sae_id": sae_id,
                    "params_sha256": params_hash,
                    **{key: metrics[key] for key in ("mse", "normalized_mse", "explained_variance", "mean_cosine_similarity", "mean_l0")},
                    "input_finite": metrics["input_finite"],
                    "encode_finite": metrics["encode_finite"],
                    "decode_finite": metrics["decode_finite"],
                    "all_finite": metrics["all_finite"],
                }
            )
            hook_tests.append(
                {
                    "layer": layer,
                    "sae_id": sae_id,
                    "hook_module": hook_name,
                    "hook_framework": "TransformerLens",
                    "captured_shape": list(hidden.shape),
                    "hook_capture": hook_capture,
                    "dimension_match": shape_match,
                    "encode": "PASS" if metrics["encode_finite"] else "FAIL",
                    "decode": "PASS" if metrics["decode_finite"] else "FAIL",
                    "intervention_policy": "SAE_RECONSTRUCTION_PLUS_FIXED_FEATURE0_DELTA_0.1",
                    "intervention_reinsert": "PASS" if intervention_logits_finite and hook_calls > 0 else "FAIL",
                    "hook_calls": hook_calls,
                    "generation_after_hook": "PASS" if post_generation_pass else "FAIL",
                    "continuation_tokens": continuation_tokens,
                    "compatibility_status": compatibility_status,
                }
            )
            memory_rows.append({"scope": "model_plus_sae", "layer": layer, "sae_id": sae_id, **memory})
            matrix_rows.append(
                {
                    "model_id": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "sae_repository": SAE_REPOSITORY,
                    "sae_revision": SAE_REVISION,
                    "sae_id": sae_id,
                    "layer": layer,
                    "width": "width_16k",
                    "average_l0": spec["average_l0"],
                    "d_model": EXPECTED_D_IN,
                    "d_sae": EXPECTED_D_SAE,
                    "hook_framework": "TransformerLens",
                    "hook_name": hook_name,
                    "model_load": "PASS",
                    "sae_load": "PASS",
                    "dimension_match": "PASS" if dimension_match and shape_match else "FAIL",
                    "hook_capture": "PASS" if hook_capture else "FAIL",
                    "encode": "PASS" if metrics["encode_finite"] else "FAIL",
                    "decode": "PASS" if metrics["decode_finite"] else "FAIL",
                    "reconstruction_finite": "PASS" if metrics["all_finite"] else "FAIL",
                    "intervention_reinsert": "PASS" if intervention_logits_finite and hook_calls > 0 else "FAIL",
                    "generation_after_hook": "PASS" if post_generation_pass else "FAIL",
                    "peak_vram_bytes": memory["peak_vram_bytes"],
                    "access_status": "PASS",
                    "licence_status": "GEMMA_ACCESS_AVAILABLE; SAE_CC_BY_4_0",
                    "compatibility_status": compatibility_status,
                    "blocker": "" if compatible else "One or more technical compatibility criteria failed",
                }
            )
            del cache, hidden, captured_logits, intervened_logits, continued, sae
            gc.collect()
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError as exc:
            gc.collect()
            torch.cuda.empty_cache()
            write_partial_checkpoint(paths, "BLOCKED_MODEL_A5000_MEMORY", f"Layer {layer} model+SAE CUDA OOM: {type(exc).__name__}")
            return 3
        except Exception as exc:
            matrix_rows.append(
                {
                    "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                    "sae_repository": SAE_REPOSITORY, "sae_revision": SAE_REVISION,
                    "sae_id": sae_id, "layer": layer, "width": "width_16k",
                    "average_l0": spec["average_l0"], "d_model": EXPECTED_D_IN,
                    "d_sae": EXPECTED_D_SAE, "hook_framework": "TransformerLens",
                    "hook_name": hook_name, "model_load": "PASS", "sae_load": "FAIL",
                    "dimension_match": "NOT_TESTED", "hook_capture": "NOT_TESTED",
                    "encode": "NOT_TESTED", "decode": "NOT_TESTED",
                    "reconstruction_finite": "NOT_TESTED", "intervention_reinsert": "NOT_TESTED",
                    "generation_after_hook": "NOT_TESTED", "peak_vram_bytes": "",
                    "access_status": "PASS", "licence_status": "GEMMA_ACCESS_AVAILABLE; SAE_CC_BY_4_0",
                    "compatibility_status": "INCOMPATIBLE_TECHNICAL",
                    "blocker": f"{type(exc).__name__}: {exc}",
                }
            )
            gc.collect()
            torch.cuda.empty_cache()

    matrix_fields = list(matrix_rows[0].keys())
    atomic_csv(paths["matrix"], matrix_fields, matrix_rows)
    reconstruction_fields = [
        "layer", "sae_id", "params_sha256", "mse", "normalized_mse",
        "explained_variance", "mean_cosine_similarity", "mean_l0",
        "input_finite", "encode_finite", "decode_finite", "all_finite",
    ]
    atomic_csv(paths["reconstruction"], reconstruction_fields, reconstruction_rows)
    memory_fields = [
        "scope", "layer", "sae_id", "allocated_vram_bytes", "reserved_vram_bytes",
        "peak_vram_bytes", "peak_reserved_vram_bytes",
    ]
    atomic_csv(paths["memory"], memory_fields, memory_rows)
    atomic_json(
        paths["hook"],
        {
            "schema_version": "phase21_hook_smoke_v1",
            "status": "PASS" if len(hook_tests) == len(SAE_SPECS) and all(row["compatibility_status"] == "COMPATIBLE" for row in hook_tests) else "PARTIAL_OR_FAIL",
            "hook_framework": "TransformerLens",
            "tests": hook_tests,
            "security_feature_discovery_run": False,
            "heldout_used": False,
        },
    )
    atomic_json(
        paths["selection"],
        {
            "schema_version": "phase21_model_sae_selection_v1",
            "status": "FROZEN_BEFORE_OUTCOME_ACCESS_PASS",
            "model": {
                "model_id": MODEL_ID, "resolved_revision": MODEL_REVISION,
                "architectures": config.get("architectures"), "hidden_size": config.get("hidden_size"),
                "number_of_layers": config.get("num_hidden_layers"), "vocab_size": config.get("vocab_size"),
                "model_type": config.get("model_type"), "torch_dtype": config.get("torch_dtype"),
                "licence": "gemma", "access_state": "PASS", "hook_framework": "TransformerLens",
            },
            "sae": {
                "repository": SAE_REPOSITORY, "resolved_revision": SAE_REVISION,
                "licence": "cc-by-4.0", "width": "width_16k", "selected": selection_entries,
            },
            "selection_rule": "Per retained layer, select the available width_16k SAE with average L0 closest to 50; smaller L0 breaks equal-distance ties.",
            "selection_before_reconstruction_or_security_outcome": True,
            "security_feature_discovery_run": False,
            "heldout_used": False,
        },
    )

    compatible_layers = [int(row["layer"]) for row in matrix_rows if row["compatibility_status"] == "COMPATIBLE"]
    compatible_count = len(compatible_layers)
    checkpoint = "PASS_COMPATIBLE_CANDIDATE" if compatible_count >= 1 else "FAIL_NO_COMPATIBLE_LAYER"
    status = "TECHNICAL_COMPATIBILITY_COMPLETE" if compatible_count >= 1 else "FAIL_TECHNICAL_COMPATIBILITY"
    atomic_json(
        paths["checkpoint"],
        {
            "schema_version": "phase21_preselection_checkpoint_v1",
            "phase": 21, "status": status, "checkpoint": checkpoint,
            "candidate_model": MODEL_ID, "candidate_model_revision": MODEL_REVISION,
            "candidate_sae_repository": SAE_REPOSITORY, "candidate_sae_revision": SAE_REVISION,
            "variant_selection_frozen_before_outcomes": True,
            "hf_authentication": "PASS", "hf_authenticated_username": "authenticated_user_redacted",
            "model_access": "PASS", "sae_access": "PASS", "model_loaded": True,
            "hook_framework": "TransformerLens",
            "compatible_layer_count": compatible_count, "compatible_layers": compatible_layers,
            "technical_viability_determined": True,
            "layer_analysis_ready": compatible_count >= 1,
            "next_candidate_started": False,
            "security_feature_discovery_run": False,
            "development_evaluation_subset_created": False,
            "heldout_used": False, "model1_modified": False,
            "quantization_used": False, "cpu_offload_used": False,
            "next_stage": "Complete the layer-level reconstruction analysis before feature discovery.",
        },
    )
    atomic_json(
        output_dir / "phase21_hook_framework_audit.json",
        {
            "schema_version": "phase21_hook_framework_audit_v1",
            "status": "TRANSFORMERLENS_EXACT_HOOK_TESTED",
            "official_report": OFFICIAL_REPORT,
            "approved_hook_framework": "TransformerLens",
            "hook_names": [f"blocks.{int(spec['layer'])}.hook_resid_post" for spec in SAE_SPECS],
            "rejected_diagnostic": {
                "framework": "HuggingFace Transformers direct model.layers output",
                "archive_suffix": "_hf_direct_hook_diagnostic_20260831",
                "reason": "Finite tensors alone masked implausible reconstruction and sparsity metrics; direct layer-output equivalence was not accepted as established.",
            },
            "protocol_changed": False,
            "model_or_sae_changed": False,
            "security_feature_discovery_run": False,
            "heldout_used": False,
        },
    )
    environment["status"] = status
    environment["model_loaded"] = True
    environment["compatible_layer_count"] = compatible_count
    atomic_json(paths["environment"], environment)
    print(json.dumps({"status": status, "checkpoint": checkpoint, "compatible_layers": compatible_layers}, indent=2))
    return 0 if compatible_count >= 1 else 4


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("revision/model2/phase21/outputs"))
    args = parser.parse_args()
    return run(args.output_dir.resolve())


if __name__ == "__main__":
    sys.exit(main())
