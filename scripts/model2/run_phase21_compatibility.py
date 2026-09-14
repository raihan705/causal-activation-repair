#!/usr/bin/env python
"""Frozen Phase 21 Gemma 2 / Gemma Scope technical compatibility verifier.

This runner uses one synthetic technical prompt. It performs no feature
discovery, ranking, security evaluation, or held-out data access.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from huggingface_hub import hf_hub_download
from sae_lens.saes.jumprelu_sae import JumpReLUSAE, JumpReLUSAEConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "google/gemma-2-9b-it"
MODEL_REVISION = "11c9b309abf73637e4b6f9a3fa1e92e615547819"
SAE_REPOSITORY = "google/gemma-scope-9b-it-res"
SAE_REVISION = "e86af97a5b6fbbccca28ab654f2fda1b0768f770"
SAE_SPECS = (
    {"layer": 9, "average_l0": 47, "sae_id": "layer_9/width_16k/average_l0_47"},
    {"layer": 20, "average_l0": 47, "sae_id": "layer_20/width_16k/average_l0_47"},
    {"layer": 31, "average_l0": 43, "sae_id": "layer_31/width_16k/average_l0_43"},
)
DEVICE = "cuda:0"
MODEL_DTYPE = torch.bfloat16
SAE_DTYPE = torch.float32
EXPECTED_D_IN = 3584
EXPECTED_D_SAE = 16384
SMOKE_TEXT = "Return only the integer sum of 2 and 3."
INTERVENTION_FEATURE = 0
INTERVENTION_DELTA = 0.1


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_tensor(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all().item())


def current_memory() -> dict[str, int]:
    return {
        "allocated_vram_bytes": int(torch.cuda.memory_allocated(0)),
        "reserved_vram_bytes": int(torch.cuda.memory_reserved(0)),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(0)),
        "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(0)),
    }


def replace_hidden(output: Any, replacement: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (replacement, *output[1:])
    return replacement


def hidden_from_output(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def load_exact_sae(params_path: str, layer: int) -> tuple[JumpReLUSAE, list[str]]:
    cfg = JumpReLUSAEConfig(
        d_in=EXPECTED_D_IN,
        d_sae=EXPECTED_D_SAE,
        dtype="float32",
        device=DEVICE,
        apply_b_dec_to_input=False,
        normalize_activations="none",
    )
    sae = JumpReLUSAE(cfg).eval()
    state_dict: dict[str, torch.Tensor] = {}
    with np.load(params_path) as data:
        source_keys = sorted(data.files)
        for key in data.files:
            mapped = "W_" + key[2:] if key.startswith("w_") else key
            state_dict[mapped] = torch.from_numpy(np.asarray(data[key])).to(
                device=DEVICE, dtype=SAE_DTYPE
            )
    if "scaling_factor" in state_dict:
        scaling = state_dict.pop("scaling_factor")
        if not torch.allclose(scaling, torch.ones_like(scaling)):
            raise ValueError("Non-unit scaling_factor is incompatible with the frozen loader configuration")
    missing, unexpected = sae.load_state_dict(state_dict, strict=False)
    del state_dict
    gc.collect()
    if missing or unexpected:
        raise ValueError(f"SAE state mismatch at layer {layer}: missing={missing}, unexpected={unexpected}")
    for parameter in sae.parameters():
        parameter.requires_grad_(False)
    return sae, source_keys


def reconstruction_metrics(hidden: torch.Tensor, sae: JumpReLUSAE) -> dict[str, Any]:
    source = hidden.detach().to(dtype=SAE_DTYPE)
    acts = sae.encode(source)
    reconstruction = sae.decode(acts)
    error = reconstruction - source
    centered = source - source.mean(dim=-1, keepdim=True)
    denominator = centered.square().sum().clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(source, reconstruction, dim=-1)
    metrics = {
        "input_shape": list(source.shape),
        "feature_shape": list(acts.shape),
        "reconstruction_shape": list(reconstruction.shape),
        "input_finite": finite_tensor(source),
        "encode_finite": finite_tensor(acts),
        "decode_finite": finite_tensor(reconstruction),
        "mse": float(error.square().mean().item()),
        "normalized_mse": float((error.square().sum() / denominator).item()),
        "explained_variance": float((1.0 - error.square().sum() / denominator).item()),
        "mean_cosine_similarity": float(cosine.mean().item()),
        "mean_l0": float((acts != 0).sum(dim=-1).float().mean().item()),
    }
    metrics["all_finite"] = bool(
        metrics["input_finite"]
        and metrics["encode_finite"]
        and metrics["decode_finite"]
        and all(np.isfinite(metrics[key]) for key in ("mse", "normalized_mse", "explained_variance", "mean_cosine_similarity", "mean_l0"))
    )
    return metrics


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "selection": output_dir / "model_sae_selection_record.json",
        "matrix": output_dir / "model_sae_compatibility_matrix.csv",
        "environment": output_dir / "phase21_environment_manifest.json",
        "model_smoke": output_dir / "phase21_model_smoke.json",
        "reconstruction": output_dir / "phase21_sae_reconstruction.csv",
        "hook": output_dir / "phase21_hook_smoke.json",
        "memory": output_dir / "phase21_memory_report.csv",
        "checkpoint": output_dir / "phase21_preselection_checkpoint.json",
    }


def write_partial_checkpoint(paths: dict[str, Path], status: str, blocker: str) -> None:
    atomic_json(
        paths["checkpoint"],
        {
            "schema_version": "phase21_preselection_checkpoint_v1",
            "phase": 21,
            "status": status,
            "checkpoint": "BLOCKED_MODEL_A5000_MEMORY" if "MEMORY" in status else "FAIL_TECHNICAL_COMPATIBILITY",
            "candidate_model": MODEL_ID,
            "candidate_model_revision": MODEL_REVISION,
            "candidate_sae_repository": SAE_REPOSITORY,
            "candidate_sae_revision": SAE_REVISION,
            "model_loaded": False,
            "compatible_layer_count": 0,
            "security_feature_discovery_run": False,
            "development_evaluation_subset_created": False,
            "heldout_used": False,
            "model1_modified": False,
            "quantization_used": False,
            "cpu_offload_used": False,
            "blocker": blocker,
        },
    )


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

    environment = {
        "schema_version": "phase21_environment_manifest_v1",
        "status": "ACCESS_PASS_COMPATIBILITY_RUNNING",
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": True,
        "transformers": transformers.__version__,
        "sae_lens": __import__("sae_lens").__version__,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_vram_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        "hf_cli_authentication_status": "PASS",
        "hf_authenticated_username": "authenticated_user_redacted",
        "gemma_gated_access": "PASS",
        "model_revision": MODEL_REVISION,
        "sae_repository_access": "PASS",
        "sae_revision": SAE_REVISION,
        "package_changes": [],
        "environment_rebuilt": False,
        "hardware_migrated": False,
        "quantization_used": False,
        "cpu_offload_used": False,
    }
    atomic_json(paths["environment"], environment)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": SMOKE_TEXT}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    model_load_started = time.perf_counter()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            torch_dtype=MODEL_DTYPE,
            low_cpu_mem_usage=True,
            device_map={"": DEVICE},
        ).eval()
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        write_partial_checkpoint(paths, "BLOCKED_MODEL_A5000_MEMORY", f"Model load CUDA OOM: {type(exc).__name__}")
        return 3
    model_load_seconds = time.perf_counter() - model_load_started

    device_map = getattr(model, "hf_device_map", None)
    if device_map is not None:
        mapped = {str(value) for value in device_map.values()}
        if any("cpu" in value or "disk" in value for value in mapped):
            write_partial_checkpoint(paths, "FAIL_UNAUTHORIZED_OFFLOAD", f"Unexpected device map: {device_map}")
            return 2

    with torch.inference_mode():
        base = model(**inputs, use_cache=False)
        base_logits_finite = finite_tensor(base.logits)
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=8,
            pad_token_id=tokenizer.eos_token_id,
        )
    input_tokens = int(inputs["input_ids"].shape[-1])
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
            captured: dict[str, torch.Tensor] = {}

            def capture_hook(_module: Any, _args: Any, output: Any) -> Any:
                captured["hidden"] = hidden_from_output(output).detach()
                return output

            capture_handle = model.model.layers[layer].register_forward_hook(capture_hook)
            with torch.inference_mode():
                capture_output = model(**inputs, use_cache=False)
            capture_handle.remove()
            hook_capture = "hidden" in captured
            hidden = captured["hidden"]
            shape_match = bool(hidden.shape[-1] == EXPECTED_D_IN)
            metrics = reconstruction_metrics(hidden, sae)

            hook_calls = 0

            def intervention_hook(_module: Any, _args: Any, output: Any) -> Any:
                nonlocal hook_calls
                hook_calls += 1
                original = hidden_from_output(output)
                source = original.to(dtype=SAE_DTYPE)
                acts = sae.encode(source)
                modified = acts.clone()
                modified[..., INTERVENTION_FEATURE] += INTERVENTION_DELTA
                replacement = sae.decode(modified).to(dtype=original.dtype)
                return replace_hidden(output, replacement)

            intervention_handle = model.model.layers[layer].register_forward_hook(intervention_hook)
            with torch.inference_mode():
                intervened = model(**inputs, use_cache=False)
                intervention_logits_finite = finite_tensor(intervened.logits)
                continued = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=2,
                    pad_token_id=tokenizer.eos_token_id,
                )
            intervention_handle.remove()
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
                    "hook_point_sae_lens": f"blocks.{layer}.hook_resid_post",
                    "hook_point_transformers": f"model.layers.{layer}.output",
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
                    "hook_module": f"model.layers.{layer}",
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
            del captured, hidden, capture_output, intervened, continued, sae
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
                    "model_load": "PASS",
                    "sae_load": "FAIL",
                    "dimension_match": "NOT_TESTED",
                    "hook_capture": "NOT_TESTED",
                    "encode": "NOT_TESTED",
                    "decode": "NOT_TESTED",
                    "reconstruction_finite": "NOT_TESTED",
                    "intervention_reinsert": "NOT_TESTED",
                    "generation_after_hook": "NOT_TESTED",
                    "peak_vram_bytes": "",
                    "access_status": "PASS",
                    "licence_status": "GEMMA_ACCESS_AVAILABLE; SAE_CC_BY_4_0",
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
            "status": "PASS" if all(row["compatibility_status"] == "COMPATIBLE" for row in hook_tests) else "PARTIAL_OR_FAIL",
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
                "model_id": MODEL_ID,
                "resolved_revision": MODEL_REVISION,
                "architectures": config.get("architectures"),
                "hidden_size": config.get("hidden_size"),
                "number_of_layers": config.get("num_hidden_layers"),
                "vocab_size": config.get("vocab_size"),
                "model_type": config.get("model_type"),
                "torch_dtype": config.get("torch_dtype"),
                "licence": "gemma",
                "access_state": "PASS",
            },
            "sae": {
                "repository": SAE_REPOSITORY,
                "resolved_revision": SAE_REVISION,
                "licence": "cc-by-4.0",
                "width": "width_16k",
                "selected": selection_entries,
            },
            "selection_rule": "Per retained layer, select the available width_16k SAE with average L0 closest to 50; smaller L0 breaks equal-distance ties.",
            "selection_before_reconstruction_or_security_outcome": True,
            "security_feature_discovery_run": False,
            "heldout_used": False,
        },
    )

    compatible_count = sum(row["compatibility_status"] == "COMPATIBLE" for row in matrix_rows)
    checkpoint = "PASS_COMPATIBLE_CANDIDATE" if compatible_count >= 1 else "FAIL_NO_COMPATIBLE_LAYER"
    status = "TECHNICAL_COMPATIBILITY_COMPLETE" if compatible_count >= 1 else "FAIL_TECHNICAL_COMPATIBILITY"
    atomic_json(
        paths["checkpoint"],
        {
            "schema_version": "phase21_preselection_checkpoint_v1",
            "phase": 21,
            "status": status,
            "checkpoint": checkpoint,
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
            "compatible_layer_count": compatible_count,
            "compatible_layers": [int(row["layer"]) for row in matrix_rows if row["compatibility_status"] == "COMPATIBLE"],
            "technical_viability_determined": True,
            "layer_analysis_ready": compatible_count >= 1,
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
    environment["status"] = status
    environment["model_loaded"] = True
    environment["compatible_layer_count"] = compatible_count
    atomic_json(paths["environment"], environment)
    print(json.dumps({"status": status, "checkpoint": checkpoint, "compatible_layers": [int(row["layer"]) for row in matrix_rows if row["compatibility_status"] == "COMPATIBLE"]}, indent=2))
    return 0 if compatible_count >= 1 else 4


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("revision/model2/phase21/outputs"),
    )
    args = parser.parse_args()
    return run(args.output_dir.resolve())


if __name__ == "__main__":
    sys.exit(main())
