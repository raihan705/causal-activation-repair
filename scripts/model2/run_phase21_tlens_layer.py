#!/usr/bin/env python
"""Run one frozen Gemma Scope layer in a fresh process for A5000 isolation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
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
    atomic_json,
    current_memory,
    finite_tensor,
    load_exact_sae,
    reconstruction_metrics,
    sha256_file,
)


def blocked(path: Path, layer: int, sae_id: str, status: str, detail: str) -> int:
    atomic_json(
        path,
        {
            "schema_version": "phase21_tlens_layer_technical_v1",
            "status": status,
            "layer": layer,
            "sae_id": sae_id,
            "detail": detail,
            "quantization_used": False,
            "cpu_offload_used": False,
            "security_feature_discovery_run": False,
            "heldout_used": False,
            "model1_modified": False,
        },
    )
    return 3 if status == "BLOCKED_MODEL_A5000_MEMORY" else 2


def run(layer: int, output_dir: Path) -> int:
    spec = next(item for item in SAE_SPECS if int(item["layer"]) == layer)
    sae_id = str(spec["sae_id"])
    result_path = output_dir / f"phase21_layer_{layer}_technical.json"
    if not torch.cuda.is_available():
        return blocked(result_path, layer, sae_id, "BLOCKED_NO_CUDA", "PyTorch CUDA unavailable")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    hook_name = f"blocks.{layer}.hook_resid_post"

    config_path = hf_hub_download(MODEL_ID, "config.json", revision=MODEL_REVISION)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": SMOKE_TEXT}],
        tokenize=False,
        add_generation_prompt=True,
    )
    tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEVICE)

    try:
        load_started = time.perf_counter()
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
        model_load_seconds = time.perf_counter() - load_started
        parameter_devices = sorted({str(parameter.device) for parameter in model.parameters()})
        if parameter_devices != [DEVICE]:
            return blocked(result_path, layer, sae_id, "FAIL_UNAUTHORIZED_OFFLOAD", str(parameter_devices))

        with torch.inference_mode():
            base_logits = model(tokens, return_type="logits")
            base_generated = model.generate(
                tokens, do_sample=False, max_new_tokens=8,
                prepend_bos=False, verbose=False,
            )
        input_tokens = int(tokens.shape[-1])
        base_generation_tokens = int(base_generated.shape[-1] - input_tokens)
        base_pass = bool(finite_tensor(base_logits) and base_generation_tokens > 0)
        model_only_memory = current_memory()

        torch.cuda.reset_peak_memory_stats(0)
        params_path = hf_hub_download(
            repo_id=SAE_REPOSITORY,
            filename=f"{sae_id}/params.npz",
            revision=SAE_REVISION,
        )
        params_hash = sha256_file(params_path)
        sae, source_keys = load_exact_sae(params_path, layer)
        with torch.inference_mode():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=lambda name: name == hook_name,
                return_type="logits",
            )
        hidden = cache[hook_name]
        metrics = reconstruction_metrics(hidden, sae)
        hook_calls = 0

        def intervention_hook(activation: torch.Tensor, hook: Any) -> torch.Tensor:
            nonlocal hook_calls
            del hook
            hook_calls += 1
            source = activation.to(dtype=torch.float32)
            acts = sae.encode(source)
            modified = acts.clone()
            modified[..., INTERVENTION_FEATURE] += INTERVENTION_DELTA
            return sae.decode(modified).to(dtype=activation.dtype)

        with torch.inference_mode(), model.hooks(fwd_hooks=[(hook_name, intervention_hook)]):
            intervened_logits = model(tokens, return_type="logits")
            continued = model.generate(
                tokens, do_sample=False, max_new_tokens=2,
                prepend_bos=False, verbose=False,
            )
        continuation_tokens = int(continued.shape[-1] - input_tokens)
        model_sae_memory = current_memory()
        intervention_pass = bool(
            finite_tensor(intervened_logits) and hook_calls > 0 and continuation_tokens > 0
        )
        compatible = bool(
            base_pass
            and hidden.shape[-1] == EXPECTED_D_IN
            and sae.cfg.d_in == EXPECTED_D_IN
            and sae.cfg.d_sae == EXPECTED_D_SAE
            and metrics["all_finite"]
            and intervention_pass
        )
        result = {
            "schema_version": "phase21_tlens_layer_technical_v1",
            "status": "COMPATIBLE" if compatible else "INCOMPATIBLE_TECHNICAL",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_config_revision_verified": MODEL_REVISION in str(config_path),
            "model_config": {
                "architectures": config.get("architectures"),
                "hidden_size": config.get("hidden_size"),
                "num_hidden_layers": config.get("num_hidden_layers"),
                "vocab_size": config.get("vocab_size"),
                "model_type": config.get("model_type"),
                "torch_dtype": config.get("torch_dtype"),
            },
            "hook_framework": "TransformerLens",
            "hook_name": hook_name,
            "model_loading_options": {
                "fold_ln": True,
                "center_writing_weights": False,
                "center_unembed": False,
                "dtype": "bfloat16",
                "parameter_devices": parameter_devices,
            },
            "model_load_seconds": model_load_seconds,
            "base_generation": {
                "status": "PASS" if base_pass else "FAIL",
                "prompt_source": "SYNTHETIC_TECHNICAL_SMOKE_NOT_DATASET",
                "prompt_sha256": hashlib.sha256(SMOKE_TEXT.encode("utf-8")).hexdigest(),
                "input_tokens": input_tokens,
                "generated_tokens": base_generation_tokens,
                "logits_finite": finite_tensor(base_logits),
            },
            "sae_repository": SAE_REPOSITORY,
            "sae_revision": SAE_REVISION,
            "sae_id": sae_id,
            "layer": layer,
            "average_l0_variant": int(spec["average_l0"]),
            "params_sha256": params_hash,
            "source_parameter_keys": source_keys,
            "d_in": int(sae.cfg.d_in),
            "d_sae": int(sae.cfg.d_sae),
            "sae_dtype": str(next(sae.parameters()).dtype),
            "captured_shape": list(hidden.shape),
            "reconstruction": metrics,
            "intervention": {
                "policy": "SAE_RECONSTRUCTION_PLUS_FIXED_FEATURE0_DELTA_0.1",
                "feature": INTERVENTION_FEATURE,
                "delta": INTERVENTION_DELTA,
                "hook_calls": hook_calls,
                "logits_finite": finite_tensor(intervened_logits),
                "continued_generation_tokens": continuation_tokens,
                "status": "PASS" if intervention_pass else "FAIL",
            },
            "memory": {
                "model_only": model_only_memory,
                "model_plus_sae": model_sae_memory,
            },
            "quantization_used": False,
            "cpu_offload_used": False,
            "security_feature_discovery_run": False,
            "heldout_used": False,
            "model1_modified": False,
        }
        atomic_json(result_path, result)
        del cache, hidden, sae, model
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps({
            "layer": layer,
            "status": result["status"],
            "mean_l0": metrics["mean_l0"],
            "normalized_mse": metrics["normalized_mse"],
            "peak_vram_bytes": model_sae_memory["peak_vram_bytes"],
        }, indent=2))
        return 0 if compatible else 4
    except torch.cuda.OutOfMemoryError as exc:
        gc.collect()
        torch.cuda.empty_cache()
        return blocked(result_path, layer, sae_id, "BLOCKED_MODEL_A5000_MEMORY", type(exc).__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True, choices=[9, 20, 31])
    parser.add_argument("--output-dir", type=Path, default=Path("revision/model2/phase21/outputs"))
    args = parser.parse_args()
    return run(args.layer, args.output_dir.resolve())


if __name__ == "__main__":
    sys.exit(main())
