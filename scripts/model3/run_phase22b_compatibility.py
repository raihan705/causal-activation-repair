#!/usr/bin/env python
"""Run one clean-process Phase 22B Qwen/BatchTopK technical compatibility test.

This uses one fixed synthetic technical prompt. It performs no benchmark
generation, scanner execution, feature discovery, ranking, or held-out access.
Run one frozen SAE layer per fresh process.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
SAE_REPOSITORY = "andyrdt/saes-qwen2.5-7b-instruct"
SAE_REVISION = "c37e53c4bb07127ad17ab88f28b93d4e87142e59"
LOADER_REPOSITORY = "https://github.com/andyrdt/dictionary_learning"
LOADER_REVISION = "de9138b02fffdf9919c53cee828beb4e05049741"
DEVICE = "cuda:0"
MODEL_DTYPE = torch.bfloat16
SAE_DTYPE = torch.float32
EXPECTED_D_IN = 3584
EXPECTED_D_SAE = 131072
EXPECTED_K = 32
BOS_OFFSET = 8
OUTLIER_NORM_FACTOR = 10.0
SMOKE_TEXT = "Return only the integer sum of 2 and 3."
SMOKE_TEXT_SHA256 = hashlib.sha256(SMOKE_TEXT.encode("utf-8")).hexdigest()
LAYERS = (7, 15, 23)
EXPECTED_SAE_SHA256 = {
    7: "e9c88dc39dd89bc80bcea688c91739a37d1ec55bd7bf676f9cfb4bbf847fbd73",
    15: "1339c52258e95bc64535a90e45067187722e659e0ee03c976db813614b29ab5b",
    23: "a593d0da4a10cde4674b48749bf573b14e40119c1cff34cf2d499f334940ef4b",
}
EXPECTED_CONFIG_SHA256 = {
    7: "b9d2d76eea9fb3eb3c9ca51f8ed319ca118de6ac455c30da4990a8f8aa320dee",
    15: "b61f27dca073d4eb069580ba705f998cbb7c58fc85033bce6b9d1e26a4dcd3e2",
    23: "2a60e761525b555d0e0ea76562e5f98394eb21447bab2034d5a2491bfbb03841",
}
EXPECTED_EVAL_SHA256 = {
    7: "f82a1bb704ebdcc7ad05af0abd58d726a0f29890aa176d3a986981e9e93cc918",
    15: "a4b2ce403137cee3d4e4f452348af9297a51ef3eae689a2b7d8f94edcebc1f93",
    23: "83686762ff6586a31b38e093b4b8f86f827cba85537568073da977dfcd5f9884",
}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all().item())


def memory_snapshot() -> dict[str, int]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    return {
        "allocated_vram_bytes": int(torch.cuda.memory_allocated(0)),
        "reserved_vram_bytes": int(torch.cuda.memory_reserved(0)),
        "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(0)),
        "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(0)),
        "free_vram_bytes": int(free_bytes),
        "total_vram_bytes": int(total_bytes),
    }


def causal_lm_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> float:
    shifted_logits = logits[:, :-1, :].float().contiguous()
    shifted_labels = input_ids[:, 1:].contiguous()
    return float(
        F.cross_entropy(
            shifted_logits.view(-1, shifted_logits.shape[-1]),
            shifted_labels.view(-1),
        ).item()
    )


def hidden_from_output(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Unsupported decoder-layer output type: {type(output)!r}")
    return output


def replace_hidden(output: Any, replacement: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (replacement, *output[1:])
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Unsupported decoder-layer output type: {type(output)!r}")
    return replacement


class BatchTopKSAE(nn.Module):
    """Inference-equivalent subset of the pinned upstream BatchTopKSAE.

    Source: dictionary_learning/trainers/batch_top_k.py at LOADER_REVISION.
    Training-only code and nnsight imports are intentionally excluded.
    """

    def __init__(self, activation_dim: int, dict_size: int, k: int) -> None:
        super().__init__()
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.register_buffer("k", torch.tensor(k, dtype=torch.int))
        self.register_buffer("threshold", torch.tensor(-1.0, dtype=torch.float32))
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        self.encoder = nn.Linear(activation_dim, dict_size)
        self.b_dec = nn.Parameter(torch.zeros(activation_dim))

    def encode(self, x: torch.Tensor, use_threshold: bool = True) -> torch.Tensor:
        post_relu = F.relu(self.encoder(x - self.b_dec))
        if use_threshold:
            return post_relu * (post_relu > self.threshold)
        flattened = post_relu.flatten()
        count = int(self.k.item()) * x.size(0)
        topk = flattened.topk(count, sorted=False, dim=-1)
        return torch.zeros_like(flattened).scatter_(-1, topk.indices, topk.values).reshape(post_relu.shape)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(features) + self.b_dec


def validate_config(layer: int, config: dict[str, Any]) -> None:
    trainer = config.get("trainer", {})
    buffer = config.get("buffer", {})
    expected = {
        "trainer_class": "BatchTopKTrainer",
        "dict_class": "BatchTopKSAE",
        "activation_dim": EXPECTED_D_IN,
        "dict_size": EXPECTED_D_SAE,
        "k": EXPECTED_K,
        "layer": layer,
        "lm_name": MODEL_ID,
        "submodule_name": f"resid_post_layer_{layer}",
    }
    mismatches = {key: {"expected": value, "observed": trainer.get(key)} for key, value in expected.items() if trainer.get(key) != value}
    if buffer.get("d_submodule") != EXPECTED_D_IN:
        mismatches["buffer.d_submodule"] = {"expected": EXPECTED_D_IN, "observed": buffer.get("d_submodule")}
    if buffer.get("io") != "out":
        mismatches["buffer.io"] = {"expected": "out", "observed": buffer.get("io")}
    if mismatches:
        raise ValueError(f"Frozen SAE config mismatch: {mismatches}")


def load_sae(weight_path: Path) -> tuple[BatchTopKSAE, list[str]]:
    state = torch.load(weight_path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, dict):
        raise TypeError("SAE checkpoint is not a state dictionary")
    keys = sorted(state)
    required = {"encoder.weight", "encoder.bias", "decoder.weight", "b_dec", "k", "threshold"}
    if set(keys) != required:
        raise ValueError(f"Unexpected SAE state keys: {keys}")
    if tuple(state["encoder.weight"].shape) != (EXPECTED_D_SAE, EXPECTED_D_IN):
        raise ValueError(f"Unexpected encoder shape: {tuple(state['encoder.weight'].shape)}")
    if tuple(state["decoder.weight"].shape) != (EXPECTED_D_IN, EXPECTED_D_SAE):
        raise ValueError(f"Unexpected decoder shape: {tuple(state['decoder.weight'].shape)}")
    if int(state["k"].item()) != EXPECTED_K:
        raise ValueError(f"Unexpected k in checkpoint: {state['k'].item()}")
    sae = BatchTopKSAE(EXPECTED_D_IN, EXPECTED_D_SAE, EXPECTED_K)
    sae.load_state_dict(state, strict=True)
    del state
    gc.collect()
    sae = sae.to(device=DEVICE, dtype=SAE_DTYPE).eval()
    for parameter in sae.parameters():
        parameter.requires_grad_(False)
    return sae, keys


def preflight(layer: int, output_dir: Path) -> int:
    result = {
        "schema_version": "phase22b_layer_preflight_v1",
        "status": "PASS_APPROVED",
        "layer": layer,
        "model": {"repository": MODEL_ID, "revision": MODEL_REVISION},
        "sae": {
            "repository": SAE_REPOSITORY,
            "revision": SAE_REVISION,
            "weight_path": f"resid_post_layer_{layer}/trainer_0/ae.pt",
            "expected_weight_sha256": EXPECTED_SAE_SHA256[layer],
        },
        "loader": {"repository": LOADER_REPOSITORY, "revision": LOADER_REVISION},
        "smoke_prompt_sha256": SMOKE_TEXT_SHA256,
        "model_downloaded": False,
        "sae_downloaded": False,
        "model_loaded": False,
        "sae_loaded": False,
        "gpu_execution": False,
        "benchmark_generation": False,
        "scanner_run": False,
        "heldout_used": False,
    }
    atomic_json(output_dir / f"phase22b_layer_{layer}_preflight.json", result)
    print(json.dumps(result, indent=2))
    return 0


def run(layer: int, output_dir: Path) -> int:
    output_path = output_dir / f"phase22b_layer_{layer}_technical.json"
    started = time.perf_counter()
    base_record: dict[str, Any] = {
        "schema_version": "phase22b_layer_technical_v1",
        "status": "IN_PROGRESS",
        "layer": layer,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "sae_repository": SAE_REPOSITORY,
        "sae_revision": SAE_REVISION,
        "sae_id": f"resid_post_layer_{layer}/trainer_0",
        "loader_revision": LOADER_REVISION,
        "smoke_prompt_source": "SYNTHETIC_TECHNICAL_SMOKE_NOT_DATASET",
        "smoke_prompt_sha256": SMOKE_TEXT_SHA256,
        "benchmark_generation": False,
        "scanner_run": False,
        "heldout_used": False,
        "quantization_used": False,
        "cpu_offload_used": False,
    }
    atomic_json(output_path, base_record)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("BLOCKED_NO_CUDA")
        torch.cuda.set_device(DEVICE)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)

        paths: dict[str, Path] = {}
        remote_names = {
            "config": f"resid_post_layer_{layer}/trainer_0/config.json",
            "eval": f"resid_post_layer_{layer}/trainer_0/eval_results.json",
            "weight": f"resid_post_layer_{layer}/trainer_0/ae.pt",
        }
        for key, filename in remote_names.items():
            paths[key] = Path(hf_hub_download(SAE_REPOSITORY, filename, revision=SAE_REVISION))
        observed_hashes = {key: sha256_file(path) for key, path in paths.items()}
        expected_hashes = {
            "config": EXPECTED_CONFIG_SHA256[layer],
            "eval": EXPECTED_EVAL_SHA256[layer],
            "weight": EXPECTED_SAE_SHA256[layer],
        }
        if observed_hashes != expected_hashes:
            raise ValueError(f"SAE hash mismatch: expected={expected_hashes}, observed={observed_hashes}")
        sae_config = json.loads(paths["config"].read_text(encoding="utf-8"))
        upstream_eval = json.loads(paths["eval"].read_text(encoding="utf-8"))
        validate_config(layer, sae_config)

        model_config_path = Path(hf_hub_download(MODEL_ID, "config.json", revision=MODEL_REVISION))
        model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
        if MODEL_REVISION not in str(model_config_path):
            raise ValueError("Model config cache path does not contain the pinned revision")
        if model_config.get("hidden_size") != EXPECTED_D_IN or model_config.get("num_hidden_layers", 0) <= layer:
            raise ValueError("Pinned model dimension/layer mismatch")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": SMOKE_TEXT}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(DEVICE) for key, value in encoded.items()}

        model_load_start = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            torch_dtype=MODEL_DTYPE,
            low_cpu_mem_usage=True,
            device_map={"": DEVICE},
        ).eval()
        model_load_seconds = time.perf_counter() - model_load_start
        device_map = getattr(model, "hf_device_map", None)
        if device_map is not None and any("cpu" in str(v) or "disk" in str(v) for v in device_map.values()):
            raise RuntimeError(f"BLOCKED_UNAUTHORIZED_OFFLOAD: {device_map}")
        parameter_devices = sorted({str(parameter.device) for parameter in model.parameters()})
        if parameter_devices != [DEVICE]:
            raise RuntimeError(f"BLOCKED_MODEL_NOT_ENTIRELY_CUDA: {parameter_devices}")
        if next(model.parameters()).dtype != MODEL_DTYPE:
            raise RuntimeError(f"BLOCKED_MODEL_DTYPE: {next(model.parameters()).dtype}")
        model_memory = memory_snapshot()

        captured: dict[str, torch.Tensor] = {}

        def capture_hook(_module: Any, _args: Any, output: Any) -> Any:
            captured["hidden"] = hidden_from_output(output).detach()
            return output

        capture_handle = model.model.layers[layer].register_forward_hook(capture_hook)
        with torch.inference_mode():
            baseline = model(**inputs, use_cache=False)
            baseline_loss = causal_lm_loss(baseline.logits, inputs["input_ids"])
        capture_handle.remove()
        with torch.inference_mode():
            baseline_generation = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=8,
                pad_token_id=tokenizer.pad_token_id,
            )
        if "hidden" not in captured:
            raise RuntimeError("Decoder-layer output hook did not capture activations")
        hidden = captured["hidden"]
        if hidden.shape[-1] != EXPECTED_D_IN:
            raise ValueError(f"Captured hidden dimension mismatch: {list(hidden.shape)}")

        sae_load_start = time.perf_counter()
        sae, state_keys = load_sae(paths["weight"])
        sae_load_seconds = time.perf_counter() - sae_load_start
        sae_devices = sorted({str(value.device) for value in [*sae.parameters(), *sae.buffers()]})
        if sae_devices != [DEVICE]:
            raise RuntimeError(f"BLOCKED_SAE_NOT_ENTIRELY_CUDA: {sae_devices}")
        sae_memory = memory_snapshot()

        with torch.inference_mode():
            source = hidden.to(dtype=SAE_DTYPE)
            metric_source = source[:, BOS_OFFSET:, :].reshape(-1, EXPECTED_D_IN)
            norms = torch.linalg.vector_norm(metric_source, dim=-1)
            median_norm = torch.median(norms)
            inlier_mask = norms <= OUTLIER_NORM_FACTOR * median_norm
            metric_source = metric_source[inlier_mask]
            features = sae.encode(metric_source)
            reconstruction = sae.decode(features)
            error = reconstruction - metric_source
            total_variance = torch.var(metric_source, dim=0).sum().clamp_min(1e-12)
            residual_variance = torch.var(error, dim=0).sum()
            explained_variance = float((1.0 - residual_variance / total_variance).item())
            cosine = F.cosine_similarity(metric_source, reconstruction, dim=-1)
            reconstruction_metrics = {
                "captured_input_shape": list(source.shape),
                "metric_input_shape": list(metric_source.shape),
                "feature_shape": list(features.shape),
                "reconstruction_shape": list(reconstruction.shape),
                "input_finite": finite(metric_source),
                "encode_finite": finite(features),
                "decode_finite": finite(reconstruction),
                "mse": float(error.square().mean().item()),
                "normalized_mse": float((residual_variance / total_variance).item()),
                "explained_variance": explained_variance,
                "mean_cosine_similarity": float(cosine.mean().item()),
                "mean_l0": float((features != 0).sum(dim=-1).float().mean().item()),
                "bos_tokens_excluded": BOS_OFFSET,
                "pre_outlier_filter_token_count": int(norms.numel()),
                "post_outlier_filter_token_count": int(inlier_mask.sum().item()),
                "outlier_tokens_excluded": int((~inlier_mask).sum().item()),
                "outlier_norm_factor": OUTLIER_NORM_FACTOR,
                "median_token_norm": float(median_norm.item()),
            }

        reconstruction_hook_calls = 0

        def reconstruction_hook(_module: Any, _args: Any, output: Any) -> Any:
            nonlocal reconstruction_hook_calls
            reconstruction_hook_calls += 1
            original = hidden_from_output(output)
            with torch.inference_mode():
                replacement = original.clone()
                start = BOS_OFFSET if original.shape[1] > BOS_OFFSET else 0
                tail = original[:, start:, :].to(dtype=SAE_DTYPE)
                replacement[:, start:, :] = sae.decode(sae.encode(tail)).to(dtype=original.dtype)
            return replace_hidden(output, replacement)

        reconstruction_handle = model.model.layers[layer].register_forward_hook(reconstruction_hook)
        with torch.inference_mode():
            reconstructed_output = model(**inputs, use_cache=False)
            reconstructed_loss = causal_lm_loss(reconstructed_output.logits, inputs["input_ids"])
            reconstructed_generation = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=4,
                pad_token_id=tokenizer.pad_token_id,
            )
        reconstruction_handle.remove()

        zero_hook_calls = 0

        def zero_hook(_module: Any, _args: Any, output: Any) -> Any:
            nonlocal zero_hook_calls
            zero_hook_calls += 1
            original = hidden_from_output(output)
            replacement = original.clone()
            start = BOS_OFFSET if original.shape[1] > BOS_OFFSET else 0
            replacement[:, start:, :] = 0
            return replace_hidden(output, replacement)

        zero_handle = model.model.layers[layer].register_forward_hook(zero_hook)
        with torch.inference_mode():
            zero_output = model(**inputs, use_cache=False)
            zero_loss = causal_lm_loss(zero_output.logits, inputs["input_ids"])
        zero_handle.remove()

        denominator_loss = zero_loss - baseline_loss
        fraction_loss_recovered = (
            (zero_loss - reconstructed_loss) / denominator_loss
            if denominator_loss > 0
            else None
        )
        baseline_tokens = int(baseline_generation.shape[-1] - inputs["input_ids"].shape[-1])
        reconstructed_tokens = int(reconstructed_generation.shape[-1] - inputs["input_ids"].shape[-1])
        all_finite = all(
            [
                reconstruction_metrics["input_finite"],
                reconstruction_metrics["encode_finite"],
                reconstruction_metrics["decode_finite"],
                finite(baseline.logits),
                finite(reconstructed_output.logits),
                finite(zero_output.logits),
                all(math.isfinite(x) for x in [baseline_loss, reconstructed_loss, zero_loss]),
            ]
        )
        reconstruction_gate = bool(
            all_finite
            and explained_variance > 0.0
            and fraction_loss_recovered is not None
            and fraction_loss_recovered > 0.0
            and reconstructed_loss < zero_loss
        )
        technical_gate = bool(
            reconstruction_gate
            and reconstruction_hook_calls > 0
            and zero_hook_calls > 0
            and baseline_tokens > 0
            and reconstructed_tokens > 0
        )
        final_memory = memory_snapshot()
        record = {
            **base_record,
            "status": "COMPATIBLE" if technical_gate else "INCOMPATIBLE_TECHNICAL",
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "transformers": transformers.__version__,
                "gpu": torch.cuda.get_device_name(0),
                "gpu_total_vram_bytes": int(torch.cuda.get_device_properties(0).total_memory),
            },
            "file_verification": {
                "model_config_revision_path_verified": True,
                "sae_observed_sha256": observed_hashes,
                "sae_expected_sha256": expected_hashes,
                "all_hashes_match": True,
                "torch_load_weights_only": True,
                "torch_load_mmap": True,
            },
            "model_config": {
                key: model_config.get(key)
                for key in ("architectures", "model_type", "hidden_size", "num_hidden_layers", "vocab_size", "torch_dtype")
            },
            "sae_config": sae_config,
            "upstream_eval": upstream_eval,
            "state_keys": state_keys,
            "model_loading": {
                "seconds": model_load_seconds,
                "dtype": str(next(model.parameters()).dtype),
                "parameter_devices": parameter_devices,
                "device_map": {str(k): str(v) for k, v in (device_map or {}).items()},
                "low_cpu_mem_usage": True,
            },
            "sae_loading": {
                "seconds": sae_load_seconds,
                "dtype": str(next(sae.parameters()).dtype),
                "devices": sae_devices,
                "d_in": EXPECTED_D_IN,
                "d_sae": EXPECTED_D_SAE,
                "k": int(sae.k.item()),
                "threshold": float(sae.threshold.item()),
            },
            "hook": {
                "module": f"model.model.layers[{layer}]",
                "semantic": "decoder_layer_output_residual",
                "upstream_remove_bos": True,
                "bos_offset": BOS_OFFSET,
                "incremental_generation_rule": "Preserve first eight only when local sequence length exceeds eight; reconstruct all single-token cached steps.",
                "captured_type": type(hidden).__name__,
                "captured_shape": list(hidden.shape),
                "reconstruction_hook_calls": reconstruction_hook_calls,
                "zero_hook_calls": zero_hook_calls,
                "reinserted_dtype": str(hidden.dtype),
            },
            "reconstruction": {
                **reconstruction_metrics,
                "baseline_cross_entropy": baseline_loss,
                "reconstructed_cross_entropy": reconstructed_loss,
                "zero_ablation_cross_entropy": zero_loss,
                "fraction_loss_recovered": fraction_loss_recovered,
                "positive_explained_variance": explained_variance > 0.0,
                "reconstruction_better_than_zero": reconstructed_loss < zero_loss,
                "gate_status": "PASS" if reconstruction_gate else "FAIL",
            },
            "generation": {
                "input_tokens": int(inputs["input_ids"].shape[-1]),
                "baseline_generated_tokens": baseline_tokens,
                "reconstructed_generated_tokens": reconstructed_tokens,
                "baseline_generated_token_sha256": hashlib.sha256(baseline_generation[:, inputs["input_ids"].shape[-1]:].cpu().numpy().tobytes()).hexdigest(),
                "reconstructed_generated_token_sha256": hashlib.sha256(reconstructed_generation[:, inputs["input_ids"].shape[-1]:].cpu().numpy().tobytes()).hexdigest(),
            },
            "memory": {
                "model_only": model_memory,
                "after_sae_load": sae_memory,
                "final_and_peak": final_memory,
            },
            "elapsed_seconds": time.perf_counter() - started,
            "gate_criteria_frozen_before_execution": {
                "all_expected_hashes_match": True,
                "model_and_sae_entirely_cuda": True,
                "no_quantization_or_offload": True,
                "dimensions_match": True,
                "all_tested_tensors_and_losses_finite": True,
                "local_explained_variance_gt_zero": True,
                "local_fraction_loss_recovered_gt_zero": True,
                "reconstruction_loss_lt_zero_ablation_loss": True,
                "hook_calls_gt_zero": True,
                "deterministic_continuations_nonempty": True,
                "upstream_bos_offset_and_outlier_metric_filter_reproduced": True,
            },
            "blocker": None if technical_gate else "ONE_OR_MORE_FROZEN_TECHNICAL_OR_RECONSTRUCTION_CRITERIA_FAILED",
        }
        atomic_json(output_path, record)
        print(json.dumps({"status": record["status"], "layer": layer, "peak_vram_bytes": final_memory["peak_allocated_vram_bytes"], "reconstruction": record["reconstruction"]}, indent=2))
        return 0 if technical_gate else 4
    except torch.cuda.OutOfMemoryError as exc:
        base_record.update(
            {
                "status": "BLOCKED_A5000_FIT",
                "elapsed_seconds": time.perf_counter() - started,
                "blocker": f"CUDA_OUT_OF_MEMORY: {type(exc).__name__}",
                "traceback_recorded": True,
                "traceback": traceback.format_exc(),
            }
        )
        atomic_json(output_path, base_record)
        print(json.dumps(base_record, indent=2))
        return 3
    except Exception as exc:
        base_record.update(
            {
                "status": "FAIL_TECHNICAL_COMPATIBILITY",
                "elapsed_seconds": time.perf_counter() - started,
                "blocker": f"{type(exc).__name__}: {exc}",
                "traceback_recorded": True,
                "traceback": traceback.format_exc(),
            }
        )
        atomic_json(output_path, base_record)
        print(json.dumps(base_record, indent=2))
        return 2
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True, choices=LAYERS)
    parser.add_argument("--output-dir", type=Path, default=Path("revision/model3/phase22/outputs"))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if args.preflight_only:
        return preflight(args.layer, output_dir)
    return run(args.layer, output_dir)


if __name__ == "__main__":
    sys.exit(main())
