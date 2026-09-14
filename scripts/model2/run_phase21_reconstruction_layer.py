#!/usr/bin/env python
"""Run one frozen SAE reconstruction audit layer in a fresh GPU process."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from sae_lens import SAE
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

MODEL_ID = "google/gemma-2-9b-it"
MODEL_REVISION = "11c9b309abf73637e4b6f9a3fa1e92e615547819"
SAE_REPOSITORY = "google/gemma-scope-9b-it-res"
SAE_REVISION = "e86af97a5b6fbbccca28ab654f2fda1b0768f770"
LABELS = {9: 47, 20: 47, 31: 43}
PARAM_HASHES = {
    9: "60dd98386fa1361f47f9ba25da29dca575c213dc1392f0c57ccefc738b254da5",
    20: "63337f10014d4c9096c51ecc372d1b61a8ef2642be2ca01a0d04ccd3dc0e6bf2",
    31: "f5b2265ffe36c4e55f6268224b9d6f47742b6ae0e5a393844c1402bf43e0b3b4",
}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_bytes(t: torch.Tensor) -> bytes:
    return t.detach().to(dtype=torch.float32, device="cpu").contiguous().numpy().tobytes()


class Accumulator:
    def __init__(self, d_in: int):
        self.tokens = 0
        self.elements = 0
        self.sse = 0.0
        self.input_sq = 0.0
        self.token_centered_ss = 0.0
        self.sum_features = torch.zeros(d_in, dtype=torch.float64)
        self.sum_sq_features = torch.zeros(d_in, dtype=torch.float64)
        self.active = 0

    def add(self, x: torch.Tensor, r: torch.Tensor, f: torch.Tensor, mask: torch.Tensor) -> None:
        xx = x[mask].detach().to(dtype=torch.float64, device="cpu")
        rr = r[mask].detach().to(dtype=torch.float64, device="cpu")
        ff = f[mask].detach().to(device="cpu")
        if xx.numel() == 0:
            return
        e = rr - xx
        self.tokens += int(xx.shape[0])
        self.elements += int(xx.numel())
        self.sse += float(e.square().sum())
        self.input_sq += float(xx.square().sum())
        self.token_centered_ss += float((xx - xx.mean(dim=-1, keepdim=True)).square().sum())
        self.sum_features += xx.sum(dim=0)
        self.sum_sq_features += xx.square().sum(dim=0)
        self.active += int((ff != 0).sum())

    def metrics(self) -> dict[str, float | int]:
        featurewise_ss = float((self.sum_sq_features - self.sum_features.square() / self.tokens).sum())
        return {
            "token_count": self.tokens,
            "element_count": self.elements,
            "raw_mse": self.sse / self.elements,
            "relative_l2_error": math.sqrt(self.sse / self.input_sq),
            "existing_normalized_mse": self.sse / self.token_centered_ss,
            "existing_explained_variance": 1.0 - self.sse / self.token_centered_ss,
            "audited_featurewise_normalized_mse": self.sse / featurewise_ss,
            "audited_featurewise_explained_variance": 1.0 - self.sse / featurewise_ss,
            "audited_mean_l0": self.active / self.tokens,
        }


def main(layer: int) -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    out_dir = Path("revision/model2/phase21/outputs")
    subset_path = out_dir / "phase21_reconstruction_diagnostic_subset.json"
    subset = json.loads(subset_path.read_text(encoding="utf-8"))
    source_path = Path(subset["source_path"])
    if sha256_file(source_path) != subset["source_sha256"]:
        raise RuntimeError("development source hash mismatch")
    rows = json.loads(source_path.read_text(encoding="utf-8"))
    chosen = []
    for rec in subset["records"]:
        row = rows[rec["source_index"]]
        if row["prompt_id"] != rec["prompt_id"] or hashlib.sha256(row["test_case_prompt"].encode()).hexdigest() != rec["prompt_sha256"]:
            raise RuntimeError("frozen diagnostic record mismatch")
        chosen.append((rec, row["test_case_prompt"]))

    label = LABELS[layer]
    sae_id = f"layer_{layer}/width_16k/average_l0_{label}"
    params = hf_hub_download(SAE_REPOSITORY, f"{sae_id}/params.npz", revision=SAE_REVISION, local_files_only=True)
    if sha256_file(params) != PARAM_HASHES[layer]:
        raise RuntimeError("frozen SAE hash mismatch")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    load_start = time.perf_counter()
    model = HookedTransformer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, tokenizer=tokenizer, device="cuda:0",
        n_devices=1, dtype=torch.bfloat16, fold_ln=True,
        center_writing_weights=False, center_unembed=False, move_to_device=True,
        local_files_only=True,
    ).eval()
    sae = SAE.from_pretrained("gemma-scope-9b-it-res", sae_id, device="cuda:0", dtype="float32").eval()
    load_seconds = time.perf_counter() - load_start
    if sorted({str(p.device) for p in model.parameters()}) != ["cuda:0"]:
        raise RuntimeError("unauthorized model offload")
    hook_name = f"blocks.{layer}.hook_resid_post"
    all_acc = Accumulator(3584)
    content_acc = Accumulator(3584)
    input_hash = hashlib.sha256()
    feature_hash = hashlib.sha256()
    reconstruction_hash = hashlib.sha256()
    per_prompt = []
    all_special = set(tokenizer.all_special_ids)
    replay_max = 0.0
    nan_inf = False
    observed_shapes = set()

    with torch.inference_mode():
        for rec, prompt_text in chosen:
            chat = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}], tokenize=False,
                add_generation_prompt=True,
            )
            tokens = tokenizer(chat, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda:0")
            _, cache = model.run_with_cache(
                tokens, names_filter=lambda name: name == hook_name,
                return_type=None, stop_at_layer=layer + 1,
            )
            source = cache[hook_name].detach().to(dtype=torch.float32)
            features = sae.encode(source)
            reconstruction = sae.decode(features)
            replay = sae.decode(sae.encode(source))
            replay_max = max(replay_max, float((reconstruction - replay).abs().max()))
            finite = bool(torch.isfinite(source).all() and torch.isfinite(features).all() and torch.isfinite(reconstruction).all())
            nan_inf = nan_inf or not finite
            flat_tokens = tokens.reshape(-1)
            special_mask = torch.tensor([int(v) in all_special for v in flat_tokens.tolist()], device="cuda:0", dtype=torch.bool)
            all_mask = torch.ones(flat_tokens.shape[0], device="cuda:0", dtype=torch.bool)
            content_mask = ~special_mask
            x = source.reshape(-1, 3584)
            r = reconstruction.reshape(-1, 3584)
            f = features.reshape(-1, 16384)
            p_all = Accumulator(3584)
            p_all.add(x, r, f, all_mask)
            p_content = Accumulator(3584)
            p_content.add(x, r, f, content_mask)
            all_acc.add(x, r, f, all_mask)
            content_acc.add(x, r, f, content_mask)
            input_hash.update(tensor_bytes(source))
            feature_hash.update(tensor_bytes(features))
            reconstruction_hash.update(tensor_bytes(reconstruction))
            observed_shapes.add((int(source.shape[0]), int(source.shape[2])))
            per_prompt.append({
                "source_index": rec["source_index"], "prompt_id": rec["prompt_id"],
                "sequence_tokens": int(tokens.shape[-1]),
                "special_token_positions": [i for i, v in enumerate(special_mask.tolist()) if v],
                "all_tokens": p_all.metrics(), "content_tokens": p_content.metrics(),
                "finite": finite,
            })
            del tokens, cache, source, features, reconstruction, replay, x, r, f
            torch.cuda.empty_cache()

    result = {
        "schema_version": "phase21_reconstruction_layer_audit_v1",
        "status": "PASS" if not nan_inf and replay_max == 0.0 else "FAIL",
        "layer": layer, "sae_id": sae_id, "params_sha256": PARAM_HASHES[layer],
        "hook_representation": {
            "framework": "TransformerLens 2.17.0",
            "hook": hook_name, "residual_location": "resid_post",
            "batch_dimension": 1, "sequence_dimension": "variable, unpadded per prompt",
            "hidden_dimension": 3584, "captured_dtype": "torch.bfloat16",
            "sae_input_dtype_after_official_preprocessing": "torch.float32",
            "device": "cuda:0", "attention_mask": "implicit all-attended causal sequence; batch size one and no padding",
            "padding_side": tokenizer.padding_side,
            "model_transform": "TransformerLens fold_ln=true; center_writing_weights=false; center_unembed=false",
            "observed_batch_hidden_shapes": [list(x) for x in sorted(observed_shapes)],
        },
        "token_populations": {
            "all_tokens": "all unpadded chat-template tokens, including special tokens",
            "content_tokens": "all unpadded tokens excluding tokenizer.all_special_ids",
        },
        "all_tokens": all_acc.metrics(),
        "content_tokens": content_acc.metrics(),
        "per_prompt": per_prompt,
        "tensor_hashes_float32_source_order": {
            "input_sha256": input_hash.hexdigest(),
            "features_sha256": feature_hash.hexdigest(),
            "reconstruction_sha256": reconstruction_hash.hexdigest(),
        },
        "deterministic_replay_max_abs_difference": replay_max,
        "nan_or_inf_found": nan_inf,
        "subset_manifest_sha256": sha256_file(subset_path),
        "model_load_plus_sae_seconds": load_seconds,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(0)),
        "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(0)),
        "quantization_used": False, "cpu_offload_used": False,
        "security_feature_discovery_run": False, "scanner_run": False,
        "causal_security_validation_run": False, "heldout_used": False,
        "model1_modified": False,
    }
    path = out_dir / f"phase21_reconstruction_layer_{layer}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    print(json.dumps({"layer": layer, "status": result["status"], "all_tokens": result["all_tokens"], "peak_vram_bytes": result["peak_vram_bytes"]}, indent=2))
    del sae, model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, choices=[9, 20, 31], required=True)
    main(parser.parse_args().layer)
