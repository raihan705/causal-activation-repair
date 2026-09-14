#!/usr/bin/env python
"""Encode/decode one frozen Model2 source-matched activation layer."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from sae_lens import SAE

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "revision/model2/phase21/outputs"
ACT = OUT / "activations"
SOURCE_MANIFEST = OUT / "model2_activation_source_manifest.json"
SAE_REPOSITORY = "google/gemma-scope-9b-it-res"
SAE_REVISION = "e86af97a5b6fbbccca28ab654f2fda1b0768f770"
SPECS = {
    9: (47, "60dd98386fa1361f47f9ba25da29dca575c213dc1392f0c57ccefc738b254da5"),
    20: (47, "63337f10014d4c9096c51ecc372d1b61a8ef2642be2ca01a0d04ccd3dc0e6bf2"),
    31: (43, "f5b2265ffe36c4e55f6268224b9d6f47742b6ae0e5a393844c1402bf43e0b3b4"),
}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_torch(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0])
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def main(layer: int) -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen SAE reconstruction")
    label, expected_hash = SPECS[layer]
    sae_id = f"layer_{layer}/width_16k/average_l0_{label}"
    params = hf_hub_download(
        SAE_REPOSITORY, f"{sae_id}/params.npz", revision=SAE_REVISION,
        local_files_only=True,
    )
    if sha256_file(params) != expected_hash:
        raise RuntimeError(f"frozen SAE parameter hash mismatch at layer {layer}")

    source_manifest_hash = sha256_file(SOURCE_MANIFEST)
    raw_path = ACT / f"model2_layer_{layer}_raw.pt"
    raw_hash = sha256_file(raw_path)
    payload = torch.load(raw_path, map_location="cpu", weights_only=False)
    x_cpu = payload["activations"].detach().to(dtype=torch.float32, device="cpu").contiguous()
    metadata = payload["metadata"]
    if payload["layer"] != layer or payload["source_manifest_sha256"] != source_manifest_hash:
        raise RuntimeError("raw activation provenance mismatch")
    if tuple(x_cpu.shape) != (517, 3584) or len(metadata) != 517:
        raise RuntimeError(f"raw activation shape/metadata mismatch: {tuple(x_cpu.shape)}")

    torch.manual_seed(42 + layer)
    torch.cuda.manual_seed_all(42 + layer)
    sae = SAE.from_pretrained(
        "gemma-scope-9b-it-res", sae_id, device="cuda:0", dtype="float32"
    ).eval()
    if sae.cfg.d_in != 3584 or sae.cfg.d_sae != 16384:
        raise RuntimeError("frozen SAE dimensionality mismatch")

    latent_batches: list[torch.Tensor] = []
    reconstruction_batches: list[torch.Tensor] = []
    batch_size = 64
    with torch.inference_mode():
        for start in range(0, x_cpu.shape[0], batch_size):
            x = x_cpu[start:start + batch_size].to("cuda:0")
            features = sae.encode(x)
            reconstruction = sae.decode(features)
            if tuple(features.shape) != (x.shape[0], 16384):
                raise RuntimeError("latent shape mismatch")
            if tuple(reconstruction.shape) != tuple(x.shape):
                raise RuntimeError("reconstruction shape mismatch")
            latent_batches.append(features.detach().cpu())
            reconstruction_batches.append(reconstruction.detach().cpu())
    latents = torch.cat(latent_batches).to(dtype=torch.float32).contiguous()
    reconstruction = torch.cat(reconstruction_batches).to(dtype=torch.float32).contiguous()
    finite = bool(torch.isfinite(x_cpu).all() and torch.isfinite(latents).all() and torch.isfinite(reconstruction).all())
    if not finite:
        raise RuntimeError("NaN or Inf found in source, latent, or reconstruction tensors")

    latent_path = ACT / f"model2_layer_{layer}_latents.pt"
    reconstruction_path = ACT / f"model2_layer_{layer}_reconstruction.pt"
    row_path = ACT / f"model2_layer_{layer}_reconstruction_rows.csv"
    common = {
        "source_manifest_sha256": source_manifest_hash,
        "raw_activation_sha256": raw_hash,
        "sae_repository": SAE_REPOSITORY,
        "sae_revision": SAE_REVISION,
        "sae_id": sae_id,
        "params_sha256": expected_hash,
        "official_sae_lens_encode_decode": True,
    }
    atomic_torch(latent_path, {
        "schema_version": "phase21_model2_source_latents_v1", "layer": layer,
        **common, "latents": latents, "metadata": metadata,
    })
    atomic_torch(reconstruction_path, {
        "schema_version": "phase21_model2_source_reconstruction_v1", "layer": layer,
        **common, "reconstruction": reconstruction, "metadata": metadata,
    })

    error = reconstruction - x_cpu
    relative_l2 = torch.linalg.vector_norm(error, dim=1) / torch.linalg.vector_norm(x_cpu, dim=1)
    raw_mse = error.square().mean(dim=1)
    l0 = (latents != 0).sum(dim=1)
    active_fraction = l0.to(torch.float64) / 16384.0
    rows = []
    for index, meta in enumerate(metadata):
        rows.append({
            "source_order": meta["source_order"], "source": meta["source"],
            "source_split": meta["source_split"], "record_id": meta["record_id"],
            "pair_id": meta["pair_id"], "prompt_id": meta["prompt_id"],
            "cwe_id": meta["cwe_id"], "language": meta["language"],
            "layer": layer, "hook": f"blocks.{layer}.hook_resid_post",
            "relative_l2": float(relative_l2[index]),
            "raw_mse": float(raw_mse[index]), "l0": int(l0[index]),
            "active_fraction": float(active_fraction[index]), "finite": True,
        })
    atomic_csv(row_path, rows)
    result = {
        "schema_version": "phase21_model2_source_reconstruction_layer_v1",
        "status": "COMPLETE", "layer": layer, "hook": f"blocks.{layer}.hook_resid_post",
        "record_count": 517, "raw_shape": [517, 3584], "latent_shape": [517, 16384],
        "reconstruction_shape": [517, 3584], "sae_id": sae_id,
        "sae_repository": SAE_REPOSITORY, "sae_revision": SAE_REVISION,
        "params_sha256": expected_hash, "official_sae_lens_encode_decode": True,
        "sae_dtype": "float32", "nan_or_inf_found": False,
        "artifacts": {
            "raw": {"path": str(raw_path.relative_to(ROOT)).replace("\\", "/"), "sha256": raw_hash},
            "latents": {"path": str(latent_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256_file(latent_path)},
            "reconstruction": {"path": str(reconstruction_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256_file(reconstruction_path)},
            "row_metrics": {"path": str(row_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256_file(row_path)},
        },
        "diagnostic_all_rows": {
            "mean_relative_l2": float(relative_l2.mean()),
            "mean_raw_mse": float(raw_mse.mean()),
            "mean_l0": float(l0.to(torch.float64).mean()),
            "mean_active_fraction": float(active_fraction.mean()),
        },
        "feature_ranking_run": False, "causal_intervention_run": False,
        "scanner_run": False, "heldout_used": False,
    }
    result_path = OUT / f"model2_layer_{layer}_source_reconstruction.json"
    tmp = result_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, result_path)
    print(json.dumps(result, indent=2))
    del sae, x_cpu, latents, reconstruction
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", required=True, type=int, choices=sorted(SPECS))
    main(parser.parse_args().layer)
