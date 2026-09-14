#!/usr/bin/env python
"""Encode one frozen Phase 22D residual layer with its pinned BatchTopK SAE."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch

from phase22d_sae import D_IN, D_SAE, EXPECTED_SAE_SHA256, LAYERS, SAE_REPOSITORY, SAE_REVISION, load_sae, sae_weight_path, sha256_file

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "revision/model3/phase22/outputs"
ACT = OUT / "activations"
PARTITION = OUT / "phase22d_feature_partition_manifest.json"
PROTOCOL = OUT / "phase22d_activation_ranking_protocol.json"
RAW_MANIFEST = OUT / "phase22d_raw_activation_manifest.json"
EXPECTED_COUNT = 144
RUNNER = Path(__file__).resolve()
HELPER = Path(__file__).with_name("phase22d_sae.py")


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def paths(layer: int) -> tuple[Path, Path, Path, Path]:
    return (
        ACT / f"phase22d_layer_{layer}_raw.pt",
        ACT / f"phase22d_layer_{layer}_latents.pt",
        ACT / f"phase22d_layer_{layer}_latent_checkpoint.pt",
        OUT / f"phase22d_layer_{layer}_latent_manifest.json",
    )


def validate(layer: int, resume: bool) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    require(layer in LAYERS, "unapproved layer")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    raw_manifest = json.loads(RAW_MANIFEST.read_text(encoding="utf-8"))
    raw_path, output, checkpoint, manifest = paths(layer)
    require(protocol["status"] == "FROZEN_BEFORE_ACTIVATION_EXTRACTION", "protocol not frozen")
    require(protocol["scripts"]["latent_encoding_sha256"] == sha256_file(RUNNER), "latent runner drift")
    require(protocol["scripts"]["sae_helper_sha256"] == sha256_file(HELPER), "SAE helper drift")
    require(raw_manifest["status"] == "COMPLETE" and raw_manifest["record_count"] == EXPECTED_COUNT, "raw extraction incomplete")
    require(raw_manifest["artifacts"][str(layer)]["sha256"] == sha256_file(raw_path), "raw artifact hash drift")
    require(sha256_file(sae_weight_path(layer)) == EXPECTED_SAE_SHA256[layer], "SAE weight drift")
    require(not output.exists() and not manifest.exists(), "immutable final latent output already exists")
    prior = None
    if checkpoint.exists():
        require(resume, "latent checkpoint exists; use --resume")
        prior = torch.load(checkpoint, map_location="cpu", weights_only=False)
        require(prior["protocol_sha256"] == sha256_file(PROTOCOL), "latent checkpoint protocol drift")
        require(prior["runner_sha256"] == sha256_file(RUNNER), "latent checkpoint runner drift")
        require(prior["helper_sha256"] == sha256_file(HELPER), "latent checkpoint helper drift")
        require(prior["layer"] == layer and tuple(prior["latents"].shape)[1:] == (D_SAE,), "latent checkpoint shape/layer mismatch")
    else:
        require(not resume, "--resume supplied without checkpoint")
    return protocol, raw_manifest, prior


def preflight(layer: int, resume: bool) -> dict[str, Any]:
    _protocol, _raw, prior = validate(layer, resume)
    return {
        "schema_version": "phase22d_latent_preflight_v1", "status": "PASS_APPROVED",
        "layer": layer, "record_count": EXPECTED_COUNT, "d_in": D_IN, "d_sae": D_SAE,
        "completed_checkpoint_records": int(prior["latents"].shape[0]) if prior else 0,
        "resume_required": prior is not None, "protocol_sha256": sha256_file(PROTOCOL),
        "runner_sha256": sha256_file(RUNNER), "helper_sha256": sha256_file(HELPER),
        "sae_weight_sha256": EXPECTED_SAE_SHA256[layer], "sae_loaded": False,
        "gpu_execution": False, "ranking_run": False, "scanner_run": False,
        "causal_intervention_run": False, "heldout_used": False,
    }


def run(layer: int, resume: bool) -> None:
    require(torch.cuda.is_available(), "CUDA unavailable")
    protocol, raw_manifest, prior = validate(layer, resume)
    raw_path, output, checkpoint, manifest = paths(layer)
    raw = torch.load(raw_path, map_location="cpu", weights_only=False)
    require(tuple(raw["activations"].shape) == (EXPECTED_COUNT, D_IN), "raw tensor shape mismatch")
    require([int(x["prompt_id"]) for x in raw["metadata"]] == protocol["ranking_prompt_union_ids_source_order"], "raw ID order mismatch")
    completed: list[torch.Tensor] = [] if prior is None else [x for x in prior["latents"]]
    prior_elapsed = float(prior.get("elapsed_seconds", 0.0)) if prior else 0.0
    torch.cuda.set_device("cuda:0")
    sae = load_sae(layer, "cuda:0")
    devices = sorted({str(x.device) for x in [*sae.parameters(), *sae.buffers()]})
    require(devices == ["cuda:0"], f"SAE device mismatch: {devices}")
    started = time.perf_counter()
    with torch.inference_mode():
        for index in range(len(completed), EXPECTED_COUNT):
            activation = raw["activations"][index:index + 1].to(device="cuda:0", dtype=torch.float32)
            latent = sae.encode(activation, use_threshold=True)[0].detach().cpu().contiguous()
            require(tuple(latent.shape) == (D_SAE,) and bool(torch.isfinite(latent).all()), f"invalid latent row {index}")
            completed.append(latent)
            if len(completed) % 10 == 0 or len(completed) == EXPECTED_COUNT:
                elapsed = prior_elapsed + time.perf_counter() - started
                atomic_torch(checkpoint, {
                    "schema_version": "phase22d_latent_checkpoint_v1", "layer": layer,
                    "protocol_sha256": sha256_file(PROTOCOL), "runner_sha256": sha256_file(RUNNER),
                    "helper_sha256": sha256_file(HELPER), "latents": torch.stack(completed),
                    "elapsed_seconds": elapsed,
                })
                print(f"PHASE22D LATENTS L{layer} {len(completed)}/{EXPECTED_COUNT} elapsed_min={elapsed/60:.1f}", flush=True)
    latents = torch.stack(completed)
    payload = {
        "schema_version": "phase22d_batchtopk_latents_v1", "layer": layer,
        "protocol_sha256": sha256_file(PROTOCOL), "partition_sha256": sha256_file(PARTITION),
        "raw_sha256": sha256_file(raw_path), "sae_repository": SAE_REPOSITORY,
        "sae_revision": SAE_REVISION, "sae_weight_sha256": EXPECTED_SAE_SHA256[layer],
        "encoding": "BatchTopKSAE.encode(use_threshold=True)", "latents": latents,
        "metadata": raw["metadata"],
    }
    atomic_torch(output, payload)
    active = (latents > 0).sum(dim=1)
    value = {
        "schema_version": "phase22d_latent_manifest_v1", "status": "COMPLETE",
        "layer": layer, "record_count": EXPECTED_COUNT, "shape": [EXPECTED_COUNT, D_SAE],
        "protocol_sha256": sha256_file(PROTOCOL), "partition_sha256": sha256_file(PARTITION),
        "raw_sha256": sha256_file(raw_path), "latent_path": str(output.relative_to(ROOT)).replace("\\", "/"),
        "latent_sha256": sha256_file(output), "runner_sha256": sha256_file(RUNNER),
        "helper_sha256": sha256_file(HELPER), "sae_weight_sha256": EXPECTED_SAE_SHA256[layer],
        "mean_active_features": float(active.float().mean().item()),
        "min_active_features": int(active.min().item()), "max_active_features": int(active.max().item()),
        "all_finite": bool(torch.isfinite(latents).all()), "parameter_devices": devices,
        "elapsed_seconds": prior_elapsed + time.perf_counter() - started,
        "platform": platform.platform(), "torch_version": torch.__version__,
        "batch_size": 1, "ranking_run": False, "scanner_run": False,
        "causal_intervention_run": False, "heldout_used": False,
    }
    atomic_json(manifest, value)
    print(json.dumps(value, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True, choices=LAYERS)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(preflight(args.layer, args.resume), indent=2))
    if not args.preflight_only:
        run(args.layer, args.resume)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"PHASE22D_LATENT_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
