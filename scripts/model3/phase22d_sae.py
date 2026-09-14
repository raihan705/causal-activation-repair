"""Pinned minimal BatchTopK SAE loader shared by Phase 22D scripts."""

from __future__ import annotations

import gc
import hashlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

SAE_REPOSITORY = "andyrdt/saes-qwen2.5-7b-instruct"
SAE_REVISION = "c37e53c4bb07127ad17ab88f28b93d4e87142e59"
LOADER_REVISION = "de9138b02fffdf9919c53cee828beb4e05049741"
LAYERS = (7, 15, 23)
D_IN = 3584
D_SAE = 131072
K = 32
EXPECTED_SAE_SHA256 = {
    7: "e9c88dc39dd89bc80bcea688c91739a37d1ec55bd7bf676f9cfb4bbf847fbd73",
    15: "1339c52258e95bc64535a90e45067187722e659e0ee03c976db813614b29ab5b",
    23: "a593d0da4a10cde4674b48749bf573b14e40119c1cff34cf2d499f334940ef4b",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sae_weight_path(layer: int) -> Path:
    if layer not in LAYERS:
        raise ValueError(f"unapproved SAE layer: {layer}")
    path = (
        Path.home()
        / ".cache/huggingface/hub/models--andyrdt--saes-qwen2.5-7b-instruct/snapshots"
        / SAE_REVISION
        / f"resid_post_layer_{layer}/trainer_0/ae.pt"
    )
    if not path.is_file():
        raise RuntimeError(f"pinned SAE weight missing: {path}")
    if sha256_file(path) != EXPECTED_SAE_SHA256[layer]:
        raise RuntimeError(f"pinned SAE weight hash mismatch at layer {layer}")
    return path


class BatchTopKSAE(nn.Module):
    """Inference-equivalent subset of the pinned upstream BatchTopKSAE."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("k", torch.tensor(K, dtype=torch.int))
        self.register_buffer("threshold", torch.tensor(-1.0, dtype=torch.float32))
        self.decoder = nn.Linear(D_SAE, D_IN, bias=False)
        self.encoder = nn.Linear(D_IN, D_SAE)
        self.b_dec = nn.Parameter(torch.zeros(D_IN))

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


def load_sae(layer: int, device: str = "cuda:0") -> BatchTopKSAE:
    path = sae_weight_path(layer)
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    required = {"encoder.weight", "encoder.bias", "decoder.weight", "b_dec", "k", "threshold"}
    if not isinstance(state, dict) or set(state) != required:
        raise RuntimeError(f"unexpected SAE state keys at layer {layer}")
    if tuple(state["encoder.weight"].shape) != (D_SAE, D_IN):
        raise RuntimeError(f"encoder dimension mismatch at layer {layer}")
    if tuple(state["decoder.weight"].shape) != (D_IN, D_SAE):
        raise RuntimeError(f"decoder dimension mismatch at layer {layer}")
    if int(state["k"].item()) != K:
        raise RuntimeError(f"k mismatch at layer {layer}")
    sae = BatchTopKSAE()
    sae.load_state_dict(state, strict=True)
    del state
    gc.collect()
    sae = sae.to(device=device, dtype=torch.float32).eval()
    for parameter in sae.parameters():
        parameter.requires_grad_(False)
    return sae
