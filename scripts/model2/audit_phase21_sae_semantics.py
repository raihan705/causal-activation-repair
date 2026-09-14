#!/usr/bin/env python
"""Audit the frozen Gemma Scope loader semantics without model execution."""

from __future__ import annotations

import gc
import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from sae_lens import SAE
from sae_lens.saes.jumprelu_sae import JumpReLUSAE, JumpReLUSAEConfig

MODEL_ID = "google/gemma-2-9b-it"
MODEL_REVISION = "11c9b309abf73637e4b6f9a3fa1e92e615547819"
SAE_REPOSITORY = "google/gemma-scope-9b-it-res"
SAE_REVISION = "e86af97a5b6fbbccca28ab654f2fda1b0768f770"
SPECS = ((9, 47), (20, 47), (31, 43))
EXPECTED_HASHES = {
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


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def manual_sae(params_path: str) -> JumpReLUSAE:
    sae = JumpReLUSAE(JumpReLUSAEConfig(
        d_in=3584, d_sae=16384, dtype="float32", device="cpu",
        apply_b_dec_to_input=False, normalize_activations="none",
    )).eval()
    state = {}
    with np.load(params_path) as data:
        for key in data.files:
            mapped = "W_" + key[2:] if key.startswith("w_") else key
            state[mapped] = torch.from_numpy(np.asarray(data[key]))
    missing, unexpected = sae.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"state mismatch: missing={missing}, unexpected={unexpected}")
    return sae


def main() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    out = Path("revision/model2/phase21/outputs")
    comparisons = []
    for layer, label_l0 in SPECS:
        sae_id = f"layer_{layer}/width_16k/average_l0_{label_l0}"
        pinned = hf_hub_download(
            SAE_REPOSITORY, f"{sae_id}/params.npz", revision=SAE_REVISION,
            local_files_only=True,
        )
        pinned_hash = sha256_file(pinned)
        if pinned_hash != EXPECTED_HASHES[layer]:
            raise RuntimeError(f"frozen parameter hash mismatch at L{layer}")
        official = SAE.from_pretrained(
            "gemma-scope-9b-it-res", sae_id, device="cpu", dtype="float32"
        ).eval()
        manual = manual_sae(pinned)
        official_state = official.state_dict()
        manual_state = manual.state_dict()
        state_max = max(float((official_state[k] - manual_state[k]).abs().max()) for k in official_state)
        torch.manual_seed(42 + layer)
        x = torch.randn(1, 3, 3584, dtype=torch.float32)
        official_features = official.encode(x)
        manual_features = manual.encode(x)
        official_reconstruction = official.decode(official_features)
        manual_reconstruction = manual.decode(manual_features)
        cfg = official.cfg.to_dict() if hasattr(official.cfg, "to_dict") else dict(official.cfg.__dict__)
        comparisons.append({
            "layer": layer,
            "sae_id": sae_id,
            "params_sha256": pinned_hash,
            "official_class": type(official).__name__,
            "official_config": cfg,
            "preflight_config": {
                "d_in": 3584, "d_sae": 16384, "dtype": "float32",
                "apply_b_dec_to_input": False,
                "normalize_activations": "none",
                "reshape_activations": "none (default)",
                "architecture": "jumprelu",
            },
            "state_dict_max_abs_difference": state_max,
            "encode_max_abs_difference": float((official_features - manual_features).abs().max()),
            "encode_mean_abs_difference": float((official_features - manual_features).abs().mean()),
            "decode_max_abs_difference": float((official_reconstruction - manual_reconstruction).abs().max()),
            "decode_mean_abs_difference": float((official_reconstruction - manual_reconstruction).abs().mean()),
            "equivalent": state_max == 0.0 and torch.equal(official_features, manual_features) and torch.equal(official_reconstruction, manual_reconstruction),
        })
        del official, manual, official_state, manual_state, official_features, manual_features
        del official_reconstruction, manual_reconstruction
        gc.collect()

    module_path = Path(inspect.getsourcefile(JumpReLUSAE) or "")
    execution = {
        "schema_version": "phase21_sae_execution_audit_v1",
        "status": "PASS" if all(x["equivalent"] for x in comparisons) else "IMPLEMENTATION_MISMATCH_FOUND",
        "frozen_model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "frozen_sae_repository": {"id": SAE_REPOSITORY, "revision": SAE_REVISION},
        "official_execution_semantics": {
            "preprocessing": "cast input to float32; no reshape; no activation normalization; no input centering because apply_b_dec_to_input=false",
            "encoder": "hidden_pre = x @ W_enc + b_enc; base = ReLU(hidden_pre); features = base * (hidden_pre > threshold)",
            "threshold_comparison": "strict greater-than on pre-activation",
            "decoder": "reconstruction = features @ W_dec + b_dec; no activation denormalization; no output reshape",
            "scaling_factor": "absent for all three frozen parameter files",
        },
        "official_vs_preflight": comparisons,
        "implementation_mismatch_found": not all(x["equivalent"] for x in comparisons),
        "first_divergent_operation": None if all(x["equivalent"] for x in comparisons) else "See first nonzero tensor comparison",
        "source_files": {
            "preflight_runner": "revision/model2/phase21/scripts/run_phase21_compatibility.py",
            "preflight_runner_sha256": sha256_file("revision/model2/phase21/scripts/run_phase21_compatibility.py"),
            "sae_lens_jumprelu_module": str(module_path),
            "sae_lens_jumprelu_module_sha256": sha256_file(module_path),
        },
        "notes": [
            "SAE-Lens release loading was forced offline and its loaded state was compared byte-semantically against each exact revision-pinned params.npz.",
            "The official loader metadata identifies blocks.N.hook_resid_post, prepend_bos=true, context_size=1024, and training dataset monology/pile-uncopyrighted.",
        ],
    }
    atomic_json(out / "phase21_sae_execution_audit.json", execution)

    l0 = {
        "schema_version": "phase21_l0_definition_audit_v1",
        "status": "L0_COMPARISON_INVALID",
        "previous_observed_definition": "For the 21-token synthetic chat-template prompt, count (official JumpReLU feature activation != 0) per token after strict preactivation thresholding and ReLU, then mean across all sequence tokens including special tokens.",
        "previous_values": {"9": 363.4285888671875, "20": 282.6666564941406, "31": 369.0952453613281},
        "frozen_variant_labels": {"9": 47, "20": 47, "31": 43},
        "frozen_label_recoverable_meaning": "Average per-token number of active latents for the SAE training/evaluation activation distribution used to name each release variant.",
        "frozen_label_population_metadata": {"dataset_path": "monology/pile-uncopyrighted", "context_size": 1024, "prepend_bos": True},
        "threshold": "features are active iff ReLU(hidden_pre) != 0 after applying hidden_pre > learned threshold; equivalently hidden_pre > max(threshold, 0) for positive thresholds",
        "aggregation": "per-token active count, followed by a token-weighted mean",
        "directly_comparable": False,
        "reason": "The arithmetic active-count definition is aligned, but the published label is an aggregate over a different training/evaluation corpus and 1024-token context population. The earlier value used one 21-token chat-formatted synthetic prompt and included its special tokens; the repository does not provide an exact label-estimation token mask/record manifest needed for a like-for-like comparison.",
        "security_inference_allowed": False,
    }
    atomic_json(out / "phase21_l0_definition_audit.json", l0)

    metric = {
        "schema_version": "phase21_reconstruction_metric_audit_v1",
        "status": "DIAGNOSTIC_METRIC_MISMATCH_FOUND",
        "formulas": {
            "raw_mse": "sum((x_hat - x)^2) / (N_tokens * d_in)",
            "relative_l2": "sqrt(sum((x_hat - x)^2) / sum(x^2))",
            "existing_normalized_mse": "sum((x_hat - x)^2) / sum((x - mean_hidden_coordinates_per_token(x))^2)",
            "existing_explained_variance": "1 - existing_normalized_mse",
            "audited_featurewise_normalized_mse": "sum((x_hat - x)^2) / sum((x - mean_over_tokens_per_hidden_coordinate(x))^2)",
            "audited_featurewise_explained_variance": "1 - audited_featurewise_normalized_mse",
        },
        "arithmetic_bug_found": False,
        "semantic_labeling_problem_found": True,
        "explanation": "The prior implementation exactly computed its written expression, but centering each token across hidden coordinates is not feature-wise variance over observations. Its value is retained as existing diagnostic history; the audit reports both it and a clearly named feature-wise alternative from the same tensors.",
        "model1_metric_changed": False,
        "phase21_only": True,
    }
    atomic_json(out / "phase21_reconstruction_metric_audit.json", metric)
    print(json.dumps({"status": execution["status"], "layers": [x["layer"] for x in comparisons]}, indent=2))


if __name__ == "__main__":
    main()
