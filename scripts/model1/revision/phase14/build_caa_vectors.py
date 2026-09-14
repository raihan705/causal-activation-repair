#!/usr/bin/env python3
"""Build or preflight frozen train-only CAA-CWE dense vectors.

Vector construction is not executed during the Phase 14 implementation freeze.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from phase14_common import (
    ACTIVE_CAA_CWES, CAA_LAYERS, EXPECTED_SOURCE_RECOVERY_SHA256,
    MODEL_CACHE_SNAPSHOT, MODEL_ID, OUTPUTS, ROOT, SOURCE_RECOVERY, atomic_json,
    read_json, relative, require, sha256_path, validate_preparation,
)


SUPPORT_CSV = ROOT / "revision/model1/phase1/caa_pair_support.csv"
SUPPORT_RULE = ROOT / "revision/model1/phase1/caa_support_rule.json"
TRAIN_PAIRS = ROOT / "data/processed/train_pairs.csv"
DENSE_REVISION = OUTPUTS / "dense_hook_readiness_revision.json"
EXPECTED_HASHES = {
    "support_csv": "30ba7ec95a61c3a6cceac7fbb6179a3c626998bfadc8e0d935dd31c164264343",
    "support_rule": "2c695f632e5520106fac6d3cde476e4090201f19c01f21c67c72f1a5072ce71f",
    "train_pairs": "599698e22ffc5eba626e8c5f0b9fccfdb9ae7edb9404669b648d809e85046069",
}
EXPECTED_SUPPORT = {
    "CWE-120": ("SUPPORTED", "PRIMARY_RAW", 13),
    "CWE-327": ("SUPPORTED", "PRIMARY_MIXED_RAW_AUGMENTED", 17),
    "CWE-89": ("SUPPORTED", "PRIMARY_RAW", 13),
    "CWE-338": ("EXPLORATORY_SUPPORT", "EXPLORATORY_SMALL_N_MIXED", 4),
}
DEFAULT_MANIFEST = OUTPUTS / "caa_vector_manifest.json"
DEFAULT_VECTOR_DIR = OUTPUTS / "caa_vectors"
RUNNER = Path(__file__).resolve()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def eligible_pairs() -> tuple[dict[str, list[dict[str, str]]], dict[str, dict[str, str]]]:
    support_rows = {row["cwe_id"]: row for row in load_csv(SUPPORT_CSV)}
    train = load_csv(TRAIN_PAIRS)
    selected: dict[str, list[dict[str, str]]] = {}
    for cwe in ACTIVE_CAA_CWES:
        status, tier, expected_count = EXPECTED_SUPPORT[cwe]
        support = support_rows[cwe]
        require(support["support_status"] == status and support["evidence_tier"] == tier,
                f"frozen support status changed for {cwe}")
        require(support["scanner_supported"] == "true" and support["source_verified"] == "true",
                f"scanner/source support failed for {cwe}")
        rows = [row for row in train if row["cwe_id"] == cwe and row["vulnerable_code"].strip()
                and row["fixed_code"].strip() and row["vulnerable_code"] != row["fixed_code"]]
        require(len(rows) == expected_count == int(support["aligned_training_pairs"]),
                f"aligned train-pair count changed for {cwe}")
        require(len({row["pair_id"] for row in rows}) == len(rows), f"duplicate pair ID for {cwe}")
        selected[cwe] = rows
    return selected, support_rows


def preflight() -> dict[str, Any]:
    validate_preparation()
    for label, path in (("support_csv", SUPPORT_CSV), ("support_rule", SUPPORT_RULE),
                        ("train_pairs", TRAIN_PAIRS)):
        require(path.is_file() and sha256_path(path) == EXPECTED_HASHES[label], f"{label} hash mismatch")
    dense = read_json(DENSE_REVISION)
    require(dense.get("verification_status") == "PASS" and dense.get("layers_verified") == list(CAA_LAYERS),
            "revised dense-hook readiness did not pass all layers")
    rule = read_json(SUPPORT_RULE)
    require(tuple(rule["candidate_routes"]) == ACTIVE_CAA_CWES, "CAA candidate route order changed")
    require(rule["alignment_rule"]["minimum_aligned_pairs"] == 1 and rule["alignment_rule"]["tuned"] is False,
            "CAA constructibility rule changed")
    selected, support = eligible_pairs()
    return {
        "schema_version": "phase14_caa_vector_builder_preflight_v1", "status": "PASS",
        "construction_executed": False, "held_out_accessed": False, "validation_or_test_pairs_accessed": False,
        "routes": {cwe: {"pair_count": len(selected[cwe]),
                         "pair_ids": [row["pair_id"] for row in selected[cwe]],
                         "support_status": support[cwe]["support_status"],
                         "evidence_tier": support[cwe]["evidence_tier"]} for cwe in ACTIVE_CAA_CWES},
        "layers": list(CAA_LAYERS), "activation_position": "final non-padding non-special code token",
        "raw_direction": "mean(fixed activations) - mean(vulnerable activations)",
        "normalization": "per-layer cross-route mean-L2-norm equalization; no unit-L2 normalization",
        "source_hashes": {relative(SUPPORT_CSV): sha256_path(SUPPORT_CSV),
                          relative(SUPPORT_RULE): sha256_path(SUPPORT_RULE),
                          relative(TRAIN_PAIRS): sha256_path(TRAIN_PAIRS),
                          relative(DENSE_REVISION): sha256_path(DENSE_REVISION),
                          relative(SOURCE_RECOVERY): EXPECTED_SOURCE_RECOVERY_SHA256},
        "runner": {"path": relative(RUNNER), "sha256": sha256_path(RUNNER)},
    }


def tensor_hash(tensor: Any) -> str:
    import hashlib
    raw = tensor.detach().float().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def run_construction(args: argparse.Namespace, pf: dict[str, Any]) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    selected, support_rows = eligible_pairs()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_CACHE_SNAPSHOT, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_CACHE_SNAPSHOT, torch_dtype=torch.float16,
        device_map="auto", local_files_only=True,
    ).eval()
    input_device = model.get_input_embeddings().weight.device

    def activation(code: str, layer: int) -> Any:
        captured: list[Any] = []
        def hook(module: Any, inputs: tuple[Any, ...], output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            captured.append(hidden.detach())
        encoded = tokenizer(code, return_tensors="pt", truncation=True, max_length=1024)
        encoded = {key: value.to(input_device) for key, value in encoded.items()}
        ids = encoded["input_ids"][0]
        valid = [index for index, token in enumerate(ids.tolist())
                 if encoded["attention_mask"][0, index].item() == 1 and token not in tokenizer.all_special_ids]
        require(bool(valid), "code has no non-padding non-special token")
        handle = model.model.layers[layer].register_forward_hook(hook)
        try:
            with torch.inference_mode():
                model(**encoded, use_cache=False)
        finally:
            handle.remove()
        require(len(captured) == 1, "dense activation hook call count mismatch")
        return captured[0][0, valid[-1], :].float().cpu()

    raw_by_layer: dict[int, dict[str, Any]] = {layer: {} for layer in CAA_LAYERS}
    means: dict[tuple[int, str], tuple[Any, Any]] = {}
    for layer in CAA_LAYERS:
        for cwe in ACTIVE_CAA_CWES:
            vuln = torch.stack([activation(row["vulnerable_code"], layer) for row in selected[cwe]])
            fixed = torch.stack([activation(row["fixed_code"], layer) for row in selected[cwe]])
            vuln_mean, fixed_mean = vuln.mean(0), fixed.mean(0)
            raw_by_layer[layer][cwe] = fixed_mean - vuln_mean
            means[(layer, cwe)] = (vuln_mean, fixed_mean)

    args.vector_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for layer in CAA_LAYERS:
        mean_norm = torch.stack([torch.linalg.vector_norm(raw_by_layer[layer][cwe]) for cwe in ACTIVE_CAA_CWES]).mean()
        for cwe in ACTIVE_CAA_CWES:
            raw = raw_by_layer[layer][cwe]
            norm = torch.linalg.vector_norm(raw)
            require(float(norm.item()) > 0.0, f"zero CAA direction for {cwe} layer {layer}")
            prepared = raw * (mean_norm / norm)
            vuln_mean, fixed_mean = means[(layer, cwe)]
            artifact = args.vector_dir / f"caa_{cwe.lower().replace('-', '')}_layer{layer}.pt"
            torch.save({"vulnerable_mean": vuln_mean, "fixed_mean": fixed_mean,
                        "raw_direction": raw, "prepared_vector": prepared}, artifact)
            support = support_rows[cwe]
            entries.append({
                "cwe_id": cwe, "layer": layer, "pair_ids": [row["pair_id"] for row in selected[cwe]],
                "source_split": "train", "support_status": support["support_status"],
                "evidence_tier": support["evidence_tier"],
                "activation_position": "final non-padding non-special code token",
                "raw_direction": "mean(fixed)-mean(vulnerable)",
                "normalization": "scaled to layer mean raw-vector L2 norm across supported routes",
                "raw_vector_sha256": tensor_hash(raw), "prepared_vector_sha256": tensor_hash(prepared),
                "artifact_path": relative(artifact), "artifact_sha256": sha256_path(artifact),
            })
    manifest = {"schema_version": "phase14_caa_vector_manifest_v1", "status": "CONSTRUCTED",
                "source_preflight": pf, "model_id": MODEL_ID, "model_cache_snapshot": MODEL_CACHE_SNAPSHOT,
                "entries": entries}
    atomic_json(args.manifest, manifest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--vector-dir", type=Path, default=DEFAULT_VECTOR_DIR)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-output", type=Path)
    args = parser.parse_args()
    args.manifest, args.vector_dir = args.manifest.resolve(), args.vector_dir.resolve()
    if args.preflight_output:
        args.preflight_output = args.preflight_output.resolve()
    pf = preflight()
    if args.preflight_output:
        atomic_json(args.preflight_output, pf)
    print(json.dumps(pf, indent=2, sort_keys=True))
    if args.preflight_only:
        return 0
    require(not args.manifest.exists(), "CAA vector manifest exists; immutable review required")
    run_construction(args, pf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
