#!/usr/bin/env python
"""Extract only missing Model2 B0 final-token activations and freeze ranking latents."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "revision/model2/phase21/outputs"
ACT = OUT / "activations"
PARTITION = OUT / "model2_feature_partition_manifest.json"
B0 = OUT / "model2_b0_dev_outputs.json"
SOURCE_MANIFEST = OUT / "model2_activation_source_manifest.json"
VIABILITY = OUT / "model2_layer_viability.json"
CHECKPOINT = ACT / "model2_feature_missing_raw_checkpoint.pt"
LATENT_MANIFEST = OUT / "model2_feature_latent_manifest.csv"
RUN_MANIFEST = OUT / "model2_feature_latent_extraction_run.json"
MODEL_ID = "google/gemma-2-9b-it"
MODEL_REVISION = "11c9b309abf73637e4b6f9a3fa1e92e615547819"
SAE_REPOSITORY = "google/gemma-scope-9b-it-res"
SAE_REVISION = "e86af97a5b6fbbccca28ab654f2fda1b0768f770"
LAYERS = (9, 20, 31)
PARTITION_HASH = "4aefb5aa35ee7845c556bac03a4e64a29d3b41422824a18767e36a5800e710c2"
SAE_SPECS = {
    9: (47, "60dd98386fa1361f47f9ba25da29dca575c213dc1392f0c57ccefc738b254da5"),
    20: (47, "63337f10014d4c9096c51ecc372d1b61a8ef2642be2ca01a0d04ccd3dc0e6bf2"),
    31: (43, "f5b2265ffe36c4e55f6268224b9d6f47742b6ae0e5a393844c1402bf43e0b3b4"),
}


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(tensor) -> str:
    import torch
    raw = tensor.detach().to(dtype=torch.float32, device="cpu").contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def atomic_json(path: Path, value: object) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_torch(path: Path, value: object) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def inputs():
    require(file_hash(PARTITION) == PARTITION_HASH, "frozen feature partition hash drift")
    partition = json.loads(PARTITION.read_text(encoding="utf-8"))
    b0_rows = json.loads(B0.read_text(encoding="utf-8"))
    b0 = {int(row["prompt_id"]): row for row in b0_rows}
    union = partition["ranking_prompt_union_ids_source_order"]
    reusable = set(partition["reusable_union_prompt_ids_source_order"])
    missing = partition["missing_union_prompt_ids_source_order"]
    require(len(union) == 302 and len(reusable) == 70 and len(missing) == 232, "frozen union/reuse/missing counts drift")
    require(set(union) == reusable | set(missing) and not reusable & set(missing), "union partition mismatch")
    for pid in union:
        require(pid in b0 and b0[pid]["generation_status"] == "SUCCESS", f"B0 source unavailable: {pid}")
        require(bool(b0[pid]["generated_code"].strip()), f"empty B0 generated code: {pid}")
    return partition, b0, union, reusable, missing


def verify_reusable(reusable: set[int]):
    import torch
    viability = json.loads(VIABILITY.read_text(encoding="utf-8"))
    layer_data = {}
    shared_meta = None
    for layer in LAYERS:
        raw_path = ACT / f"model2_layer_{layer}_raw.pt"
        latent_path = ACT / f"model2_layer_{layer}_latents.pt"
        artifacts = viability["artifacts"][str(layer)]
        require(file_hash(raw_path) == artifacts["raw"]["sha256"], f"accepted raw hash drift L{layer}")
        require(file_hash(latent_path) == artifacts["latents"]["sha256"], f"accepted latent hash drift L{layer}")
        raw = torch.load(raw_path, map_location="cpu", weights_only=False)
        latent = torch.load(latent_path, map_location="cpu", weights_only=False)
        require(tuple(raw["activations"].shape) == (517, 3584), f"accepted raw shape drift L{layer}")
        require(tuple(latent["latents"].shape) == (517, 16384), f"accepted latent shape drift L{layer}")
        require(bool(torch.isfinite(latent["latents"]).all()), f"accepted latent nonfinite L{layer}")
        if shared_meta is None:
            shared_meta = raw["metadata"]
        else:
            require(raw["metadata"] == shared_meta, "accepted metadata differs across layers")
        require(latent["metadata"] == shared_meta, f"raw/latent metadata mismatch L{layer}")
        layer_data[layer] = {"raw": raw, "latent": latent, "raw_path": raw_path, "latent_path": latent_path}
    mapping = {int(row["prompt_id"]): index for index, row in enumerate(shared_meta) if row["source"] == "MODEL2_B0"}
    require(reusable <= set(mapping), "a reusable prompt is missing from accepted source-matched artifacts")
    return layer_data, shared_meta, mapping


def preflight(resume: bool) -> dict:
    partition, _, _, reusable, missing = inputs()
    layer_data, metadata, mapping = verify_reusable(reusable)
    completed = 0
    if CHECKPOINT.exists():
        require(resume, "missing-only activation checkpoint exists; use --resume")
        import torch
        state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        require(state["partition_sha256"] == PARTITION_HASH, "checkpoint partition hash drift")
        completed = len(state["metadata"])
        require([int(row["prompt_id"]) for row in state["metadata"]] == missing[:completed], "checkpoint is not a frozen source-order prefix")
        require(all(tuple(state["activations"][layer].shape) == (completed, 3584) for layer in LAYERS), "checkpoint shape mismatch")
    else:
        require(not resume, "--resume supplied without checkpoint")
    del layer_data
    return {
        "status": "PASS_APPROVED", "partition_sha256": PARTITION_HASH,
        "union_records": 302, "reused_records": len(reusable), "missing_records": len(missing),
        "completed_checkpoint_records": completed, "model_loaded": False,
        "new_scan_run": False, "heldout_used": False, "causal_intervention_run": False,
        "feature_ranking_run": False, "alpha_selected": False,
    }


def run(resume: bool) -> None:
    import torch
    from huggingface_hub import hf_hub_download
    from sae_lens import SAE
    from transformer_lens import HookedTransformer
    from transformers import AutoTokenizer

    require(torch.cuda.is_available(), "CUDA unavailable")
    partition, b0, union, reusable, missing = inputs()
    accepted, accepted_meta, accepted_map = verify_reusable(reusable)
    new_metadata = []
    layer_lists = {layer: [] for layer in LAYERS}
    prior_elapsed = 0.0
    if CHECKPOINT.exists():
        state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        new_metadata = state["metadata"]
        layer_lists = {layer: [row for row in state["activations"][layer]] for layer in LAYERS}
        prior_elapsed = float(state.get("elapsed_seconds", 0.0))
    os.environ["HF_HUB_OFFLINE"] = "1"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    model = HookedTransformer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, tokenizer=tokenizer, device="cuda:0", n_devices=1,
        dtype=torch.bfloat16, fold_ln=True, center_writing_weights=False,
        center_unembed=False, move_to_device=True, local_files_only=True,
    ).eval()
    require(sorted({str(parameter.device) for parameter in model.parameters()}) == ["cuda:0"], "unauthorized model offload")
    hooks = {layer: f"blocks.{layer}.hook_resid_post" for layer in LAYERS}
    special_ids = set(tokenizer.all_special_ids)
    started = time.perf_counter()
    atomic_json(RUN_MANIFEST, {
        "schema_version": "phase21_model2_feature_latent_extraction_run_v1", "status": "IN_PROGRESS",
        "partition_sha256": PARTITION_HASH, "union_records": 302, "reused_records": 70,
        "missing_records": 232, "completed_missing_records": len(new_metadata),
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "layers": list(LAYERS),
        "scanner_run": False, "heldout_used": False, "causal_intervention_run": False,
    })
    with torch.inference_mode():
        for position in range(len(new_metadata), len(missing)):
            pid = missing[position]
            row = b0[pid]
            text = row["generated_code"]
            encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=512)
            tokens = encoded["input_ids"].to("cuda:0")
            mask = encoded["attention_mask"].to("cuda:0")
            nonpadding = torch.nonzero(mask[0] == 1, as_tuple=False).flatten()
            require(nonpadding.numel() > 0, f"no nonpadding tokens: {pid}")
            selected = int(nonpadding[-1])
            require(selected == tokens.shape[-1] - 1, f"final nonpadding position mismatch: {pid}")
            _, cache = model.run_with_cache(
                tokens, names_filter=lambda name: name in set(hooks.values()),
                return_type=None, stop_at_layer=32,
            )
            token_id = int(tokens[0, selected])
            meta = {
                "prompt_id": pid, "source_index": int(row["source_index"]), "cwe_id": row["cwe_id"],
                "language": row["language"], "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "sequence_length": int(tokens.shape[-1]), "attention_mask_nonpadding_length": int(nonpadding.numel()),
                "selected_token_index": selected, "selected_token_id": token_id,
                "selected_token_is_special": token_id in special_ids,
            }
            require(not meta["selected_token_is_special"], f"selected final token is special: {pid}")
            for layer in LAYERS:
                vector = cache[hooks[layer]][0, selected].detach().to(dtype=torch.float32, device="cpu").contiguous()
                require(tuple(vector.shape) == (3584,) and bool(torch.isfinite(vector).all()), f"invalid raw activation: {pid}/L{layer}")
                layer_lists[layer].append(vector)
                meta[f"layer_{layer}_raw_sha256"] = tensor_hash(vector)
            new_metadata.append(meta)
            if len(new_metadata) % 10 == 0 or len(new_metadata) == len(missing):
                elapsed = prior_elapsed + time.perf_counter() - started
                atomic_torch(CHECKPOINT, {
                    "schema_version": "phase21_model2_feature_missing_raw_checkpoint_v1",
                    "partition_sha256": PARTITION_HASH, "metadata": new_metadata,
                    "activations": {layer: torch.stack(layer_lists[layer]) for layer in LAYERS},
                    "elapsed_seconds": elapsed,
                })
                state = json.loads(RUN_MANIFEST.read_text(encoding="utf-8"))
                state.update({"completed_missing_records": len(new_metadata), "activation_elapsed_seconds": elapsed})
                atomic_json(RUN_MANIFEST, state)
                print(f"MISSING_ACTIVATIONS {len(new_metadata)}/{len(missing)} elapsed_min={elapsed/60:.1f}", flush=True)
            del encoded, tokens, mask, cache
            torch.cuda.empty_cache()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    new_raw_artifacts = {}
    new_latent_artifacts = {}
    new_latents = {}
    for layer in LAYERS:
        raw_tensor = torch.stack(layer_lists[layer])
        raw_path = ACT / f"model2_feature_missing_layer_{layer}_raw.pt"
        atomic_torch(raw_path, {
            "schema_version": "phase21_model2_feature_missing_raw_v1", "layer": layer,
            "partition_sha256": PARTITION_HASH, "activations": raw_tensor, "metadata": new_metadata,
        })
        new_raw_artifacts[layer] = {"path": str(raw_path.relative_to(ROOT)).replace("\\", "/"), "sha256": file_hash(raw_path)}
        label, params_hash = SAE_SPECS[layer]
        sae_id = f"layer_{layer}/width_16k/average_l0_{label}"
        params = hf_hub_download(SAE_REPOSITORY, f"{sae_id}/params.npz", revision=SAE_REVISION, local_files_only=True)
        require(file_hash(params) == params_hash, f"frozen SAE parameter hash mismatch L{layer}")
        sae = SAE.from_pretrained("gemma-scope-9b-it-res", sae_id, device="cuda:0", dtype="float32").eval()
        batches = []
        with torch.inference_mode():
            for start in range(0, raw_tensor.shape[0], 64):
                batches.append(sae.encode(raw_tensor[start:start + 64].to("cuda:0")).detach().cpu())
        latent_tensor = torch.cat(batches).to(dtype=torch.float32).contiguous()
        require(tuple(latent_tensor.shape) == (232, 16384) and bool(torch.isfinite(latent_tensor).all()), f"invalid missing latent L{layer}")
        latent_path = ACT / f"model2_feature_missing_layer_{layer}_latents.pt"
        atomic_torch(latent_path, {
            "schema_version": "phase21_model2_feature_missing_latents_v1", "layer": layer,
            "partition_sha256": PARTITION_HASH, "sae_id": sae_id, "params_sha256": params_hash,
            "latents": latent_tensor, "metadata": new_metadata,
        })
        new_latents[layer] = latent_tensor
        new_latent_artifacts[layer] = {"path": str(latent_path.relative_to(ROOT)).replace("\\", "/"), "sha256": file_hash(latent_path)}
        del sae, raw_tensor, latent_tensor, batches
        gc.collect()
        torch.cuda.empty_cache()

    new_map = {int(row["prompt_id"]): index for index, row in enumerate(new_metadata)}
    membership = {}
    for target, spec in partition["partitions"].items():
        for pid in spec["unsafe_ids_source_order"]:
            membership.setdefault(pid, {"unsafe": [], "safe": []})["unsafe"].append(target)
        for pid in spec["selected_safe_ids_sampling_order"]:
            membership.setdefault(pid, {"unsafe": [], "safe": []})["safe"].append(target)
    manifest_rows = []
    union_artifacts = {}
    for layer in LAYERS:
        union_vectors = []
        union_metadata = []
        for union_index, pid in enumerate(union):
            if pid in reusable:
                origin_index = accepted_map[pid]
                meta = accepted_meta[origin_index]
                raw_vector = accepted[layer]["raw"]["activations"][origin_index]
                latent_vector = accepted[layer]["latent"]["latents"][origin_index]
                origin = "REUSED_SOURCE_MATCHED"
                raw_origin_path = accepted[layer]["raw_path"]
                latent_origin_path = accepted[layer]["latent_path"]
            else:
                origin_index = new_map[pid]
                meta = new_metadata[origin_index]
                raw_vector = layer_lists[layer][origin_index]
                latent_vector = new_latents[layer][origin_index]
                origin = "NEW_MISSING_ONLY"
                raw_origin_path = ROOT / new_raw_artifacts[layer]["path"]
                latent_origin_path = ROOT / new_latent_artifacts[layer]["path"]
            row = b0[pid]
            require(meta["text_sha256"] == hashlib.sha256(row["generated_code"].encode()).hexdigest(), f"prompt text provenance mismatch: {pid}")
            require(meta["selected_token_index"] == meta["sequence_length"] - 1 and not meta["selected_token_is_special"], f"final-token metadata mismatch: {pid}")
            require(bool(torch.isfinite(latent_vector).all()), f"nonfinite union latent: {pid}/L{layer}")
            union_vectors.append(latent_vector)
            union_metadata.append({"prompt_id": pid, "source_index": int(row["source_index"]), "origin": origin})
            manifest_rows.append({
                "prompt_id": pid, "source_index": int(row["source_index"]), "source_cwe": row["cwe_id"],
                "language": row["language"], "text_sha256": meta["text_sha256"], "layer": layer,
                "hook": f"blocks.{layer}.hook_resid_post", "extraction_origin": origin,
                "sequence_length": meta["sequence_length"], "selected_token_index": meta["selected_token_index"],
                "selected_token_id": meta["selected_token_id"], "selected_token_is_special": meta["selected_token_is_special"],
                "raw_width": 3584, "latent_width": 16384,
                "raw_origin_artifact": str(raw_origin_path.relative_to(ROOT)).replace("\\", "/"),
                "raw_origin_artifact_sha256": file_hash(raw_origin_path), "raw_origin_artifact_row": origin_index,
                "raw_vector_sha256": tensor_hash(raw_vector),
                "latent_origin_artifact": str(latent_origin_path.relative_to(ROOT)).replace("\\", "/"),
                "latent_origin_artifact_sha256": file_hash(latent_origin_path), "latent_origin_artifact_row": origin_index,
                "latent_vector_sha256": tensor_hash(latent_vector), "union_artifact_row": union_index,
                "unsafe_for_cwes": "|".join(membership[pid]["unsafe"]),
                "selected_safe_for_cwes": "|".join(membership[pid]["safe"]),
            })
        union_tensor = torch.stack(union_vectors).to(dtype=torch.float32).contiguous()
        require(tuple(union_tensor.shape) == (302, 16384), f"union latent shape mismatch L{layer}")
        union_path = ACT / f"model2_feature_union_layer_{layer}_latents.pt"
        atomic_torch(union_path, {
            "schema_version": "phase21_model2_feature_union_latents_v1", "layer": layer,
            "partition_sha256": PARTITION_HASH, "latents": union_tensor, "metadata": union_metadata,
        })
        union_artifacts[layer] = {"path": str(union_path.relative_to(ROOT)).replace("\\", "/"), "sha256": file_hash(union_path), "shape": [302, 16384]}
        for row in manifest_rows[-302:]:
            row["union_artifact"] = union_artifacts[layer]["path"]
            row["union_artifact_sha256"] = union_artifacts[layer]["sha256"]

    fields = list(manifest_rows[0])
    tmp = Path(str(LATENT_MANIFEST) + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)
    os.replace(tmp, LATENT_MANIFEST)
    run_state = json.loads(RUN_MANIFEST.read_text(encoding="utf-8"))
    run_state.update({
        "status": "COMPLETE", "completed_missing_records": 232,
        "reused_records": 70, "newly_extracted_records": 232,
        "latent_manifest_path": str(LATENT_MANIFEST.relative_to(ROOT)).replace("\\", "/"),
        "latent_manifest_sha256": file_hash(LATENT_MANIFEST),
        "new_raw_artifacts": new_raw_artifacts, "new_latent_artifacts": new_latent_artifacts,
        "union_artifacts": union_artifacts, "selected_special_token_count": 0,
        "final_token_validation": "PASS", "scanner_run": False, "heldout_used": False,
        "causal_intervention_run": False, "alpha_selected": False,
    })
    atomic_json(RUN_MANIFEST, run_state)
    print(json.dumps(run_state, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = preflight(args.resume)
    print(json.dumps(result, indent=2))
    if not args.preflight_only:
        require(not LATENT_MANIFEST.exists(), "final feature latent manifest already exists")
        run(args.resume)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"MODEL2_FEATURE_LATENT_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
