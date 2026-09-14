#!/usr/bin/env python
"""Extract exact final-nonpadding TransformerLens activations for Phase21."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
PHASE = ROOT / "revision/model2/phase21"
OUT = PHASE / "outputs"
ACT = OUT / "activations"
SOURCE_MANIFEST = OUT / "model2_activation_source_manifest.json"
CHECKPOINT = ACT / "model2_raw_activation_checkpoint.pt"
RUN_MANIFEST = OUT / "model2_activation_run_manifest.json"
FINAL_MANIFEST = OUT / "model2_activation_manifest.csv"
TRAIN = ROOT / "data/processed/train_pairs.csv"
VAL = ROOT / "data/processed/val_pairs.csv"
B0 = OUT / "model2_b0_dev_outputs.json"
MODEL_ID = "google/gemma-2-9b-it"
MODEL_REVISION = "11c9b309abf73637e4b6f9a3fa1e92e615547819"
LAYERS = (9, 20, 31)
RUNNER = Path(__file__).resolve()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_hash(t: Any) -> str:
    return hashlib.sha256(t.detach().to(dtype=__import__("torch").float32, device="cpu").contiguous().numpy().tobytes()).hexdigest()


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def reconstruct_records(manifest: dict[str, Any]) -> list[tuple[dict[str, Any], str]]:
    train, val = read_csv(TRAIN), read_csv(VAL)
    pairs = {(split, i): row for split, rows in (("train", train), ("validation", val)) for i, row in enumerate(rows)}
    b0 = {int(x["prompt_id"]): x for x in json.loads(B0.read_text(encoding="utf-8"))}
    result = []
    for frozen in manifest["records"]:
        if frozen["source"] in {"CVE_VULNERABLE", "CVE_FIXED"}:
            row = pairs[(frozen["source_split"], int(frozen["source_split_index"]))]
            text = row[frozen["text_field"]]
            require(row["pair_id"] == frozen["pair_id"], "CVE pair alignment drift")
        else:
            row = b0[int(frozen["prompt_id"])]
            text = row["generated_code"]
        require(hashlib.sha256(text.encode()).hexdigest() == frozen["text_sha256"], "activation source text hash drift")
        result.append((frozen, text))
    return result


def preflight(resume: bool) -> dict[str, Any]:
    require(SOURCE_MANIFEST.is_file(), "frozen activation source manifest missing")
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    require(manifest["status"] == "FROZEN_BEFORE_ACTIVATION_EXTRACTION" and manifest["record_count"] == 517, "source manifest state mismatch")
    require(manifest["runner"]["sha256"] == sha256_file(RUNNER), "runner changed after source freeze")
    records = reconstruct_records(manifest)
    completed = 0
    if CHECKPOINT.exists():
        require(resume, "activation checkpoint exists; use --resume")
        import torch
        state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        completed = len(state["metadata"])
        require(state["source_manifest_sha256"] == sha256_file(SOURCE_MANIFEST), "checkpoint source hash mismatch")
        require(state["runner_sha256"] == sha256_file(RUNNER), "checkpoint runner hash mismatch")
        require([x["record_id"] for x in state["metadata"]] == [x[0]["record_id"] for x in records[:completed]], "checkpoint is not a source-order prefix")
        require(all(state["activations"][layer].shape == (completed, 3584) for layer in LAYERS), "checkpoint activation shape mismatch")
    else:
        require(not resume, "--resume supplied without checkpoint")
    return {"status": "PASS_APPROVED", "source_count": len(records), "completed_checkpoint_records": completed, "model_loaded": False, "activation_run": False, "sae_loaded": False, "scanner_run": False, "heldout_used": False, "feature_ranking_run": False, "causal_intervention_run": False, "source_manifest_sha256": sha256_file(SOURCE_MANIFEST), "runner_sha256": sha256_file(RUNNER)}


def atomic_torch_save(torch: Any, path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def run(resume: bool) -> None:
    import torch
    from transformer_lens import HookedTransformer
    from transformers import AutoTokenizer

    require(torch.cuda.is_available(), "CUDA unavailable")
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    records = reconstruct_records(manifest)
    metadata: list[dict[str, Any]] = []
    layer_lists: dict[int, list[Any]] = {layer: [] for layer in LAYERS}
    prior_elapsed = 0.0
    if CHECKPOINT.exists():
        state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        metadata = state["metadata"]
        layer_lists = {layer: [row for row in state["activations"][layer]] for layer in LAYERS}
        prior_elapsed = float(state.get("elapsed_seconds", 0.0))
    os.environ["HF_HUB_OFFLINE"] = "1"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    model = HookedTransformer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, tokenizer=tokenizer, device="cuda:0",
        n_devices=1, dtype=torch.bfloat16, fold_ln=True,
        center_writing_weights=False, center_unembed=False, move_to_device=True,
        local_files_only=True,
    ).eval()
    require(sorted({str(x.device) for x in model.parameters()}) == ["cuda:0"], "unauthorized model offload")
    start = len(metadata)
    started = time.perf_counter()
    atomic_json(RUN_MANIFEST, {"schema_version": "phase21_model2_activation_run_v1", "status": "IN_PROGRESS", "completed_records": start, "source_count": 517, "source_manifest_sha256": sha256_file(SOURCE_MANIFEST), "runner_sha256": sha256_file(RUNNER), "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "layers": list(LAYERS), "sae_loaded": False, "scanner_run": False, "feature_ranking_run": False, "causal_intervention_run": False, "heldout_used": False})
    hooks = {layer: f"blocks.{layer}.hook_resid_post" for layer in LAYERS}
    special_ids = set(tokenizer.all_special_ids)
    with torch.inference_mode():
        for index in range(start, len(records)):
            frozen, text = records[index]
            encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=512)
            tokens = encoded["input_ids"].to("cuda:0")
            mask = encoded["attention_mask"].to("cuda:0")
            nonpadding = torch.nonzero(mask[0] == 1, as_tuple=False).flatten()
            require(nonpadding.numel() > 0, "tokenization produced no nonpadding token")
            selected_index = int(nonpadding[-1].item())
            require(selected_index == int(tokens.shape[-1] - 1), "batch-one final nonpadding index is not final position")
            _, cache = model.run_with_cache(tokens, names_filter=lambda name: name in set(hooks.values()), return_type=None, stop_at_layer=32)
            selected_id = int(tokens[0, selected_index].item())
            row_meta = dict(frozen)
            row_meta.update({"sequence_length": int(tokens.shape[-1]), "attention_mask_nonpadding_length": int(nonpadding.numel()), "selected_token_index": selected_index, "selected_token_id": selected_id, "selected_token_is_special": selected_id in special_ids})
            for layer in LAYERS:
                activation = cache[hooks[layer]][0, selected_index].detach().to(dtype=torch.float32, device="cpu").contiguous()
                require(tuple(activation.shape) == (3584,) and bool(torch.isfinite(activation).all()), f"invalid activation at row {index}/L{layer}")
                layer_lists[layer].append(activation)
                row_meta[f"layer_{layer}_activation_sha256"] = tensor_hash(activation)
            metadata.append(row_meta)
            if len(metadata) % 10 == 0 or len(metadata) == len(records):
                elapsed = prior_elapsed + time.perf_counter() - started
                state = {"schema_version": "phase21_model2_raw_activation_checkpoint_v1", "source_manifest_sha256": sha256_file(SOURCE_MANIFEST), "runner_sha256": sha256_file(RUNNER), "metadata": metadata, "activations": {layer: torch.stack(layer_lists[layer]) for layer in LAYERS}, "elapsed_seconds": elapsed}
                atomic_torch_save(torch, CHECKPOINT, state)
                run_state = json.loads(RUN_MANIFEST.read_text(encoding="utf-8"))
                run_state.update({"completed_records": len(metadata), "elapsed_seconds": elapsed})
                atomic_json(RUN_MANIFEST, run_state)
                print(f"ACTIVATIONS {len(metadata)}/517 elapsed_min={elapsed/60:.1f}", flush=True)
            del encoded, tokens, mask, cache
            torch.cuda.empty_cache()
    artifacts = {}
    for layer in LAYERS:
        path = ACT / f"model2_layer_{layer}_raw.pt"
        value = {"schema_version": "phase21_model2_raw_activations_v1", "layer": layer, "hook": hooks[layer], "source_manifest_sha256": sha256_file(SOURCE_MANIFEST), "activations": torch.stack(layer_lists[layer]), "metadata": metadata}
        atomic_torch_save(torch, path, value)
        artifacts[layer] = {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256_file(path), "shape": [517, 3584]}
    fields = ["source_order", "source", "source_split", "source_split_index", "record_id", "pair_id", "prompt_id", "cve_id", "cwe_id", "language", "text_sha256", "sequence_length", "attention_mask_nonpadding_length", "selected_token_index", "selected_token_id", "selected_token_is_special", "layer", "activation_shape", "activation_sha256", "activation_artifact", "activation_artifact_row"]
    tmp = FINAL_MANIFEST.with_name(FINAL_MANIFEST.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row_index, row in enumerate(metadata):
            for layer in LAYERS:
                writer.writerow({**{k: row.get(k) for k in fields if k not in {"layer", "activation_shape", "activation_sha256", "activation_artifact", "activation_artifact_row"}}, "layer": layer, "activation_shape": "3584", "activation_sha256": row[f"layer_{layer}_activation_sha256"], "activation_artifact": artifacts[layer]["path"], "activation_artifact_row": row_index})
    os.replace(tmp, FINAL_MANIFEST)
    run_state = json.loads(RUN_MANIFEST.read_text(encoding="utf-8"))
    run_state.update({"status": "COMPLETE", "completed_records": 517, "activation_manifest_sha256": sha256_file(FINAL_MANIFEST), "artifacts": artifacts, "selected_special_token_count": sum(x["selected_token_is_special"] for x in metadata), "final_token_validation": "PASS"})
    atomic_json(RUN_MANIFEST, run_state)
    print(json.dumps(run_state, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    pf = preflight(args.resume)
    print(json.dumps(pf, indent=2))
    if not args.preflight_only:
        require(not FINAL_MANIFEST.exists(), "final activation manifest already exists")
        run(args.resume)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"PHASE21_ACTIVATION_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
