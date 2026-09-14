#!/usr/bin/env python
"""Extract frozen final-token Model3 residual activations for Stage 22D."""

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

ROOT = Path(__file__).resolve().parents[4]
PHASE = ROOT / "revision/model3/phase22"
OUT = PHASE / "outputs"
ACT = OUT / "activations"
PARTITION = OUT / "phase22d_feature_partition_manifest.json"
PROTOCOL = OUT / "phase22d_activation_ranking_protocol.json"
B0 = OUT / "phase22c_b0_dev_outputs.json"
CHECKPOINT = ACT / "phase22d_raw_activation_checkpoint.pt"
RUN_MANIFEST = OUT / "phase22d_raw_activation_run_manifest.json"
FINAL_MANIFEST = OUT / "phase22d_raw_activation_manifest.json"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
LAYERS = (7, 15, 23)
D_IN = 3584
EXPECTED_COUNT = 144
RUNNER = Path(__file__).resolve()


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_hash(tensor: Any) -> str:
    return hashlib.sha256(tensor.detach().to(dtype=__import__("torch").float32, device="cpu").contiguous().numpy().tobytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_torch(torch: Any, path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def model_snapshot_path() -> Path:
    path = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots" / MODEL_REVISION
    require(path.is_dir() and (path / "config.json").is_file(), f"pinned model snapshot missing: {path}")
    return path


def hidden_from_output(output: Any) -> Any:
    if isinstance(output, tuple):
        return output[0]
    if not hasattr(output, "shape"):
        raise TypeError(f"unsupported decoder-layer output: {type(output)!r}")
    return output


def load_inputs(resume: bool) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    require(PARTITION.is_file() and PROTOCOL.is_file() and B0.is_file(), "frozen Stage 22D inputs missing")
    partition = json.loads(PARTITION.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    b0 = json.loads(B0.read_text(encoding="utf-8"))
    require(partition["status"] == "FROZEN_BEFORE_ACTIVATION_EXTRACTION", "partition not frozen")
    require(protocol["status"] == "FROZEN_BEFORE_ACTIVATION_EXTRACTION", "protocol not frozen")
    require(protocol["scripts"]["raw_extraction_sha256"] == sha256_file(RUNNER), "raw runner drift")
    require(protocol["partition_sha256"] == sha256_file(PARTITION), "partition hash drift")
    by_id = {int(x["prompt_id"]): x for x in b0}
    ids = partition["ranking_prompt_union_ids_source_order"]
    require(len(ids) == EXPECTED_COUNT and len(set(ids)) == EXPECTED_COUNT, "ranking union count/uniqueness mismatch")
    records = [by_id[int(pid)] for pid in ids]
    require(all(x["generation_status"] == "SUCCESS" and x["generated_code"].strip() for x in records), "activation source contains failed/empty generation")
    require([int(x["prompt_id"]) for x in protocol["records"]] == ids, "protocol record ID order mismatch")
    for row, fixed in zip(records, protocol["records"], strict=True):
        require(hashlib.sha256(row["generated_code"].encode()).hexdigest() == fixed["generated_code_sha256"], f"activation text hash drift: {row['prompt_id']}")
    prior = None
    if CHECKPOINT.exists():
        require(resume, "checkpoint exists; use --resume")
        import torch
        prior = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        require(prior["protocol_sha256"] == sha256_file(PROTOCOL), "checkpoint protocol drift")
        require(prior["runner_sha256"] == sha256_file(RUNNER), "checkpoint runner drift")
        completed = len(prior["metadata"])
        require([int(x["prompt_id"]) for x in prior["metadata"]] == ids[:completed], "checkpoint not a source-order prefix")
        require(all(tuple(prior["activations"][layer].shape) == (completed, D_IN) for layer in LAYERS), "checkpoint shape mismatch")
    else:
        require(not resume, "--resume supplied without checkpoint")
    return partition, protocol, b0, records, prior


def preflight(resume: bool) -> dict[str, Any]:
    partition, protocol, _b0, records, prior = load_inputs(resume)
    return {
        "schema_version": "phase22d_raw_activation_preflight_v1", "status": "PASS_APPROVED",
        "record_count": len(records), "layers": list(LAYERS), "d_in": D_IN,
        "completed_checkpoint_records": len(prior["metadata"]) if prior else 0,
        "resume_required": prior is not None, "partition_sha256": sha256_file(PARTITION),
        "protocol_sha256": sha256_file(PROTOCOL), "runner_sha256": sha256_file(RUNNER),
        "model_loaded": False, "activation_run": False, "sae_loaded": False,
        "scanner_run": False, "feature_ranking_run": False, "heldout_used": False,
    }


def run(resume: bool) -> None:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    require(torch.cuda.is_available(), "CUDA unavailable")
    partition, protocol, _b0, records, prior = load_inputs(resume)
    snapshot = model_snapshot_path()
    os.environ["HF_HUB_OFFLINE"] = "1"
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map={"": "cuda:0"}, local_files_only=True,
    ).eval()
    devices = sorted({str(x.device) for x in model.parameters()})
    require(devices == ["cuda:0"], f"unauthorized model offload: {devices}")
    metadata: list[dict[str, Any]] = [] if prior is None else prior["metadata"]
    layer_lists: dict[int, list[Any]] = {layer: [] for layer in LAYERS}
    if prior is not None:
        layer_lists = {layer: [row for row in prior["activations"][layer]] for layer in LAYERS}
    prior_elapsed = float(prior.get("elapsed_seconds", 0.0)) if prior else 0.0
    run_state = {
        "schema_version": "phase22d_raw_activation_run_v1", "status": "IN_PROGRESS",
        "completed_records": len(metadata), "record_count": EXPECTED_COUNT,
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "layers": list(LAYERS),
        "partition_sha256": sha256_file(PARTITION), "protocol_sha256": sha256_file(PROTOCOL),
        "runner_sha256": sha256_file(RUNNER), "torch_version": torch.__version__,
        "transformers_version": transformers.__version__, "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(0), "parameter_devices": devices,
        "sae_loaded": False, "scanner_run": False, "feature_ranking_run": False,
        "causal_intervention_run": False, "heldout_used": False,
    }
    atomic_json(RUN_MANIFEST, run_state)
    selected: dict[int, Any] = {}

    def make_hook(layer: int):
        def hook(_module: Any, _args: Any, output: Any) -> Any:
            selected[layer] = hidden_from_output(output)[0, selected_index].detach().float().cpu().contiguous()
            return output
        return hook

    handles = [model.model.layers[layer].register_forward_hook(make_hook(layer)) for layer in LAYERS]
    special_ids = set(tokenizer.all_special_ids)
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for index in range(len(metadata), len(records)):
                row = records[index]
                encoded = tokenizer(row["generated_code"], return_tensors="pt", add_special_tokens=True, truncation=True, max_length=512)
                tokens = encoded["input_ids"].to("cuda:0")
                mask = encoded["attention_mask"].to("cuda:0")
                nonpadding = torch.nonzero(mask[0] == 1, as_tuple=False).flatten()
                require(nonpadding.numel() > 8, f"sequence outside Qwen SAE post-offset support: {row['prompt_id']}")
                require(int(tokens.shape[-1]) == int(protocol["records"][index]["input_token_count"]), f"tokenization drift: {row['prompt_id']}")
                selected_index = int(nonpadding[-1].item())
                require(selected_index == int(tokens.shape[-1] - 1), "batch-one final nonpadding position mismatch")
                selected.clear()
                model(input_ids=tokens, attention_mask=mask, use_cache=False)
                require(set(selected) == set(LAYERS), "one or more layer hooks did not fire")
                selected_id = int(tokens[0, selected_index].item())
                meta = {
                    "ranking_order": index, "population_index": row["population_index"],
                    "source_index": row["source_index"], "prompt_id": int(row["prompt_id"]),
                    "cwe_id": row["cwe_id"], "language": row["language"].lower(),
                    "generated_code_sha256": hashlib.sha256(row["generated_code"].encode()).hexdigest(),
                    "sequence_length": int(tokens.shape[-1]), "selected_token_index": selected_index,
                    "selected_token_id": selected_id, "selected_token_is_special": selected_id in special_ids,
                }
                for layer in LAYERS:
                    activation = selected[layer]
                    require(tuple(activation.shape) == (D_IN,) and bool(torch.isfinite(activation).all()), f"invalid activation at {row['prompt_id']}/L{layer}")
                    layer_lists[layer].append(activation)
                    meta[f"layer_{layer}_activation_sha256"] = tensor_hash(activation)
                metadata.append(meta)
                if len(metadata) % 10 == 0 or len(metadata) == len(records):
                    elapsed = prior_elapsed + time.perf_counter() - started
                    state = {
                        "schema_version": "phase22d_raw_activation_checkpoint_v1",
                        "partition_sha256": sha256_file(PARTITION), "protocol_sha256": sha256_file(PROTOCOL),
                        "runner_sha256": sha256_file(RUNNER), "metadata": metadata,
                        "activations": {layer: torch.stack(layer_lists[layer]) for layer in LAYERS},
                        "elapsed_seconds": elapsed,
                    }
                    atomic_torch(torch, CHECKPOINT, state)
                    run_state.update({"completed_records": len(metadata), "elapsed_seconds": elapsed})
                    atomic_json(RUN_MANIFEST, run_state)
                    print(f"PHASE22D RAW {len(metadata)}/{EXPECTED_COUNT} elapsed_min={elapsed/60:.1f}", flush=True)
                del encoded, tokens, mask
    finally:
        for handle in handles:
            handle.remove()
    artifacts = {}
    for layer in LAYERS:
        path = ACT / f"phase22d_layer_{layer}_raw.pt"
        payload = {
            "schema_version": "phase22d_raw_activations_v1", "layer": layer,
            "hook": f"model.model.layers[{layer}] forward output (resid_post_layer_{layer})",
            "partition_sha256": sha256_file(PARTITION), "protocol_sha256": sha256_file(PROTOCOL),
            "activations": torch.stack(layer_lists[layer]), "metadata": metadata,
        }
        atomic_torch(torch, path, payload)
        artifacts[str(layer)] = {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256_file(path), "shape": [EXPECTED_COUNT, D_IN]}
    run_state.update({
        "status": "COMPLETE", "completed_records": EXPECTED_COUNT, "artifacts": artifacts,
        "selected_special_token_count": sum(x["selected_token_is_special"] for x in metadata),
        "final_token_validation": "PASS",
    })
    atomic_json(RUN_MANIFEST, run_state)
    atomic_json(FINAL_MANIFEST, {
        "schema_version": "phase22d_raw_activation_manifest_v1", "status": "COMPLETE",
        "partition_sha256": sha256_file(PARTITION), "protocol_sha256": sha256_file(PROTOCOL),
        "runner_sha256": sha256_file(RUNNER), "record_count": EXPECTED_COUNT,
        "prompt_ids_source_order": partition["ranking_prompt_union_ids_source_order"],
        "artifacts": artifacts, "metadata": metadata, "scanner_run": False,
        "feature_ranking_run": False, "causal_intervention_run": False, "heldout_used": False,
    })
    print(json.dumps(run_state, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(preflight(args.resume), indent=2))
    if not args.preflight_only:
        require(not FINAL_MANIFEST.exists(), "final raw activation manifest already exists")
        run(args.resume)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"PHASE22D_RAW_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
