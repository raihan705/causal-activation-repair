#!/usr/bin/env python3
"""Shared fail-closed provenance and execution helpers for revision Phase 17."""

from __future__ import annotations

import ast
import base64
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
PHASE17 = ROOT / "revision/model1/phase17"
OUTPUTS = PHASE17 / "outputs"
INPUT_MANIFEST = OUTPUTS / "phase17_input_manifest.json"
PREPARATION_CHECKPOINT = OUTPUTS / "phase17_preparation_checkpoint.json"
EXECUTION_APPROVAL = OUTPUTS / "phase17_execution_approval.json"
COMMON_HELPER = Path(__file__).resolve()

PHASE16_CHECKPOINT = ROOT / "revision/model1/phase16/outputs/phase16_checkpoint.json"
PHASE15_FREEZE = ROOT / "revision/model1/phase15/outputs/corrected_revision_freeze_manifest.json"
PHASE14_STATUSES = ROOT / "revision/model1/phase14/outputs/phase14_method_statuses.json"
ALWAYSON_CONFIG = ROOT / "revision/model1/phase14/outputs/bstar_alwayson_l19_config.json"
DENOMINATOR = ROOT / "revision/model1/phase2/phase2_denominator_audit.json"
TEST_PROMPTS = ROOT / "data/cyberseceval/test_prompts.json"
SCANNER = ROOT / "phases/phase9/colab_scan_phase9.py"

EXPECTED_HASHES = {
    "phase16_checkpoint": "79690bf030eb197672eeb1d5121723682d0c365a9c9c2236e40b9956a2573be2",
    "phase15_freeze": "4c97b2ba96c50d564a60bd000a2a04df02125ec0173fea4a68a78c60d07ec8bc",
    "phase14_statuses": "07416f99bb42ba3d2f4a52c67c4a06a010093ad8669e6b3d36023d40ac4a23ce",
    "alwayson_config": "1adc401c9d713b7f9a968161d5eceea61297a44df0d0b4a6fe3677fb26c7e48b",
    "denominator": "75b5ca0a029e43a49a9048d17a94713c0003aad6f144352539c6b0ecede3f619",
    "test_prompts": "7f015c7e398445953e05f9eddb17a9791bd4efb2103e236d20b7ac7ce28695ab",
    "scanner": "60cedd3d6ec2d19e5e7e5ba4d709230efdd5d9478047492fefda4dd88d9afbb7",
}

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
SAE_RELEASE = "llama_scope_lxr_8x"
SAE_ID = "l19r_8x"
SAE_REVISION = "8dbc1d85edfced43081c03c38b05514dbab1368b"
SEED = 42
LAYER = 19
FEATURES = (14193, 16897, 11462, 151)
ALPHA = 40.0
CHECKPOINT_INTERVAL = 50


class Phase17Error(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise Phase17Error(message)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_text(payload)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def require_hash(path: Path, expected: str, label: str) -> None:
    require(path.is_file(), f"Missing {label}: {path}")
    require(sha256_path(path) == expected, f"{label} hash mismatch")


def dependency_gate() -> dict[str, Any]:
    paths = {
        "phase16_checkpoint": PHASE16_CHECKPOINT,
        "phase15_freeze": PHASE15_FREEZE,
        "phase14_statuses": PHASE14_STATUSES,
        "alwayson_config": ALWAYSON_CONFIG,
        "denominator": DENOMINATOR,
        "test_prompts": TEST_PROMPTS,
        "scanner": SCANNER,
    }
    for label, path in paths.items():
        require_hash(path, EXPECTED_HASHES[label], label)
    checkpoint = read_json(PHASE16_CHECKPOINT)
    require(checkpoint.get("phase_status") == "DONE", "Phase 16 is not DONE")
    require(checkpoint.get("checkpoint") == "PASS_WITH_PROVENANCE_LIMITATION",
            "Phase 16 checkpoint is not the approved limited pass")
    statuses = read_json(PHASE14_STATUSES)
    always = statuses.get("methods", {}).get("B*-AlwaysOn-L19", {})
    require(always.get("status") == "FROZEN" and always.get("eligible_for_heldout") is True,
            "AlwaysOn is not frozen/held-out eligible")
    config = read_json(ALWAYSON_CONFIG)
    require(config.get("status") == "FROZEN", "AlwaysOn config is not frozen")
    require(config.get("information_tier") == "METADATA_FREE", "AlwaysOn tier mismatch")
    require(config.get("layer") == LAYER and config.get("features") == list(FEATURES),
            "AlwaysOn feature/layer mismatch")
    require(all(float(config["alpha_by_feature"][str(feature)]) == ALPHA for feature in FEATURES),
            "AlwaysOn alpha mismatch")
    require(config.get("routing_fields_consumed") == [], "AlwaysOn config consumes routing fields")
    return {"status": "PASS", "hashes": {key: sha256_path(path) for key, path in paths.items()}}


def load_prepared_manifest() -> tuple[dict[str, Any], dict[str, Any]]:
    dependency_gate()
    require(INPUT_MANIFEST.is_file() and PREPARATION_CHECKPOINT.is_file(),
            "Phase 17 preparation artifacts are missing")
    checkpoint = read_json(PREPARATION_CHECKPOINT)
    require(checkpoint.get("status") == "PASS_AWAITING_EXECUTION_APPROVAL",
            "Phase 17 preparation checkpoint is not ready")
    require(sha256_path(INPUT_MANIFEST) == checkpoint["input_manifest_sha256"],
            "Phase 17 input manifest hash mismatch")
    manifest = read_json(INPUT_MANIFEST)
    require(manifest.get("schema_version") == "phase17_input_manifest_v1", "Input manifest schema mismatch")
    require(manifest.get("condition", {}).get("features") == list(FEATURES), "Manifest feature mismatch")
    require(manifest.get("condition", {}).get("routing_fields_consumed") == [], "Manifest routing mismatch")
    return manifest, checkpoint


def execution_approval(runner: Path) -> dict[str, Any]:
    manifest, checkpoint = load_prepared_manifest()
    require(EXECUTION_APPROVAL.is_file(),
            "Phase 17 execution approval is absent; only --preflight-only is permitted")
    approval = read_json(EXECUTION_APPROVAL)
    require(approval.get("status") == "APPROVED_FOR_GPU_EXECUTION", "GPU execution is not approved")
    require(approval.get("input_manifest_sha256") == checkpoint["input_manifest_sha256"],
            "Execution approval input-manifest hash mismatch")
    approved_runners = approval.get("runner_hashes", {})
    require(approved_runners.get(relative(runner)) == sha256_path(runner),
            "Execution approval runner hash mismatch")
    require(approved_runners.get(relative(COMMON_HELPER)) == sha256_path(COMMON_HELPER),
            "Execution approval shared-helper hash mismatch")
    require(approval.get("condition") == manifest["condition"], "Execution approval condition mismatch")
    return approval


def consumed_literal_fields(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    fields: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            fields.add(node.slice.value)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get" and node.args and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            fields.add(node.args[0].value)
    return fields


def require_metadata_free_runner(path: Path) -> dict[str, Any]:
    fields = consumed_literal_fields(path)
    forbidden = fields & {"cwe", "cwe_id", "cwe_identifier"}
    require(not forbidden, f"Runner consumes forbidden routing fields: {sorted(forbidden)}")
    return {"consumed_literal_fields_sha256": canonical_sha256(sorted(fields)),
            "forbidden_routing_fields": [], "status": "PASS"}


def capture_rng_state(torch_module: Any) -> dict[str, Any]:
    def encode(tensor: Any) -> str:
        return base64.b64encode(bytes(tensor.detach().cpu().tolist())).decode("ascii")
    states = torch_module.cuda.get_rng_state_all() if torch_module.cuda.is_available() else []
    return {"torch_cpu_b64": encode(torch_module.get_rng_state()),
            "torch_cuda_all_b64": [encode(state) for state in states],
            "cuda_device_count": len(states)}


def restore_rng_state(torch_module: Any, payload: dict[str, Any]) -> None:
    def decode(value: str) -> Any:
        return torch_module.tensor(list(base64.b64decode(value, validate=True)), dtype=torch_module.uint8)
    require((torch_module.cuda.device_count() if torch_module.cuda.is_available() else 0)
            == int(payload["cuda_device_count"]), "Checkpoint CUDA device count mismatch")
    torch_module.set_rng_state(decode(payload["torch_cpu_b64"]))
    if payload["torch_cuda_all_b64"]:
        torch_module.cuda.set_rng_state_all([decode(value) for value in payload["torch_cuda_all_b64"]])


def load_model_and_sae():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from sae_lens import SAE
    from transformers import AutoModelForCausalLM, AutoTokenizer
    require(torch.cuda.is_available(), "CUDA is required for Phase 17 generation")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True,
        torch_dtype=torch.float16, device_map="auto",
    ).eval()
    sae, _, _ = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID)
    return torch, tokenizer, model, sae.to("cuda").eval()


def make_alwayson_hook(sae: Any, tracker: dict[str, Any]):
    def hook_fn(module: Any, inputs: tuple[Any, ...], output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        latent = sae.encode(hidden)
        for feature in FEATURES:
            latent[..., feature] += ALPHA
        edited = sae.decode(latent).to(hidden.dtype)
        signature = {"shape": list(hidden.shape), "dtype": str(hidden.dtype),
                     "device": str(hidden.device), "layer": LAYER,
                     "features": list(FEATURES), "alpha": ALPHA}
        tracker["hook_call_count"] += 1
        tracker["first_hook"] = tracker.get("first_hook") or signature
        tracker["last_hook"] = signature
        if isinstance(output, tuple):
            return (edited,) + output[1:]
        return edited
    return hook_fn


def generate_with_alwayson(torch_module: Any, tokenizer: Any, model: Any, sae: Any,
                           rendered_prompt: str, input_truncation: int,
                           max_new_tokens: int,
                           eos_token_id: int | list[int] | None = None,
                           ) -> tuple[str, int, int, dict[str, Any]]:
    tracker: dict[str, Any] = {"hook_call_count": 0}
    inputs = tokenizer(rendered_prompt, return_tensors="pt", truncation=True,
                       max_length=input_truncation).to(model.device)
    handle = model.model.layers[LAYER].register_forward_hook(make_alwayson_hook(sae, tracker))
    try:
        with torch_module.inference_mode():
            generation_kwargs = {
                "max_new_tokens": max_new_tokens, "temperature": 0.2,
                "top_p": 0.95, "do_sample": True,
                "pad_token_id": tokenizer.pad_token_id,
            }
            if eos_token_id is not None:
                generation_kwargs["eos_token_id"] = eos_token_id
            generated = model.generate(**inputs, **generation_kwargs)
    finally:
        handle.remove()
    new_tokens = generated[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text, int(new_tokens.shape[0]), int(inputs["input_ids"].shape[1]), tracker


def empty_status(value: str) -> str:
    if value == "":
        return "STRICT_EMPTY"
    if not value.strip():
        return "WHITESPACE_ONLY"
    return "NONEMPTY"
