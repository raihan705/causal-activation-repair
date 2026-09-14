#!/usr/bin/env python3
"""Generate the frozen Model2 alpha20/alpha40 development-strength conditions."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
PHASE = ROOT / "revision/model2/phase21"
OUT = PHASE / "outputs"
RUNNER = Path(__file__).resolve()
ROUTES = OUT / "model2_selected_routes.json"
POPULATION = OUT / "model2_strength_population_manifest.json"
PROTOCOL = OUT / "model2_strength_selection_protocol.json"

MODEL_ID = "google/gemma-2-9b-it"
MODEL_REVISION = "11c9b309abf73637e4b6f9a3fa1e92e615547819"
SAE_REPOSITORY = "google/gemma-scope-9b-it-res"
SAE_REVISION = "e86af97a5b6fbbccca28ab654f2fda1b0768f770"
SAE_ID = "layer_20/width_16k/average_l0_47"
SAE_PARAMS_SHA256 = "63337f10014d4c9096c51ecc372d1b61a8ef2642be2ca01a0d04ccd3dc0e6bf2"
HOOK_NAME = "blocks.20.hook_resid_post"
LAYER = 20
ALPHAS = (20, 40)
SEED = 42
CHECKPOINT_INTERVAL = 25
GENERATION = {
    "temperature": 0.2,
    "top_p": 0.95,
    "max_new_tokens": 512,
    "do_sample": True,
}
FROZEN_ROUTES = {
    "CWE-120": {"layer": 20, "feature_id": 14471},
    "CWE-327": {"layer": 20, "feature_id": 529},
    "CWE-89": {"layer": 20, "feature_id": 12328},
}
EXPECTED_HASHES = {
    ROUTES: "0465330304be21d3c516fc30c12c51f1cb5786c807ce3c6ba17cfbf81399b9da",
    POPULATION: "ff3b76b306d1dcfdbfd208259a23cc35a4b8b6b44dfb582d3f8fb8fcdc84cc71",
    PROTOCOL: "8ef127425206dc04250ff79041266d00ed244a1f07844266829ee7f29ae9adf3",
}
ROUTE_MAP_CANONICAL_SHA256 = "3c4d060906e863b037d66b033234c9a8911753c5dc029f124232be5c229e87f0"
PROMPT_IDS_SHA256 = "0e5056866ddcd9df381c4bb275d86c44dda1e1c0b054b59a987516173e05e000"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def output_path(alpha: int) -> Path:
    return OUT / f"model2_alpha{alpha}_dev_outputs.json"


def manifest_path(alpha: int) -> Path:
    return OUT / f"model2_alpha{alpha}_generation_manifest.json"


def checkpoint_path(alpha: int) -> Path:
    return OUT / f"model2_alpha{alpha}_generation_checkpoint.json"


def condition_key(alpha: int, source_cwe: str, prompt_id: int) -> str:
    return f"A{alpha}|{source_cwe}|P{prompt_id}"


def load_and_validate_frozen() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    for path, expected in EXPECTED_HASHES.items():
        require(path.is_file(), f"Missing frozen artifact: {path.name}")
        require(sha256_file(path) == expected, f"Frozen artifact hash mismatch: {path.name}")
    routes = load_json(ROUTES)
    population = load_json(POPULATION)
    protocol = load_json(PROTOCOL)
    require(routes["routes"] == FROZEN_ROUTES, "Frozen routes changed")
    require(routes["route_map_canonical_sha256"] == ROUTE_MAP_CANONICAL_SHA256, "Route-map hash changed")
    require(population["status"] == "FROZEN_BEFORE_STRENGTH_GENERATION", "Population is not frozen")
    require(population["split"] == "DEVELOPMENT", "Population is not development")
    require(population["heldout_file_read"] is False and population["heldout_ids_used"] is False, "Population reports held-out access")
    records = population["records"]
    require(len(records) == 180, "Population record count is not 180")
    ids = [int(record["prompt_id"]) for record in records]
    require(len(set(ids)) == 180, "Population contains duplicate prompt IDs")
    require(canonical_sha256(ids) == PROMPT_IDS_SHA256, "Frozen prompt-ID sequence hash mismatch")
    require([int(record["population_index"]) for record in records] == list(range(180)), "Population index/order mismatch")
    require([int(record["source_index"]) for record in records] == sorted(int(record["source_index"]) for record in records), "Population source order mismatch")
    counts = {cwe: sum(record["cwe_identifier"] == cwe for record in records) for cwe in FROZEN_ROUTES}
    require(counts == {"CWE-120": 110, "CWE-327": 46, "CWE-89": 24}, f"Source CWE count mismatch: {counts}")
    for record in records:
        cwe = record["cwe_identifier"]
        require(record["b0_scanner_eligible"] is True, "Scanner-ineligible population record")
        require(record["selected_route"] == FROZEN_ROUTES[cwe], "Per-record route mismatch")
    require(protocol["status"] == "FROZEN_BEFORE_GENERATION_OR_STRENGTH_OUTCOME", "Strength protocol is not frozen")
    require(protocol["strength_candidates"] == [20, 40], "Strength grid changed")
    require(protocol["forbidden_rescue_alphas"] == [10, 30, 50, 60, 80], "Forbidden alpha list changed")
    require(protocol["generation"] == {
        "seed": 42,
        "temperature": 0.2,
        "top_p": 0.95,
        "max_new_tokens": 512,
        "do_sample": True,
        "decode_generated_tokens_only": True,
        "skip_special_tokens": True,
    }, "Generation protocol changed")
    require(protocol["model"]["id"] == MODEL_ID and protocol["model"]["revision"] == MODEL_REVISION, "Model protocol changed")
    require(protocol["routing"]["selected_layer"] == 20 and protocol["routing"]["features_per_prompt"] == 1, "Routing protocol changed")
    return routes, records, protocol


def validate_record(record: dict[str, Any], alpha: int, population_record: dict[str, Any]) -> None:
    cwe = population_record["cwe_identifier"]
    expected_key = condition_key(alpha, cwe, int(population_record["prompt_id"]))
    require(record["condition_key"] == expected_key, "Generation condition key mismatch")
    require(int(record["prompt_id"]) == int(population_record["prompt_id"]), "Generation prompt ID mismatch")
    require(record["source_cwe"] == cwe, "Generation source CWE mismatch")
    require(int(record["layer"]) == 20, "Generation layer mismatch")
    require(int(record["feature_id"]) == FROZEN_ROUTES[cwe]["feature_id"], "Generation feature mismatch")
    require(int(record["alpha"]) == alpha, "Generation alpha mismatch")
    require(int(record["seed"]) == 42, "Generation seed mismatch")
    require(record["model_id"] == MODEL_ID and record["model_revision"] == MODEL_REVISION, "Generation model mismatch")
    require(record["sae_id"] == SAE_ID and record["sae_revision"] == SAE_REVISION, "Generation SAE mismatch")
    require(record["fallback_status"] == "NONE", "Generation fallback detected")


def validate_complete_condition(alpha: int, population: list[dict[str, Any]]) -> bool:
    out_path = output_path(alpha)
    man_path = manifest_path(alpha)
    if not out_path.exists() and not man_path.exists():
        return False
    require(out_path.is_file() and man_path.is_file(), f"Partial immutable alpha{alpha} finalization")
    records = load_json(out_path)
    manifest = load_json(man_path)
    require(isinstance(records, list) and len(records) == 180, f"Alpha{alpha} final count mismatch")
    for record, population_record in zip(records, population):
        validate_record(record, alpha, population_record)
    keys = [record["condition_key"] for record in records]
    require(len(set(keys)) == 180, f"Alpha{alpha} duplicate final keys")
    require(manifest["status"] in {"COMPLETE", "COMPLETE_WITH_PRESERVED_FAILURES"}, f"Alpha{alpha} manifest incomplete")
    require(manifest["generation_output_sha256"] == sha256_file(out_path), f"Alpha{alpha} output hash linkage mismatch")
    require(manifest["runner_sha256"] == sha256_file(RUNNER), f"Alpha{alpha} runner linkage mismatch")
    require(manifest["strength_protocol_sha256"] == EXPECTED_HASHES[PROTOCOL], f"Alpha{alpha} protocol linkage mismatch")
    return True


def load_checkpoint(alpha: int, population: list[dict[str, Any]], resume: bool) -> tuple[list[dict[str, Any]], float]:
    path = checkpoint_path(alpha)
    if not path.exists():
        require(not resume, f"--resume supplied but alpha{alpha} has no checkpoint")
        return [], 0.0
    require(resume, f"Alpha{alpha} checkpoint exists; use --resume")
    checkpoint = load_json(path)
    require(checkpoint["runner_sha256"] == sha256_file(RUNNER), f"Alpha{alpha} checkpoint runner mismatch")
    require(checkpoint["strength_protocol_sha256"] == EXPECTED_HASHES[PROTOCOL], f"Alpha{alpha} checkpoint protocol mismatch")
    require(int(checkpoint["alpha"]) == alpha, f"Alpha{alpha} checkpoint alpha mismatch")
    records = checkpoint["records"]
    require(len(records) <= 180, f"Alpha{alpha} checkpoint overflow")
    for record, population_record in zip(records, population):
        validate_record(record, alpha, population_record)
    require(len({record["condition_key"] for record in records}) == len(records), f"Alpha{alpha} duplicate checkpoint keys")
    return records, float(checkpoint.get("elapsed_seconds", 0.0))


def preflight(resume: bool) -> dict[str, Any]:
    _, population, _ = load_and_validate_frozen()
    conditions = []
    for alpha in ALPHAS:
        complete = validate_complete_condition(alpha, population)
        checkpoint_count = 0
        if not complete and checkpoint_path(alpha).exists():
            records, _ = load_checkpoint(alpha, population, resume)
            checkpoint_count = len(records)
        elif not complete:
            require(not resume or any(validate_complete_condition(prior, population) for prior in ALPHAS if prior < alpha), "--resume supplied without any completed condition or checkpoint")
        conditions.append({"alpha": alpha, "complete": complete, "checkpoint_records": checkpoint_count})
    require(not conditions[1]["complete"] or conditions[0]["complete"], "Alpha40 complete while alpha20 incomplete")
    return {
        "schema_version": "phase21_model2_strength_generation_preflight_v1",
        "status": "PASS_APPROVED",
        "routes_sha256": EXPECTED_HASHES[ROUTES],
        "route_map_canonical_sha256": ROUTE_MAP_CANONICAL_SHA256,
        "population_sha256": EXPECTED_HASHES[POPULATION],
        "prompt_ids_sha256": PROMPT_IDS_SHA256,
        "strength_protocol_sha256": EXPECTED_HASHES[PROTOCOL],
        "runner_sha256": sha256_file(RUNNER),
        "conditions": conditions,
        "expected_records_per_condition": 180,
        "expected_total_records": 360,
        "model_loaded": False,
        "sae_loaded": False,
        "generation_run": False,
        "scanner_run": False,
        "heldout_accessed": False,
        "model1_modified": False,
    }


def load_exact_sae(params_path: str) -> Any:
    import numpy as np
    import torch
    from sae_lens.saes.jumprelu_sae import JumpReLUSAE, JumpReLUSAEConfig

    sae = JumpReLUSAE(JumpReLUSAEConfig(
        d_in=3584,
        d_sae=16384,
        dtype="float32",
        device="cuda:0",
        apply_b_dec_to_input=False,
        normalize_activations="none",
    )).eval()
    state: dict[str, Any] = {}
    with np.load(params_path) as data:
        for key in data.files:
            mapped = "W_" + key[2:] if key.startswith("w_") else key
            state[mapped] = torch.from_numpy(np.asarray(data[key])).to(device="cuda:0", dtype=torch.float32)
    if "scaling_factor" in state:
        scaling = state.pop("scaling_factor")
        require(torch.allclose(scaling, torch.ones_like(scaling)), "Non-unit SAE scaling factor")
    missing, unexpected = sae.load_state_dict(state, strict=False)
    require(not missing and not unexpected, f"SAE state mismatch: missing={missing}, unexpected={unexpected}")
    for parameter in sae.parameters():
        parameter.requires_grad_(False)
    return sae


def generate_condition(
    alpha: int,
    population: list[dict[str, Any]],
    resume: bool,
    model: Any,
    tokenizer: Any,
    sae: Any,
) -> None:
    import torch

    if validate_complete_condition(alpha, population):
        print(f"ALPHA{alpha} ALREADY_COMPLETE", flush=True)
        return
    records, prior_elapsed = load_checkpoint(alpha, population, resume)
    started = time.perf_counter()
    for index in range(len(records), len(population)):
        item = population[index]
        source_cwe = item["cwe_identifier"]
        feature_id = int(FROZEN_ROUTES[source_cwe]["feature_id"])
        prompt = item["test_case_prompt"]
        require(canonical_sha256(prompt) == item["source_prompt_sha256"], f"Source prompt drift: {item['prompt_id']}")
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        require(sha256_text(rendered) == item["b0_rendered_prompt_sha256"], f"Rendered prompt drift: {item['prompt_id']}")
        tokens = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda:0")
        input_token_count = int(tokens.shape[-1])
        diagnostics = {
            "hook_call_count": 0,
            "hook_finite": True,
            "residual_replacement_delta_l2_min": None,
            "residual_replacement_delta_l2_max": 0.0,
        }
        feature_delta_l2 = float((float(alpha) * sae.W_dec[feature_id].detach()).norm().item())

        def steering_hook(activation: Any, hook: Any) -> Any:
            diagnostics["hook_call_count"] += 1
            source = activation[:, -1:, :].to(dtype=torch.float32)
            latents = sae.encode(source)
            modified = latents.clone()
            modified[..., feature_id] += float(alpha)
            replacement = sae.decode(modified)
            delta_l2 = float((replacement - source).norm(dim=-1).max().item())
            diagnostics["residual_replacement_delta_l2_max"] = max(diagnostics["residual_replacement_delta_l2_max"], delta_l2)
            previous_min = diagnostics["residual_replacement_delta_l2_min"]
            diagnostics["residual_replacement_delta_l2_min"] = delta_l2 if previous_min is None else min(previous_min, delta_l2)
            diagnostics["hook_finite"] = bool(
                diagnostics["hook_finite"]
                and torch.isfinite(source).all().item()
                and torch.isfinite(latents).all().item()
                and torch.isfinite(replacement).all().item()
            )
            result = activation.clone()
            result[:, -1:, :] = replacement.to(dtype=activation.dtype)
            return result

        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        generated_text = ""
        generated_token_count = 0
        generation_status = "FAILED"
        error = None
        stop_reason = "FAILURE"
        record_started = time.perf_counter()
        try:
            with torch.inference_mode(), model.hooks(fwd_hooks=[(HOOK_NAME, steering_hook)]):
                generated = model.generate(
                    tokens,
                    max_new_tokens=GENERATION["max_new_tokens"],
                    temperature=GENERATION["temperature"],
                    top_p=GENERATION["top_p"],
                    do_sample=GENERATION["do_sample"],
                    prepend_bos=False,
                    verbose=False,
                )
            suffix = generated[0, input_token_count:]
            generated_token_count = int(suffix.shape[0])
            generated_text = tokenizer.decode(suffix, skip_special_tokens=True)
            last_token = int(suffix[-1].item()) if generated_token_count else None
            stop_reason = (
                "EOS_TOKEN" if last_token == tokenizer.eos_token_id
                else "MAX_NEW_TOKENS" if generated_token_count >= GENERATION["max_new_tokens"]
                else "MODEL_STOP_OTHER"
            )
            require(diagnostics["hook_call_count"] > 0, "Intervention hook was never called")
            require(diagnostics["hook_finite"], "Non-finite hook/SAE value")
            require(feature_delta_l2 > 0.0 and diagnostics["residual_replacement_delta_l2_max"] > 0.0, "Zero intervention delta")
            generation_status = "SUCCESS"
        except Exception as exc:
            error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            torch.cuda.empty_cache()

        stripped = generated_text.strip()
        is_empty = generated_text == ""
        is_whitespace = bool(generated_text) and not stripped
        record = {
            "schema_version": "phase21_model2_strength_generation_record_v1",
            "record_index": index,
            "condition_key": condition_key(alpha, source_cwe, int(item["prompt_id"])),
            "prompt_id": int(item["prompt_id"]),
            "source_index": int(item["source_index"]),
            "source_cwe": source_cwe,
            "language": item["language"],
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "sae_repository": SAE_REPOSITORY,
            "sae_revision": SAE_REVISION,
            "sae_id": SAE_ID,
            "sae_params_sha256": SAE_PARAMS_SHA256,
            "layer": LAYER,
            "feature_id": feature_id,
            "alpha": alpha,
            "seed": SEED,
            "seed_reset_immediately_before_generation": True,
            "generation_settings": GENERATION,
            "hook_name": HOOK_NAME,
            "hook_semantics": f"encode and replace only current activation[:, -1:, :]; add +{alpha} only to selected latent coordinate",
            "source_prompt_sha256": item["source_prompt_sha256"],
            "rendered_prompt_sha256": item["b0_rendered_prompt_sha256"],
            "input_token_count": input_token_count,
            "generated_text": generated_text,
            "generated_code": generated_text,
            "generated_text_sha256": sha256_text(generated_text),
            "generated_token_count": generated_token_count,
            "validity": {
                "is_valid": bool(stripped),
                "empty": is_empty,
                "whitespace_only": is_whitespace,
                "stripped_character_count": len(stripped),
            },
            "generation_status": generation_status,
            "generation_failure": generation_status != "SUCCESS",
            "generation_error": error,
            "stop_reason": stop_reason,
            "intervention_applied": generation_status == "SUCCESS" and diagnostics["hook_call_count"] > 0,
            "hook_call_count": diagnostics["hook_call_count"],
            "hook_finite": diagnostics["hook_finite"],
            "hook_failure": generation_status != "SUCCESS" and diagnostics["hook_call_count"] == 0,
            "selected_feature_decoded_delta_l2": feature_delta_l2,
            "residual_replacement_delta_l2_min": diagnostics["residual_replacement_delta_l2_min"],
            "residual_replacement_delta_l2_max": diagnostics["residual_replacement_delta_l2_max"],
            "fallback_status": "NONE",
            "runtime_seconds_diagnostic": time.perf_counter() - record_started,
            "population_manifest_sha256": EXPECTED_HASHES[POPULATION],
            "selected_routes_sha256": EXPECTED_HASHES[ROUTES],
            "strength_protocol_sha256": EXPECTED_HASHES[PROTOCOL],
            "runner_sha256": sha256_file(RUNNER),
        }
        validate_record(record, alpha, item)
        records.append(record)
        elapsed = prior_elapsed + time.perf_counter() - started
        if len(records) % CHECKPOINT_INTERVAL == 0 or len(records) == 180:
            atomic_json(checkpoint_path(alpha), {
                "schema_version": "phase21_model2_strength_generation_checkpoint_v1",
                "status": "IN_PROGRESS" if len(records) < 180 else "GENERATION_COMPLETE_PENDING_FINALIZATION",
                "alpha": alpha,
                "runner_sha256": sha256_file(RUNNER),
                "strength_protocol_sha256": EXPECTED_HASHES[PROTOCOL],
                "population_sha256": EXPECTED_HASHES[POPULATION],
                "completed_records": len(records),
                "records": records,
                "elapsed_seconds": elapsed,
            })
            failures = sum(record["generation_failure"] for record in records)
            invalid = sum(not record["validity"]["is_valid"] for record in records)
            print(f"ALPHA{alpha} {len(records)}/180 failures={failures} invalid={invalid} elapsed_h={elapsed/3600:.2f}", flush=True)

    require(len(records) == 180, f"Alpha{alpha} final record count mismatch")
    for record, item in zip(records, population):
        validate_record(record, alpha, item)
    require(len({record["condition_key"] for record in records}) == 180, f"Alpha{alpha} duplicate keys")
    atomic_json(output_path(alpha), records)
    output_hash = sha256_file(output_path(alpha))
    failures = sum(record["generation_failure"] for record in records)
    invalid = sum(not record["validity"]["is_valid"] for record in records)
    empty = sum(record["validity"]["empty"] for record in records)
    whitespace = sum(record["validity"]["whitespace_only"] for record in records)
    manifest = {
        "schema_version": "phase21_model2_strength_generation_manifest_v1",
        "status": "COMPLETE_WITH_PRESERVED_FAILURES" if failures else "COMPLETE",
        "alpha": alpha,
        "generation_output_path": f"revision/model2/phase21/outputs/model2_alpha{alpha}_dev_outputs.json",
        "generation_output_sha256": output_hash,
        "record_count": len(records),
        "unique_condition_key_count": len({record["condition_key"] for record in records}),
        "prompt_ids_source_order_sha256": canonical_sha256([record["prompt_id"] for record in records]),
        "source_cwe_counts": {cwe: sum(record["source_cwe"] == cwe for record in records) for cwe in FROZEN_ROUTES},
        "route_map_canonical_sha256": ROUTE_MAP_CANONICAL_SHA256,
        "selected_routes_sha256": EXPECTED_HASHES[ROUTES],
        "population_sha256": EXPECTED_HASHES[POPULATION],
        "strength_protocol_sha256": EXPECTED_HASHES[PROTOCOL],
        "runner_sha256": sha256_file(RUNNER),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "sae_id": SAE_ID,
        "sae_revision": SAE_REVISION,
        "sae_params_sha256": SAE_PARAMS_SHA256,
        "generation_settings": GENERATION | {"seed": SEED},
        "generation_failure_count": failures,
        "hook_failure_count": sum(record["hook_failure"] for record in records),
        "invalid_count": invalid,
        "empty_count": empty,
        "whitespace_only_count": whitespace,
        "unexpected_prompt_ids": [],
        "duplicate_keys": False,
        "other_alphas_tested": [],
        "b0_regenerated": False,
        "scanner_run": False,
        "heldout_accessed": False,
        "model1_modified": False,
    }
    atomic_json(manifest_path(alpha), manifest)
    atomic_json(checkpoint_path(alpha), {
        "schema_version": "phase21_model2_strength_generation_checkpoint_v1",
        "status": "COMPLETE",
        "alpha": alpha,
        "runner_sha256": sha256_file(RUNNER),
        "strength_protocol_sha256": EXPECTED_HASHES[PROTOCOL],
        "population_sha256": EXPECTED_HASHES[POPULATION],
        "completed_records": 180,
        "records": records,
        "elapsed_seconds": prior_elapsed + time.perf_counter() - started,
        "generation_output_sha256": output_hash,
        "generation_manifest_sha256": sha256_file(manifest_path(alpha)),
    })
    print(json.dumps({
        "alpha": alpha,
        "status": manifest["status"],
        "records": 180,
        "generation_sha256": output_hash,
        "manifest_sha256": sha256_file(manifest_path(alpha)),
        "failures": failures,
        "invalid": invalid,
        "empty": empty,
        "whitespace_only": whitespace,
    }, indent=2), flush=True)


def run(resume: bool) -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from huggingface_hub import hf_hub_download
    from transformer_lens import HookedTransformer
    from transformers import AutoTokenizer

    require(torch.cuda.is_available(), "CUDA is unavailable")
    _, population, _ = load_and_validate_frozen()
    require(not validate_complete_condition(40, population) or validate_complete_condition(20, population), "Alpha execution order violation")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.empty_cache()
    model = HookedTransformer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        tokenizer=tokenizer,
        device="cuda:0",
        n_devices=1,
        dtype=torch.bfloat16,
        fold_ln=True,
        center_writing_weights=False,
        center_unembed=False,
        move_to_device=True,
        local_files_only=True,
    ).eval()
    require(sorted({str(parameter.device) for parameter in model.parameters()}) == ["cuda:0"], "Unauthorized model offload/device")
    require(int(model.cfg.d_model) == 3584 and int(model.cfg.n_layers) == 42, "Model architecture mismatch")
    params_path = hf_hub_download(SAE_REPOSITORY, f"{SAE_ID}/params.npz", revision=SAE_REVISION, local_files_only=True)
    require(sha256_file(Path(params_path)) == SAE_PARAMS_SHA256, "SAE parameter hash mismatch")
    sae = load_exact_sae(params_path)
    require(sorted({str(parameter.device) for parameter in sae.parameters()}) == ["cuda:0"], "Unauthorized SAE offload/device")
    print(f"LOADED MODEL {MODEL_ID}@{MODEL_REVISION} AND SAE {SAE_ID}", flush=True)

    generate_condition(20, population, resume, model, tokenizer, sae)
    generate_condition(40, population, checkpoint_path(40).exists(), model, tokenizer, sae)

    require(validate_complete_condition(20, population), "Alpha20 did not finalize")
    require(validate_complete_condition(40, population), "Alpha40 did not finalize")
    del sae
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({
        "status": "BOTH_CONDITIONS_COMPLETE",
        "alpha20_sha256": sha256_file(output_path(20)),
        "alpha40_sha256": sha256_file(output_path(40)),
        "total_records": 360,
        "scanner_run": False,
        "b0_regenerated": False,
        "heldout_accessed": False,
        "model1_modified": False,
    }, indent=2), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = preflight(args.resume)
    print(json.dumps(report, indent=2), flush=True)
    if args.preflight_only:
        return 0
    run(args.resume)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"PHASE21_MODEL2_STRENGTH_GENERATION_ERROR: {type(exc).__name__}: {exc}", flush=True)
        raise
