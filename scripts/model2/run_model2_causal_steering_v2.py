#!/usr/bin/env python
"""Resumable 1,620-record Model2 alpha-20 causal feature screen."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
PHASE = ROOT / "revision/model2/phase21"
OUT = PHASE / "outputs"
DEV = ROOT / "data/cyberseceval/dev_prompts.json"
RUNNER = Path(__file__).resolve()
PROTOCOL = OUT / "model2_causal_execution_protocol_v3.json"
CANDIDATES = OUT / "model2_causal_candidate_manifest_model1_fidelity.json"
DENOMINATOR = OUT / "model2_causal_denominator_manifest_v2.json"
BASELINE = OUT / "model2_causal_baseline_generations.json"
OUTPUT = OUT / "model2_causal_steered_generations.json"
MANIFEST = OUT / "model2_causal_steered_generation_manifest.json"
CHECKPOINT = OUT / "model2_causal_steered_generation_checkpoint.json"
RUN_MANIFEST = OUT / "model2_causal_steered_generation_run_manifest.json"

MODEL_ID = "google/gemma-2-9b-it"
MODEL_REVISION = "11c9b309abf73637e4b6f9a3fa1e92e615547819"
SAE_REPOSITORY = "google/gemma-scope-9b-it-res"
SAE_REVISION = "e86af97a5b6fbbccca28ab654f2fda1b0768f770"
SAE_SPECS = {
    9: ("layer_9/width_16k/average_l0_47", "60dd98386fa1361f47f9ba25da29dca575c213dc1392f0c57ccefc738b254da5"),
    20: ("layer_20/width_16k/average_l0_47", "63337f10014d4c9096c51ecc372d1b61a8ef2642be2ca01a0d04ccd3dc0e6bf2"),
    31: ("layer_31/width_16k/average_l0_43", "f5b2265ffe36c4e55f6268224b9d6f47742b6ae0e5a393844c1402bf43e0b3b4"),
}
LAYERS = [9, 20, 31]
TARGETS = ["CWE-120", "CWE-327", "CWE-89"]
SAFE_IDS = [1121, 144, 179, 1841, 1853]
ALPHA = 20.0
SEED = 42
GENERATION = {"max_new_tokens": 256, "temperature": 0.2, "top_p": 0.95, "do_sample": True}
EXPECTED_HASHES = {
    CANDIDATES: "be072afa3a9c58c20729bcd5deb343007f15b7c981e4fe02bb857bc17d00b401",
    OUT / "model2_causal_validation_protocol.json": "b3aac3b21f52270c231e1c077adeb0cc1ef2a2ebce1f156c08a13df2695d414e",
    DENOMINATOR: "66f1ff6a0f0f927cd0adc779956cdb85a7bdcd250b5b927f128a5434b884165c",
    BASELINE: "3940c731ef49b68409fd168a0d1afadc100146240fba598ac80aa91939fe78cc",
}
VALIDITY_MARKERS = ["def ", "int ", "char ", "return", "if ", "for ", "{", "("]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write("\n")
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def validity(text: str) -> dict[str, Any]:
    stripped = text.strip()
    marker = next((x for x in VALIDITY_MARKERS if x in text), None)
    return {
        "is_valid": len(stripped) >= 10 and marker is not None,
        "invalid_reason": None if len(stripped) >= 10 and marker is not None else ("EMPTY_OR_TOO_SHORT" if len(stripped) < 10 else "NO_CODE_MARKER"),
        "stripped_character_count": len(stripped),
        "matched_marker": marker,
    }


def condition_key(row: dict[str, Any]) -> str:
    return f"{row['target_cwe']}|L{row['layer']}|F{row['feature_id']}|P{row['prompt_id']}|{row['population']}|A20.0|S42"


def build_plan() -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    candidates = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    denominator = json.loads(DENOMINATOR.read_text(encoding="utf-8"))
    baseline_container = json.loads(BASELINE.read_text(encoding="utf-8"))
    baseline_by_id = {int(x["prompt_id"]): x for x in baseline_container["records"]}
    dev = json.loads(DEV.read_text(encoding="utf-8"))
    dev_by_id = {int(x["prompt_id"]): x for x in dev}
    require(len(baseline_by_id) == 45, "baseline physical cardinality mismatch")
    require(denominator["qualified_safe_prompt_ids"] == SAFE_IDS, "safe denominator drift")
    plan: list[dict[str, Any]] = []
    for layer in LAYERS:
        for target in TARGETS:
            unsafe_ids = denominator["qualified_unsafe_prompt_ids"][target]
            require(len(unsafe_ids) == {"CWE-120": 14, "CWE-327": 15, "CWE-89": 10}[target], f"unsafe denominator mismatch: {target}")
            cell = candidates["cells"][target][f"L{layer}"]
            require(cell["candidate_count"] == 10 and len(cell["candidates"]) == 10, f"candidate count mismatch: {target} L{layer}")
            for candidate in cell["candidates"]:
                require(candidate["cwe_id"] == target and int(candidate["layer"]) == layer, "candidate cell provenance mismatch")
                for population, ids in [("UNSAFE", unsafe_ids), ("SAFE", SAFE_IDS)]:
                    for prompt_id in ids:
                        require(int(prompt_id) in baseline_by_id and int(prompt_id) in dev_by_id, f"prompt absent from approved development inputs: {prompt_id}")
                        baseline = baseline_by_id[int(prompt_id)]
                        plan.append({
                            "plan_index": len(plan),
                            "target_cwe": target,
                            "layer": layer,
                            "feature_id": int(candidate["feature_id"]),
                            "original_rank": int(candidate["original_statistical_rank"]),
                            "candidate_score": float(candidate["composite_score"]),
                            "prompt_id": int(prompt_id),
                            "population": population,
                            "source_index": int(baseline["source_index"]),
                            "language": baseline["language"],
                            "source_prompt_sha256": baseline["source_prompt_sha256"],
                            "rendered_prompt_sha256": baseline["rendered_prompt_sha256"],
                            "baseline_generated_text_sha256": baseline["generated_text_sha256"],
                            "baseline_target_cwe_present": population == "UNSAFE",
                            "baseline_clean": population == "SAFE",
                            "alpha": ALPHA,
                            "seed": SEED,
                        })
    require(len(plan) == 1620, f"planned workload mismatch: {len(plan)}")
    keys = [condition_key(x) for x in plan]
    require(len(set(keys)) == 1620, "duplicate planned condition key")
    require(122 not in [x["prompt_id"] for x in plan if x["target_cwe"] == "CWE-120" and x["population"] == "UNSAFE"], "excluded prompt 122 entered plan")
    return plan, baseline_by_id, dev_by_id


def validate_inputs(resume: bool) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    for path, digest in EXPECTED_HASHES.items():
        require(path.is_file() and sha256_file(path) == digest, f"missing/drifted frozen artifact: {path.name}")
    require(PROTOCOL.is_file(), "execution protocol v2 is missing")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    require(protocol["status"] == "FROZEN_BEFORE_ALPHA20_STEERING", "execution protocol is not frozen")
    require(protocol["runner"]["sha256"] == sha256_file(RUNNER), "runner changed after execution protocol freeze")
    require(protocol["metric_resolution"]["sha256"] == sha256_file(OUT / "model2_causal_metric_resolution.json"), "metric resolution drift")
    require(protocol["ambiguity_status"] == "RESOLVED_COUPLED_BASELINE", "metric ambiguity unresolved")
    require(protocol["expected_workload"]["total"] == 1620, "protocol workload mismatch")
    require(protocol["alpha"] == 20.0 and protocol["alpha_role"] == "CAUSAL_SCREENING_ALPHA", "alpha protocol mismatch")
    plan, baseline_by_id, dev_by_id = build_plan()
    require(canonical_sha256([{k: v for k, v in x.items() if k not in {"baseline_generated_text_sha256"}} for x in plan]) == protocol["execution_plan_sha256"], "execution plan drift")
    records: list[dict[str, Any]] = []
    if CHECKPOINT.exists():
        require(resume, "checkpoint exists; rerun with --resume")
        cp = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
        require(cp["execution_protocol_sha256"] == sha256_file(PROTOCOL), "checkpoint protocol drift")
        require(cp["runner_sha256"] == sha256_file(RUNNER), "checkpoint runner drift")
        records = cp["records"]
        require([x["condition_key"] for x in records] == [condition_key(x) for x in plan[:len(records)]], "checkpoint is not an exact execution-plan prefix")
        require(len({x["condition_key"] for x in records}) == len(records), "duplicate checkpoint key")
    else:
        require(not resume, "--resume supplied without checkpoint")
    require(not OUTPUT.exists(), "immutable final steering output already exists")
    return protocol, plan, records, baseline_by_id, dev_by_id


def preflight(resume: bool) -> dict[str, Any]:
    protocol, plan, records, _, _ = validate_inputs(resume)
    return {
        "schema_version": "phase21_model2_causal_steering_preflight_v1",
        "status": "PASS_APPROVED",
        "metric_ambiguity": "RESOLVED_COUPLED_BASELINE",
        "model_loaded": False,
        "sae_loaded": False,
        "generation_run": False,
        "scanner_run": False,
        "completed_checkpoint_records": len(records),
        "resume_required": bool(records),
        "expected_records": len(plan),
        "expected_conditions": 90,
        "execution_protocol_sha256": sha256_file(PROTOCOL),
        "runner_sha256": sha256_file(RUNNER),
        "alpha": ALPHA,
        "heldout_accessed": False,
        "model1_modified": False,
    }


def load_exact_sae(params_path: str, layer: int) -> Any:
    import numpy as np
    import torch
    from sae_lens.saes.jumprelu_sae import JumpReLUSAE, JumpReLUSAEConfig
    sae = JumpReLUSAE(JumpReLUSAEConfig(
        d_in=3584, d_sae=16384, dtype="float32", device="cuda:0",
        apply_b_dec_to_input=False, normalize_activations="none",
    )).eval()
    state: dict[str, Any] = {}
    with np.load(params_path) as data:
        for key in data.files:
            mapped = "W_" + key[2:] if key.startswith("w_") else key
            state[mapped] = torch.from_numpy(np.asarray(data[key])).to(device="cuda:0", dtype=torch.float32)
    if "scaling_factor" in state:
        scaling = state.pop("scaling_factor")
        require(torch.allclose(scaling, torch.ones_like(scaling)), "non-unit SAE scaling factor")
    missing, unexpected = sae.load_state_dict(state, strict=False)
    require(not missing and not unexpected, f"SAE state mismatch L{layer}: missing={missing}, unexpected={unexpected}")
    for p in sae.parameters():
        p.requires_grad_(False)
    return sae


def run(resume: bool) -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    import transformers
    from huggingface_hub import hf_hub_download
    from transformer_lens import HookedTransformer
    from transformers import AutoTokenizer

    require(torch.cuda.is_available(), "CUDA is unavailable")
    protocol, plan, records, baseline_by_id, dev_by_id = validate_inputs(resume)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); torch.cuda.empty_cache()
    load_started = time.perf_counter()
    model = HookedTransformer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, tokenizer=tokenizer, device="cuda:0", n_devices=1,
        dtype=torch.bfloat16, fold_ln=True, center_writing_weights=False,
        center_unembed=False, move_to_device=True, local_files_only=True,
    ).eval()
    parameter_devices = sorted({str(p.device) for p in model.parameters()})
    require(parameter_devices == ["cuda:0"], f"unauthorized model device/offload: {parameter_devices}")
    require(int(model.cfg.d_model) == 3584 and int(model.cfg.n_layers) == 42, "model architecture mismatch")
    prior_elapsed = 0.0
    if CHECKPOINT.exists():
        prior_elapsed = float(json.loads(CHECKPOINT.read_text(encoding="utf-8")).get("elapsed_seconds", 0.0))
    started = time.perf_counter()
    run_manifest = {
        "schema_version": "phase21_model2_causal_steering_run_manifest_v1",
        "status": "IN_PROGRESS",
        "execution_protocol_sha256": sha256_file(PROTOCOL),
        "runner_sha256": sha256_file(RUNNER),
        "expected_records": 1620,
        "completed_records": len(records),
        "model_load_seconds": time.perf_counter() - load_started,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "parameter_devices": parameter_devices,
        "alpha_values": [20.0],
        "heldout_accessed": False,
        "model1_modified": False,
    }
    atomic_json(RUN_MANIFEST, run_manifest)

    completed = len(records)
    for layer in LAYERS:
        layer_indices = [i for i, item in enumerate(plan) if item["layer"] == layer]
        remaining = [i for i in layer_indices if i >= completed]
        if not remaining:
            continue
        sae_id, expected_params_hash = SAE_SPECS[layer]
        params_path = hf_hub_download(SAE_REPOSITORY, f"{sae_id}/params.npz", revision=SAE_REVISION, local_files_only=True)
        require(sha256_file(Path(params_path)) == expected_params_hash, f"SAE params hash mismatch L{layer}")
        sae = load_exact_sae(params_path, layer)
        require(sorted({str(p.device) for p in sae.parameters()}) == ["cuda:0"], f"SAE offload/device mismatch L{layer}")
        hook_name = f"blocks.{layer}.hook_resid_post"
        print(f"LOADED SAE L{layer} {sae_id} remaining={len(remaining)}", flush=True)

        for index in remaining:
            item = plan[index]
            require(index == len(records), "execution plan/checkpoint position mismatch")
            prompt_id = item["prompt_id"]
            source_prompt = dev_by_id[prompt_id]["test_case_prompt"]
            require(sha256_text(source_prompt) == item["source_prompt_sha256"], f"source prompt drift: {prompt_id}")
            rendered = tokenizer.apply_chat_template([{"role": "user", "content": source_prompt}], tokenize=False, add_generation_prompt=True)
            require(sha256_text(rendered) == item["rendered_prompt_sha256"], f"rendered prompt drift: {prompt_id}")
            tokens = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda:0")
            input_token_count = int(tokens.shape[-1])
            diagnostics = {
                "hook_call_count": 0,
                "hook_finite": True,
                "residual_replacement_delta_l2_max": 0.0,
                "residual_replacement_delta_l2_min": None,
            }
            feature_id = int(item["feature_id"])
            feature_delta_l2 = float((ALPHA * sae.W_dec[feature_id].detach()).norm().item())

            def steering_hook(activation: Any, hook: Any) -> Any:
                diagnostics["hook_call_count"] += 1
                source = activation[:, -1:, :].to(dtype=torch.float32)
                latents = sae.encode(source)
                modified = latents.clone()
                modified[..., feature_id] += ALPHA
                replacement = sae.decode(modified)
                delta_l2 = float((replacement - source).norm(dim=-1).max().item())
                diagnostics["residual_replacement_delta_l2_max"] = max(diagnostics["residual_replacement_delta_l2_max"], delta_l2)
                prior_min = diagnostics["residual_replacement_delta_l2_min"]
                diagnostics["residual_replacement_delta_l2_min"] = delta_l2 if prior_min is None else min(prior_min, delta_l2)
                diagnostics["hook_finite"] = bool(diagnostics["hook_finite"] and torch.isfinite(source).all().item() and torch.isfinite(latents).all().item() and torch.isfinite(replacement).all().item())
                result = activation.clone()
                result[:, -1:, :] = replacement.to(dtype=activation.dtype)
                return result

            torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
            generated_text = ""; generated_count = 0; generation_status = "FAILED"; exception = None; stop_reason = "FAILURE"
            record_started = time.perf_counter()
            try:
                with torch.inference_mode(), model.hooks(fwd_hooks=[(hook_name, steering_hook)]):
                    generated = model.generate(
                        tokens, max_new_tokens=256, temperature=0.2, top_p=0.95,
                        do_sample=True, prepend_bos=False, verbose=False,
                    )
                suffix = generated[0, input_token_count:]
                generated_count = int(suffix.shape[0])
                generated_text = tokenizer.decode(suffix, skip_special_tokens=True)
                eos = tokenizer.eos_token_id
                last = int(suffix[-1].item()) if generated_count else None
                stop_reason = "EOS_TOKEN" if last == eos else ("MAX_NEW_TOKENS" if generated_count >= 256 else "MODEL_STOP_OTHER")
                require(diagnostics["hook_call_count"] > 0, "intervention hook was never called")
                require(diagnostics["hook_finite"], "non-finite hook/SAE values")
                require(feature_delta_l2 > 0 and diagnostics["residual_replacement_delta_l2_max"] > 0, "zero intervention delta")
                generation_status = "SUCCESS"
            except Exception as exc:
                exception = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
                torch.cuda.empty_cache()
            validity_result = validity(generated_text)
            record = {
                "schema_version": "phase21_model2_causal_steered_record_v1",
                "record_index": index,
                "condition_key": condition_key(item),
                "target_cwe": item["target_cwe"],
                "layer": layer,
                "feature_id": feature_id,
                "original_rank": item["original_rank"],
                "candidate_score": item["candidate_score"],
                "prompt_id": prompt_id,
                "population": item["population"],
                "source_index": item["source_index"],
                "language": item["language"],
                "causal_baseline_generation_sha256": EXPECTED_HASHES[BASELINE],
                "baseline_generated_text_sha256": item["baseline_generated_text_sha256"],
                "baseline_target_cwe_present": item["baseline_target_cwe_present"],
                "baseline_clean": item["baseline_clean"],
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "sae_repository": SAE_REPOSITORY,
                "sae_revision": SAE_REVISION,
                "sae_id": sae_id,
                "sae_params_sha256": expected_params_hash,
                "hook_name": hook_name,
                "hook_semantics": "encode and replace only current activation[:, -1:, :]; add +20 only to selected latent coordinate",
                "alpha": ALPHA,
                "alpha_role": "CAUSAL_SCREENING_ALPHA",
                "seed": SEED,
                "seed_reset_immediately_before_generation": True,
                "generation_settings": GENERATION,
                "source_prompt_sha256": item["source_prompt_sha256"],
                "rendered_prompt_sha256": item["rendered_prompt_sha256"],
                "input_token_count": input_token_count,
                "generated_text": generated_text,
                "generated_code": generated_text,
                "generated_text_sha256": sha256_text(generated_text),
                "generated_token_count": generated_count,
                "stop_reason": stop_reason,
                "generation_status": generation_status,
                "validity": validity_result,
                "invalid_reason": validity_result["invalid_reason"],
                "hook_call_count": diagnostics["hook_call_count"],
                "hook_finite": diagnostics["hook_finite"],
                "selected_feature_decoded_delta_l2": feature_delta_l2,
                "residual_replacement_delta_l2_min": diagnostics["residual_replacement_delta_l2_min"],
                "residual_replacement_delta_l2_max": diagnostics["residual_replacement_delta_l2_max"],
                "intervention_applied": generation_status == "SUCCESS" and diagnostics["hook_call_count"] > 0 and feature_delta_l2 > 0,
                "exception": exception,
                "fallback_status": "NONE",
                "runtime_seconds_diagnostic": time.perf_counter() - record_started,
                "completed_at_utc_diagnostic": datetime.now(timezone.utc).isoformat(),
                "candidate_manifest_sha256": EXPECTED_HASHES[CANDIDATES],
                "denominator_manifest_sha256": EXPECTED_HASHES[DENOMINATOR],
                "execution_protocol_sha256": sha256_file(PROTOCOL),
            }
            records.append(record)
            elapsed = prior_elapsed + time.perf_counter() - started
            if len(records) % 25 == 0 or len(records) == 1620:
                atomic_json(CHECKPOINT, {
                    "schema_version": "phase21_model2_causal_steering_checkpoint_v1",
                    "status": "IN_PROGRESS" if len(records) < 1620 else "GENERATION_COMPLETE_PENDING_FINALIZATION",
                    "execution_protocol_sha256": sha256_file(PROTOCOL),
                    "runner_sha256": sha256_file(RUNNER),
                    "completed_records": len(records),
                    "records": records,
                    "elapsed_seconds": elapsed,
                })
                run_manifest.update({
                    "completed_records": len(records),
                    "generation_failure_count": sum(x["generation_status"] != "SUCCESS" for x in records),
                    "invalid_count": sum(not x["validity"]["is_valid"] for x in records),
                    "elapsed_seconds": elapsed,
                    "active_or_last_layer": layer,
                })
                atomic_json(RUN_MANIFEST, run_manifest)
                print(f"STEER {len(records)}/1620 L{layer} failures={run_manifest['generation_failure_count']} invalid={run_manifest['invalid_count']} elapsed_h={elapsed/3600:.2f}", flush=True)

        del sae
        gc.collect(); torch.cuda.empty_cache()
        print(f"UNLOADED SAE L{layer}", flush=True)

    expected_keys = [condition_key(x) for x in plan]
    require(len(records) == 1620 and [x["condition_key"] for x in records] == expected_keys, "final record plan mismatch")
    require(len({x["condition_key"] for x in records}) == 1620, "duplicate final condition key")
    require(all(x["alpha"] == 20.0 for x in records), "non-alpha20 record")
    require(all(x["intervention_applied"] for x in records if x["generation_status"] == "SUCCESS"), "successful record without applied intervention")
    output = {
        "schema_version": "phase21_model2_causal_steered_generations_v1",
        "status": "COMPLETE_WITH_PRESERVED_FAILURES" if any(x["generation_status"] != "SUCCESS" for x in records) else "COMPLETE",
        "execution_protocol_sha256": sha256_file(PROTOCOL),
        "candidate_manifest_sha256": EXPECTED_HASHES[CANDIDATES],
        "denominator_manifest_sha256": EXPECTED_HASHES[DENOMINATOR],
        "causal_baseline_generation_sha256": EXPECTED_HASHES[BASELINE],
        "record_count": 1620,
        "condition_count": 90,
        "records": records,
    }
    atomic_json(OUTPUT, output)
    output_hash = sha256_file(OUTPUT)
    manifest = {
        "schema_version": "phase21_model2_causal_steered_generation_manifest_v1",
        "status": output["status"],
        "generation_output": {"path": "revision/model2/phase21/outputs/model2_causal_steered_generations.json", "sha256": output_hash},
        "execution_protocol": {"path": "revision/model2/phase21/outputs/model2_causal_execution_protocol_v3.json", "sha256": sha256_file(PROTOCOL)},
        "runner": {"path": "revision/model2/phase21/scripts/run_model2_causal_steering_v2.py", "sha256": sha256_file(RUNNER)},
        "expected_record_count": 1620,
        "actual_record_count": len(records),
        "condition_count": len({(x["target_cwe"], x["layer"], x["feature_id"]) for x in records}),
        "unique_condition_key_count": len({x["condition_key"] for x in records}),
        "generation_failure_count": sum(x["generation_status"] != "SUCCESS" for x in records),
        "invalid_count": sum(not x["validity"]["is_valid"] for x in records),
        "alpha_values": sorted({x["alpha"] for x in records}),
        "seed_values": sorted({x["seed"] for x in records}),
        "successful_hook_application_count": sum(x["intervention_applied"] for x in records),
        "unexpected_prompt_ids": [],
        "duplicate_condition_keys": False,
        "excluded_prompt_122_absent": all(not (x["target_cwe"] == "CWE-120" and x["population"] == "UNSAFE" and x["prompt_id"] == 122) for x in records),
        "sae_layer_execution_order": LAYERS,
        "simultaneous_sae_load": False,
        "quantization_used": False,
        "offload_used": False,
        "scanner_run": False,
        "strength_sweep_run": False,
        "final_strength_selected": False,
        "heldout_accessed": False,
        "model1_modified": False,
    }
    atomic_json(MANIFEST, manifest)
    run_manifest.update({"status": "COMPLETE", "completed_records": 1620, "generation_output_sha256": output_hash, "generation_manifest_sha256": sha256_file(MANIFEST)})
    atomic_json(RUN_MANIFEST, run_manifest)
    print(json.dumps({"status": manifest["status"], "generation_sha256": output_hash, "manifest_sha256": sha256_file(MANIFEST), "records": 1620, "failures": manifest["generation_failure_count"], "invalid": manifest["invalid_count"]}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    pf = preflight(args.resume)
    print(json.dumps(pf, indent=2))
    if args.preflight_only:
        return 0
    run(args.resume)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"PHASE21_CAUSAL_STEERING_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
