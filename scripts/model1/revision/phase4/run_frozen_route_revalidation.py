"""Paired development-only causal revalidation for the four evaluable frozen B* routes.

This revision-only runner never selects features or changes original artifacts. It emits
paired unsteered/steered generations and checkpoints after every arm.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from sae_lens import SAE
from safetensors.torch import load_file as load_safetensors_file
from transformers import AutoModelForCausalLM, AutoTokenizer


os.environ["HF_HUB_OFFLINE"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[4]
PHASE_DIR = PROJECT_ROOT / "revision/model1/phase4"
FEASIBILITY_PATH = PHASE_DIR / "frozen_route_revalidation_feasibility.json"
DEV_PROMPTS_PATH = PROJECT_ROOT / "data/cyberseceval/dev_prompts.json"
B0_OUTPUTS_PATH = PROJECT_ROOT / "outputs/phase2/baseline_dev_outputs.json"
B0_ICD_PATH = PROJECT_ROOT / "outputs/phase2/baseline_dev_icd.json"
BSTAR_CONFIG_PATH = PROJECT_ROOT / "configs/bstar_config.json"
OUT_DIR = PHASE_DIR / "outputs"
FINAL_PATH = OUT_DIR / "frozen_route_revalidation_generations.json"
CHECKPOINT_PATH = OUT_DIR / "frozen_route_revalidation_generations.checkpoint.json"

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_SNAPSHOT = "0e9e39f249a16976918f6564b8830bc894c89659"
SAE_RELEASE = "llama_scope_lxr_8x"
SAE_SNAPSHOT = "8dbc1d85edfced43081c03c38b05514dbab1368b"
SEED = 42
TEMPERATURE = 0.2
TOP_P = 0.95
MAX_INPUT_TOKENS = 1024
MAX_NEW_TOKENS = 512
EXPECTED_PAIRS = 66
EXPECTED_RECORDS = 132


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_model_snapshot() -> Path:
    hf_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    snapshot = (
        hf_root
        / "hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots"
        / MODEL_SNAPSHOT
    )
    if not snapshot.is_dir():
        raise RuntimeError(f"Audited local model snapshot is missing: {snapshot}")
    return snapshot


def resolve_sae_snapshot() -> Path:
    hf_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    snapshot = (
        hf_root
        / "hub/models--fnlp--Llama3_1-8B-Base-LXR-8x/snapshots"
        / SAE_SNAPSHOT
    )
    if not snapshot.is_dir():
        raise RuntimeError(f"Audited local SAE snapshot is missing: {snapshot}")
    return snapshot


def make_local_llama_scope_converter(snapshot: Path):
    """Return the SAE Lens Llama-Scope conversion using only audited cached files."""

    def converter(
        repo_id: str,
        folder_name: str,
        device: str = "cpu",
        force_download: bool = False,
        cfg_overrides: dict[str, Any] | None = None,
    ):
        del repo_id, force_download
        layer_dir = snapshot / folder_name
        config_path = layer_dir / "hyperparams.json"
        weights_path = layer_dir / "checkpoints/final.safetensors"
        if not config_path.is_file() or not weights_path.is_file():
            raise RuntimeError(f"Incomplete local SAE directory: {layer_dir}")
        old_cfg = json.loads(config_path.read_text(encoding="utf-8"))
        d_in = int(old_cfg["d_model"])
        norm_scaling_factor = d_in**0.5 / float(
            old_cfg["dataset_average_activation_norm"]["in"]
        )
        cfg = {
            "architecture": "jumprelu",
            "jump_relu_threshold": float(old_cfg["jump_relu_threshold"]) * norm_scaling_factor,
            "d_in": d_in,
            "d_sae": int(old_cfg["d_sae"]),
            "dtype": "bfloat16",
            "model_name": "meta-llama/Llama-3.1-8B",
            "hook_name": old_cfg["hook_point_in"],
            "hook_head_index": None,
            "activation_fn": "relu",
            "finetuning_scaling_factor": False,
            "sae_lens_training_version": None,
            "prepend_bos": True,
            "dataset_path": "cerebras/SlimPajama-627B",
            "context_size": 1024,
            "dataset_trust_remote_code": True,
            "apply_b_dec_to_input": False,
            "normalize_activations": "expected_average_only_in",
            "device": device,
        }
        if cfg_overrides:
            cfg.update(cfg_overrides)
        requested_dtype = getattr(torch, str(cfg["dtype"]))
        loaded = load_safetensors_file(str(weights_path), device=device)
        loaded = {k: v.to(dtype=requested_dtype) for k, v in loaded.items()}
        state_dict = {
            "W_enc": loaded["encoder.weight"].T,
            "W_dec": loaded["decoder.weight"].T,
            "b_enc": loaded["encoder.bias"],
            "b_dec": loaded["decoder.bias"],
            "threshold": torch.ones(
                cfg["d_sae"], dtype=requested_dtype, device=device
            )
            * cfg["jump_relu_threshold"],
        }
        return cfg, state_dict, None

    return converter


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def record_key(rec: dict[str, Any]) -> str:
    return "|".join(
        str(rec[k])
        for k in ("cwe_id", "layer", "feature_id", "prompt_id", "population", "arm")
    )


def load_design() -> tuple[dict[str, Any], list[dict[str, Any]], list[int]]:
    feasibility = json.loads(FEASIBILITY_PATH.read_text(encoding="utf-8"))
    bstar = json.loads(BSTAR_CONFIG_PATH.read_text(encoding="utf-8"))
    routes = [r for r in feasibility["routes"] if r["evaluation_status"] == "EVALUABLE"]
    if len(routes) != 4:
        raise RuntimeError(f"Expected exactly 4 evaluable routes, found {len(routes)}")
    expected = {
        "CWE-120": (19, 14193),
        "CWE-327": (23, 14449),
        "CWE-89": (23, 1652),
        "CWE-338": (23, 7533),
    }
    if {r["cwe_id"] for r in routes} != set(expected):
        raise RuntimeError("Evaluable route set drift")
    for route in routes:
        cwe = route["cwe_id"]
        if (route["layer"], route["feature_id"]) != expected[cwe]:
            raise RuntimeError(f"Frozen route drift for {cwe}")
        cfg = bstar["feature_map"][cwe]
        if (cfg["layer"], cfg["feature"]) != expected[cwe]:
            raise RuntimeError(f"B* config mismatch for {cwe}")
        if float(route["alpha"]) != float(bstar["alpha"]) or float(route["alpha"]) != 40.0:
            raise RuntimeError(f"Frozen alpha mismatch for {cwe}")
    safe_ids = feasibility["corruption_population"]["fixed_sample_record_ids"]
    if safe_ids != [961, 404, 953, 1793, 916]:
        raise RuntimeError("Frozen safe-ID sample drift")
    return feasibility, routes, safe_ids


def build_tasks(routes: list[dict[str, Any]], safe_ids: list[int]) -> list[dict[str, Any]]:
    dev = json.loads(DEV_PROMPTS_PATH.read_text(encoding="utf-8"))
    b0_outputs = json.loads(B0_OUTPUTS_PATH.read_text(encoding="utf-8"))
    b0_icd = json.loads(B0_ICD_PATH.read_text(encoding="utf-8"))
    dev_map = {r["prompt_id"]: r for r in dev}
    out_map = {r["prompt_id"]: r for r in b0_outputs}
    icd_map = {r["prompt_id"]: r for r in b0_icd}
    if len(dev_map) != 1341 or len(out_map) != 1341 or len(icd_map) != 1341:
        raise RuntimeError("Development source coverage drift")

    tasks: list[dict[str, Any]] = []
    for route in routes:
        cwe = route["cwe_id"]
        unsafe_ids = route["eligible_record_ids"]
        for population, ids in (("unsafe", unsafe_ids), ("safe", safe_ids)):
            for prompt_id in ids:
                if prompt_id not in dev_map or prompt_id not in out_map or prompt_id not in icd_map:
                    raise RuntimeError(f"Missing development source ID {prompt_id}")
                prompt = dev_map[prompt_id]["test_case_prompt"]
                if out_map[prompt_id].get("prompt_text") != prompt:
                    raise RuntimeError(f"Prompt mismatch for ID {prompt_id}")
                icd = icd_map[prompt_id]
                if icd.get("skipped"):
                    raise RuntimeError(f"Scanner-skipped ID in frozen task: {prompt_id}")
                findings = icd.get("findings", [])
                if population == "unsafe" and not any(f.get("cwe_id") == cwe for f in findings):
                    raise RuntimeError(f"Missing historical target finding for {cwe}/{prompt_id}")
                if population == "safe" and findings:
                    raise RuntimeError(f"Historical finding in frozen safe ID {prompt_id}")
                tasks.append(
                    {
                        "cwe_id": cwe,
                        "layer": int(route["layer"]),
                        "feature_id": int(route["feature_id"]),
                        "alpha": float(route["alpha"]),
                        "prompt_id": prompt_id,
                        "population": population,
                        "prompt_text": prompt,
                        "prompt_sha256": sha256_text(prompt),
                        "language": str(dev_map[prompt_id].get("language", "")).lower(),
                        "source_cwe": dev_map[prompt_id].get("cwe_identifier", ""),
                        "historical_baseline_findings": findings,
                    }
                )
    if len(tasks) != EXPECTED_PAIRS:
        raise RuntimeError(f"Expected {EXPECTED_PAIRS} route/prompt pairs, found {len(tasks)}")
    return tasks


def make_bstar_hook(sae: SAE, feature_id: int, alpha: float, store: dict[str, Any]):
    """Submitted B* semantics: full-tensor SAE encode/add/decode on every hook call."""

    def hook_fn(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        with torch.no_grad():
            z = sae.encode(hidden)
            direction = torch.zeros_like(z)
            direction[..., feature_id] = 1.0
            z_steered = z + alpha * direction
            edited = sae.decode(z_steered).to(hidden.dtype)
        shape = list(hidden.shape)
        store["hook_calls"] += 1
        key = "x".join(str(x) for x in shape)
        store["hook_shape_counts"][key] = store["hook_shape_counts"].get(key, 0) + 1
        if store["finite_checked_calls"] == 0:
            store["input_finite"] = bool(torch.isfinite(hidden).all().item())
            store["latent_finite"] = bool(torch.isfinite(z_steered).all().item())
            store["output_finite"] = bool(torch.isfinite(edited).all().item())
            store["finite_checked_calls"] = 1
        if isinstance(output, tuple):
            return (edited,) + output[1:]
        return edited

    return hook_fn


def paired_seed() -> None:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def generate_arm(
    task: dict[str, Any],
    arm: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    saes: dict[int, SAE],
) -> dict[str, Any]:
    inputs = tokenizer(
        task["prompt_text"],
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    ).to(model.device)
    input_ids_before = inputs["input_ids"].detach().cpu().clone()
    store = {
        "hook_calls": 0,
        "hook_shape_counts": {},
        "input_finite": True,
        "latent_finite": True,
        "output_finite": True,
        "finite_checked_calls": 0,
    }
    hook = None
    hook_removed = arm == "baseline"
    exception = None
    generated_code = ""
    generated_token_count = 0
    started = time.time()
    try:
        if arm == "steered":
            module = model.model.layers[task["layer"]]
            hook = module.register_forward_hook(
                make_bstar_hook(saes[task["layer"]], task["feature_id"], task["alpha"], store)
            )
        paired_seed()
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = generated[0][inputs["input_ids"].shape[1] :]
        generated_token_count = int(new_tokens.numel())
        generated_code = tokenizer.decode(new_tokens, skip_special_tokens=True)
    except Exception as exc:  # preserve and continue so every requested pair is attempted
        exception = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        if hook is not None:
            hook.remove()
            hook_removed = True

    rec = {
        "cwe_id": task["cwe_id"],
        "layer": task["layer"],
        "feature_id": task["feature_id"],
        "route_alpha": task["alpha"],
        "applied_alpha": task["alpha"] if arm == "steered" else 0.0,
        "prompt_id": task["prompt_id"],
        "population": task["population"],
        "arm": arm,
        "prompt_text": task["prompt_text"],
        "prompt_sha256": task["prompt_sha256"],
        "language": task["language"],
        "source_cwe": task["source_cwe"],
        "historical_baseline_findings": task["historical_baseline_findings"],
        "generated_code": generated_code,
        "generated_code_sha256": sha256_text(generated_code),
        "generated_token_count": generated_token_count,
        "input_token_count": int(inputs["input_ids"].shape[1]),
        "input_ids_preserved": bool(torch.equal(input_ids_before, inputs["input_ids"].detach().cpu())),
        "valid_nonempty_generation": bool(generated_code.strip()) and generated_token_count > 0,
        "seed": SEED,
        "generation": {
            "max_input_tokens": MAX_INPUT_TOKENS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "do_sample": True,
            "raw_prompt_no_chat_template": True,
        },
        "hook": {
            "semantics": "full_tensor_sae_encode_add_decode_every_hook_call",
            **store,
            "removed": hook_removed,
        },
        "exception": exception,
        "elapsed_seconds": round(time.time() - started, 6),
    }
    if arm == "baseline" and store["hook_calls"] != 0:
        rec["exception"] = {"type": "BaselineHookError", "message": "Baseline unexpectedly fired a hook"}
    if arm == "steered" and store["hook_calls"] == 0:
        rec["exception"] = {"type": "SteeringHookError", "message": "Steered arm fired zero hooks"}
    return rec


def build_metadata(
    source_hashes: dict[str, str], script_hash: str, model_snapshot_path: Path,
    sae_snapshot_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "phase": 4,
        "purpose": "four_route_frozen_bstar_causal_revalidation_generation",
        "scope": "development_only_four_evaluable_routes",
        "model_id": MODEL_ID,
        "model_snapshot": MODEL_SNAPSHOT,
        "model_snapshot_path": str(model_snapshot_path),
        "model_dtype": "torch.float16",
        "sae_release": SAE_RELEASE,
        "sae_snapshot": SAE_SNAPSHOT,
        "sae_snapshot_path": str(sae_snapshot_path),
        "seed": SEED,
        "paired_seed_reset_before_each_arm": True,
        "expected_pairs": EXPECTED_PAIRS,
        "expected_records": EXPECTED_RECORDS,
        "script_path": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "script_sha256": script_hash,
        "source_hashes": source_hashes,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "heldout_used": False,
        "original_artifacts_modified": False,
    }


def validate_records(records: list[dict[str, Any]], tasks: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    keys = [record_key(r) for r in records]
    if len(keys) != len(set(keys)):
        errors.append("duplicate record keys")
    if len(records) != EXPECTED_RECORDS:
        errors.append(f"expected {EXPECTED_RECORDS} records, found {len(records)}")
    expected_keys = set()
    for task in tasks:
        for arm in ("baseline", "steered"):
            expected_keys.add(
                "|".join(
                    str(x)
                    for x in (
                        task["cwe_id"], task["layer"], task["feature_id"], task["prompt_id"],
                        task["population"], arm,
                    )
                )
            )
    if set(keys) != expected_keys:
        errors.append("record-key set does not match frozen task design")
    for rec in records:
        if rec.get("exception") is not None:
            errors.append(f"generation failure: {record_key(rec)}")
        if not rec.get("input_ids_preserved"):
            errors.append(f"input IDs changed: {record_key(rec)}")
        if rec["arm"] == "steered" and rec["hook"]["hook_calls"] <= 0:
            errors.append(f"zero hook calls: {record_key(rec)}")
        if rec["arm"] == "baseline" and rec["hook"]["hook_calls"] != 0:
            errors.append(f"baseline hook fired: {record_key(rec)}")
    return errors


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feasibility, routes, safe_ids = load_design()
    tasks = build_tasks(routes, safe_ids)
    source_paths = [
        FEASIBILITY_PATH,
        DEV_PROMPTS_PATH,
        B0_OUTPUTS_PATH,
        B0_ICD_PATH,
        BSTAR_CONFIG_PATH,
        PROJECT_ROOT / "phases/phase5/run_thea_static.py",
    ]
    source_hashes = {
        str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256_file(p) for p in source_paths
    }
    script_hash = sha256_file(Path(__file__))
    model_snapshot_path = resolve_model_snapshot()
    sae_snapshot_path = resolve_sae_snapshot()
    for layer in (19, 23):
        folder = sae_snapshot_path / f"Llama3_1-8B-Base-L{layer}R-8x"
        for relative in ("hyperparams.json", "checkpoints/final.safetensors"):
            path = folder / relative
            source_hashes[f"sae_cache/L{layer}/{relative}"] = sha256_file(path)
    metadata = build_metadata(
        source_hashes, script_hash, model_snapshot_path, sae_snapshot_path
    )

    records: list[dict[str, Any]] = []
    if CHECKPOINT_PATH.exists():
        checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        if checkpoint.get("metadata", {}).get("script_sha256") != script_hash:
            raise RuntimeError("Checkpoint script hash differs from current runner; refusing unsafe resume")
        if checkpoint.get("metadata", {}).get("source_hashes") != source_hashes:
            raise RuntimeError("Checkpoint source hashes differ; refusing unsafe resume")
        records = checkpoint.get("records", [])
        print(f"Resuming from {len(records)}/{EXPECTED_RECORDS} completed records")
    completed = {record_key(r) for r in records}

    print("Loading tokenizer/model from local cache...")
    tokenizer = AutoTokenizer.from_pretrained(model_snapshot_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_snapshot_path,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )
    model.eval()
    required_layers = sorted({r["layer"] for r in routes})
    saes: dict[int, SAE] = {}
    local_converter = make_local_llama_scope_converter(sae_snapshot_path)
    for layer in required_layers:
        print(f"Loading SAE layer {layer}...")
        sae, _, _ = SAE.from_pretrained(
            release=SAE_RELEASE,
            sae_id=f"l{layer}r_8x",
            converter=local_converter,
        )
        saes[layer] = sae.to(model.device).eval()

    total_start = time.time()
    for pair_index, task in enumerate(tasks, start=1):
        for arm in ("baseline", "steered"):
            prospective = {
                "cwe_id": task["cwe_id"], "layer": task["layer"],
                "feature_id": task["feature_id"], "prompt_id": task["prompt_id"],
                "population": task["population"], "arm": arm,
            }
            key = record_key(prospective)
            if key in completed:
                continue
            print(
                f"[{len(records)+1}/{EXPECTED_RECORDS}] {task['cwe_id']} "
                f"L{task['layer']} F{task['feature_id']} {task['population']} "
                f"prompt={task['prompt_id']} arm={arm}",
                flush=True,
            )
            rec = generate_arm(task, arm, model, tokenizer, saes)
            records.append(rec)
            completed.add(key)
            checkpoint_payload = {
                "metadata": metadata,
                "status": "IN_PROGRESS",
                "completed_records": len(records),
                "records": records,
            }
            atomic_json(CHECKPOINT_PATH, checkpoint_payload)
        if pair_index % 5 == 0:
            print(
                f"Completed {pair_index}/{EXPECTED_PAIRS} pairs in "
                f"{time.time()-total_start:.1f}s",
                flush=True,
            )

    errors = validate_records(records, tasks)
    payload = {
        "metadata": metadata,
        "status": "PASS" if not errors else "FAIL",
        "completed_records": len(records),
        "validation_errors": errors,
        "runtime_seconds": round(time.time() - total_start, 6),
        "records": records,
    }
    atomic_json(FINAL_PATH, payload)
    atomic_json(CHECKPOINT_PATH, payload)
    print(f"Saved {len(records)} records to {FINAL_PATH}")
    print(f"Generation validation status: {payload['status']}")
    if errors:
        for err in errors[:20]:
            print(f"  ERROR: {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
