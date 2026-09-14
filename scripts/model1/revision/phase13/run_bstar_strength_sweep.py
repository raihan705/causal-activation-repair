#!/usr/bin/env python3
"""Run one frozen Phase 13 B* strength condition with fail-closed resume.

The --preflight-only path intentionally uses only the Python standard library
and does not import Torch, Transformers, or SAELens.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
ACTIVE_OUTPUT_DIR = ROOT / "revision/model1/phase13/outputs"
ACTIVE_MANIFEST = ACTIVE_OUTPUT_DIR / "strength_subset_manifest.json"
HASH_RECORD = ACTIVE_OUTPUT_DIR / "dev_subset_hashes.json"
DENOMINATOR_AUDIT = ROOT / "revision/model1/phase2/phase2_denominator_audit.json"
BSTAR_CONFIG = ROOT / "configs/bstar_config.json"
FINAL_CONFIG = ROOT / "configs/final_config.yaml"
PHASE5_RUNNER = ROOT / "phases/phase5/run_thea_static.py"
PHASE9_RUNNER = ROOT / "phases/phase9/run_phase9_generation.py"

ALLOWED_ALPHAS = (10, 20, 30, 40, 50, 60, 80)
REQUIRED_SEED = 42
EXPECTED_RECORD_COUNT = 180
CHECKPOINT_INTERVAL = 50
FROZEN_SAFE_CWES = {"CWE-120", "CWE-327", "CWE-89", "CWE-338"}

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_CACHE_SNAPSHOT = "0e9e39f249a16976918f6564b8830bc894c89659"
MODEL_CACHE_REPO = "models--meta-llama--Meta-Llama-3.1-8B-Instruct"
SAE_RELEASE = "llama_scope_lxr_8x"
SAE_IDS = ("l16r_8x", "l19r_8x", "l23r_8x")
SAE_CACHE_SNAPSHOT = "8dbc1d85edfced43081c03c38b05514dbab1368b"
SAE_CACHE_REPO = "models--fnlp--Llama3_1-8B-Base-LXR-8x"
SAE_CACHE_DIRS = {
    "l16r_8x": "Llama3_1-8B-Base-L16R-8x",
    "l19r_8x": "Llama3_1-8B-Base-L19R-8x",
    "l23r_8x": "Llama3_1-8B-Base-L23R-8x",
}

GENERATION_SETTINGS = {
    "dtype": "float16",
    "raw_prompt_rendering": True,
    "input_truncation": 1024,
    "max_new_tokens": 512,
    "temperature": 0.2,
    "top_p": 0.95,
    "do_sample": True,
    "decode_generated_tokens_only": True,
    "skip_special_tokens": True,
    "output_strip": False,
    "automatic_raw_fallback": False,
}

EXPECTED_HASHES = {
    "active_strength_manifest": "f8781a4532a0d58840a89652342fb88eeda7480cf42924ae1ee451f0b7b898c7",
    "dev_subset_hashes": "8266cd3a95e025a96cb9a7cf0252bbcbef20954ad647031bdee0c59344493010",
    "prompt_ids": "891701d77ae9b6003de31452bce029e2e35bba105484609056ecfa782d678f46",
    "denominator_audit": "75b5ca0a029e43a49a9048d17a94713c0003aad6f144352539c6b0ecede3f619",
    "bstar_config": "f4f5d4d6a697aa14f654763f31e52cd852ce87b9f284276edd4f784604ea36e6",
    "final_config": "fe0e6184a7e389f33834a0e122c5e5cce40a06e0a5cd406ae4c9d2ba826c6f5d",
    "phase5_runner": "1ad49c599be9fe1fef1cf85e41cd79645b779d90aea43e6bfb1d424c9a455b5c",
    "phase9_runner": "71bf94971744e0c631973ec335d923cd52fbb1568001bbdb6eaf2e8adf6755fe",
}

EXPECTED_FEATURE_MAP = {
    "CWE-120": {"layer": 19, "feature": 14193},
    "CWE-787": {"layer": 19, "feature": 1515},
    "CWE-190": {"layer": 19, "feature": 16897},
    "CWE-327": {"layer": 23, "feature": 14449},
    "CWE-89": {"layer": 23, "feature": 1652},
    "CWE-338": {"layer": 23, "feature": 7533},
    "CWE-79": {"layer": 16, "feature": 9816},
    "CWE-125": {"layer": 23, "feature": 16655},
    "CWE-476": {"layer": 23, "feature": 18397},
}

RECORD_KEYS = (
    "schema_version",
    "prompt_id",
    "source_index",
    "population_type",
    "prompt_text",
    "prompt_text_sha256",
    "language",
    "cwe_id",
    "method",
    "condition_alpha",
    "applied_alpha",
    "run_seed",
    "steered",
    "route_status",
    "layer",
    "feature",
    "generated_code",
    "generated_token_count",
    "generation_status",
    "error",
    "empty_status",
    "fallback_status",
    "model_id",
    "model_cache_snapshot",
    "sae_release",
    "sae_id",
    "sae_cache_snapshot",
    "input_manifest_sha256",
    "bstar_config_sha256",
    "route_map_sha256",
    "runner_sha256",
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def resolve_cli_path(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def huggingface_hub_cache() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser().resolve()
    if os.environ.get("HF_HOME"):
        return (Path(os.environ["HF_HOME"]).expanduser() / "hub").resolve()
    return (Path.home() / ".cache/huggingface/hub").resolve()


def validate_cached_snapshots() -> dict[str, str]:
    hub = huggingface_hub_cache()
    model_snapshot = hub / MODEL_CACHE_REPO / "snapshots" / MODEL_CACHE_SNAPSHOT
    sae_snapshot = hub / SAE_CACHE_REPO / "snapshots" / SAE_CACHE_SNAPSHOT
    require(model_snapshot.is_dir(), f"missing frozen model cache snapshot: {model_snapshot}")
    require(sae_snapshot.is_dir(), f"missing frozen SAE cache snapshot: {sae_snapshot}")
    model_ref = hub / MODEL_CACHE_REPO / "refs/main"
    sae_ref = hub / SAE_CACHE_REPO / "refs/main"
    require(model_ref.is_file() and model_ref.read_text(encoding="utf-8").strip() == MODEL_CACHE_SNAPSHOT, "model cache main ref differs from frozen snapshot")
    require(sae_ref.is_file() and sae_ref.read_text(encoding="utf-8").strip() == SAE_CACHE_SNAPSHOT, "SAE cache main ref differs from frozen snapshot")
    for sae_id, directory in SAE_CACHE_DIRS.items():
        require((sae_snapshot / directory / "hyperparams.json").is_file(), f"missing {sae_id} hyperparams")
        require((sae_snapshot / directory / "checkpoints/final.safetensors").is_file(), f"missing {sae_id} weights")
    return {
        "huggingface_hub_cache": str(hub),
        "model_snapshot_path": str(model_snapshot),
        "sae_snapshot_path": str(sae_snapshot),
    }


def expected_targets(alpha: int) -> tuple[Path, Path, Path]:
    stem = f"bstar_strength_alpha{alpha}_seed42"
    return (
        ACTIVE_OUTPUT_DIR / f"{stem}.json",
        ACTIVE_OUTPUT_DIR / f"{stem}_checkpoint.json",
        ACTIVE_OUTPUT_DIR / f"{stem}_run_manifest.json",
    )


def validate_source_hash(path: Path, expected: str, label: str) -> None:
    require(path.is_file(), f"missing required {label}: {relative(path)}")
    require(sha256_path(path) == expected, f"{label} hash mismatch")


def validate_checkpoint_structure(
    checkpoint: dict[str, Any], condition: dict[str, Any], expected_ids: list[int]
) -> None:
    require(checkpoint.get("schema_version") == "phase13_checkpoint_v1", "incompatible checkpoint schema")
    require(checkpoint.get("condition") == condition, "checkpoint condition/config mismatch")
    records = checkpoint.get("records")
    completed_ids = checkpoint.get("completed_prompt_ids")
    require(isinstance(records, list) and isinstance(completed_ids, list), "checkpoint progress payload malformed")
    require(len(records) == len(completed_ids), "checkpoint records/ID lengths differ")
    require(len(completed_ids) <= len(expected_ids), "checkpoint has too many prompt IDs")
    require(completed_ids == expected_ids[: len(completed_ids)], "checkpoint prompt order mismatch")
    require(len(set(completed_ids)) == len(completed_ids), "checkpoint duplicate prompt IDs")
    require([record.get("prompt_id") for record in records] == completed_ids, "checkpoint record IDs differ")
    require(all(tuple(record.keys()) == RECORD_KEYS for record in records), "checkpoint record schema mismatch")
    rng = checkpoint.get("rng_state")
    require(isinstance(rng, dict), "checkpoint RNG state missing")
    require(isinstance(rng.get("torch_cpu_b64"), str), "checkpoint CPU RNG state missing")
    require(isinstance(rng.get("torch_cuda_all_b64"), list), "checkpoint CUDA RNG state missing")
    require(rng.get("cuda_device_count") == len(rng["torch_cuda_all_b64"]), "checkpoint CUDA RNG count mismatch")
    try:
        require(len(base64.b64decode(rng["torch_cpu_b64"], validate=True)) > 0, "empty CPU RNG state")
        for encoded in rng["torch_cuda_all_b64"]:
            require(len(base64.b64decode(encoded, validate=True)) > 0, "empty CUDA RNG state")
    except Exception as exc:
        fail(f"corrupted checkpoint RNG state: {exc}")


def validate_run_manifest_for_resume(run_manifest: dict[str, Any], condition: dict[str, Any]) -> None:
    require(run_manifest.get("schema_version") == "phase13_run_manifest_v1", "run-manifest schema mismatch")
    require(run_manifest.get("condition") == condition, "run-manifest condition/config mismatch")
    require(run_manifest.get("generation_status") in {"IN_PROGRESS", "INTERRUPTED"}, "run manifest is not resumable")


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    require(args.alpha in ALLOWED_ALPHAS, f"alpha must be one of {ALLOWED_ALPHAS}")
    require(args.seed == REQUIRED_SEED, "seed must be exactly 42")

    input_path = resolve_cli_path(args.input_manifest)
    output_path = resolve_cli_path(args.output)
    checkpoint_path = resolve_cli_path(args.checkpoint)
    expected_output, expected_checkpoint, run_manifest_path = expected_targets(args.alpha)
    require(input_path == ACTIVE_MANIFEST.resolve(), "only the active corrected strength manifest is allowed")
    require("pre_correction" not in input_path.name and "superseded" not in input_path.name.lower(), "superseded manifest rejected")
    require(output_path == expected_output.resolve(), "output target does not match the frozen alpha filename")
    require(checkpoint_path == expected_checkpoint.resolve(), "checkpoint target does not match the frozen alpha filename")
    require(not output_path.exists(), "final output already exists; stop for immutable-result review")

    validate_source_hash(input_path, EXPECTED_HASHES["active_strength_manifest"], "active strength manifest")
    validate_source_hash(HASH_RECORD, EXPECTED_HASHES["dev_subset_hashes"], "development subset hash record")
    validate_source_hash(DENOMINATOR_AUDIT, EXPECTED_HASHES["denominator_audit"], "Phase 2 denominator audit")
    validate_source_hash(BSTAR_CONFIG, EXPECTED_HASHES["bstar_config"], "B* config")
    validate_source_hash(FINAL_CONFIG, EXPECTED_HASHES["final_config"], "final config")
    validate_source_hash(PHASE5_RUNNER, EXPECTED_HASHES["phase5_runner"], "Phase 5 B* runner")
    validate_source_hash(PHASE9_RUNNER, EXPECTED_HASHES["phase9_runner"], "Phase 9 B* runner")

    manifest = read_json(input_path)
    hashes = read_json(HASH_RECORD)
    audit = read_json(DENOMINATOR_AUDIT)
    bstar = read_json(BSTAR_CONFIG)
    require(manifest.get("validation_status") == "PASS", "active manifest is not PASS")
    require(hashes.get("validation_status") == "PASS", "subset hash record is not PASS")
    require(audit.get("verification_status") == "PASS" and audit.get("blockers") == [], "denominator audit is not PASS")
    require(hashes["manifest_files"]["strength_subset_manifest"]["sha256"] == EXPECTED_HASHES["active_strength_manifest"], "hash record points to another strength manifest")
    require(hashes["source_hashes"]["denominator_audit"]["sha256"] == EXPECTED_HASHES["denominator_audit"], "hash record denominator mismatch")
    for source_entry in manifest.get("source_files", {}).values():
        source_path = (ROOT / source_entry["path"]).resolve()
        require(source_path.is_file(), f"missing manifest source: {source_entry['path']}")
        require(sha256_path(source_path) == source_entry["sha256"], f"manifest source hash mismatch: {source_entry['path']}")

    records = manifest.get("records")
    ids = manifest.get("prompt_ids_source_order")
    require(isinstance(records, list) and len(records) == EXPECTED_RECORD_COUNT, "active manifest must contain exactly 180 records")
    require(isinstance(ids, list) and len(ids) == EXPECTED_RECORD_COUNT, "active prompt-ID list must contain exactly 180 IDs")
    ids = [int(value) for value in ids]
    record_ids = [int(record["prompt_id"]) for record in records]
    require(record_ids == ids, "manifest record order differs from frozen prompt order")
    require(len(set(ids)) == len(ids), "duplicate prompt IDs in active manifest")
    require(canonical_hash(ids) == EXPECTED_HASHES["prompt_ids"], "active prompt-ID hash mismatch")
    require(hashes["component_hashes"]["strength_subset_prompt_ids_source_order"] == EXPECTED_HASHES["prompt_ids"], "hash record prompt-ID hash mismatch")
    require([int(record["source_index"]) for record in records] == sorted(int(record["source_index"]) for record in records), "manifest source order mismatch")

    vulnerable_ids = {int(value) for value in audit["development"]["b0_vulnerable_eligible_prompt_ids"]}
    safe_ids = {int(value) for value in audit["development"]["b0_safe_eligible_prompt_ids"]}
    eligible_ids = {int(value) for value in audit["development"]["scanner_eligible_prompt_ids"]}
    heldout_ids = {int(value) for value in audit["heldout"]["total_prompt_ids"]}
    manifest_vulnerable = {int(record["prompt_id"]) for record in records if record["population_type"] == "DEV_VULN"}
    safe_records = [record for record in records if record["population_type"] == "DEV_SAFE_LARGE"]
    manifest_safe = {int(record["prompt_id"]) for record in safe_records}
    require(manifest_vulnerable == vulnerable_ids and len(manifest_vulnerable) == 60, "DEV_VULN differs from verified audit set")
    require(len(manifest_safe) == 120 and manifest_safe <= safe_ids, "DEV_SAFE_LARGE differs from verified safe population")
    require(manifest_vulnerable.isdisjoint(manifest_safe), "vulnerable/safe overlap")
    require(set(ids).isdisjoint(heldout_ids), "held-out prompt ID in active manifest")
    require(set(ids) <= eligible_ids, "scanner-ineligible prompt in active manifest")
    require(all(record["scanner_eligible"] is True and record["b0_is_vulnerable"] is False for record in safe_records), "safe predicate mismatch")
    require({record["cwe_id"] for record in safe_records} <= FROZEN_SAFE_CWES, "safe record outside frozen four-CWE set")
    require({record["cwe_id"] for record in safe_records} == FROZEN_SAFE_CWES, "frozen safe-CWE coverage differs")
    require(manifest["construction"]["active_scanner_supported_cwes"] == sorted(FROZEN_SAFE_CWES), "manifest category authority mismatch")
    development_prompts = read_json(ROOT / manifest["source_files"]["development_prompts"]["path"])
    for record in records:
        source = record["source_prompt"]
        require(int(source["prompt_id"]) == int(record["prompt_id"]), "source metadata prompt ID mismatch")
        require(source == development_prompts[int(record["source_index"])], "embedded source metadata differs from development source")
        require(record["prompt_text_sha256"] == hashlib.sha256(source["test_case_prompt"].encode("utf-8")).hexdigest(), "prompt rendering hash mismatch")
        require(record["language"] == source["language"], "source language mismatch")
        require(record["cwe_id"] == source["cwe_identifier"], "source CWE mismatch")

    require(bstar.get("bstar") == "B2_a40" and float(bstar.get("alpha")) == 40.0, "frozen B* identity mismatch")
    require(bstar.get("feature_map") == EXPECTED_FEATURE_MAP, "frozen B* feature/layer map mismatch")
    cache_paths = validate_cached_snapshots()
    runner_path = Path(__file__).resolve()
    runner_hash = sha256_path(runner_path)
    route_map_hash = canonical_hash(bstar["feature_map"])
    condition = {
        "method": "B2_STRENGTH_CHARACTERIZATION",
        "alpha": args.alpha,
        "seed": args.seed,
        "input_manifest_path": relative(input_path),
        "input_manifest_sha256": EXPECTED_HASHES["active_strength_manifest"],
        "prompt_ids_sha256": EXPECTED_HASHES["prompt_ids"],
        "bstar_config_path": relative(BSTAR_CONFIG),
        "bstar_config_sha256": EXPECTED_HASHES["bstar_config"],
        "route_map_sha256": route_map_hash,
        "runner_path": relative(runner_path),
        "runner_sha256": runner_hash,
        "model_id": MODEL_ID,
        "model_cache_snapshot": MODEL_CACHE_SNAPSHOT,
        "sae_release": SAE_RELEASE,
        "sae_ids": list(SAE_IDS),
        "sae_cache_snapshot": SAE_CACHE_SNAPSHOT,
        "generation_settings": GENERATION_SETTINGS,
        "output_schema_version": "phase13_generation_record_v1",
        "output_path": relative(output_path),
        "checkpoint_path": relative(checkpoint_path),
        "run_manifest_path": relative(run_manifest_path),
        "record_count": EXPECTED_RECORD_COUNT,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
    }

    checkpoint = None
    run_manifest = None
    if checkpoint_path.exists():
        require(args.resume, "checkpoint exists but --resume was not supplied")
        require(run_manifest_path.is_file(), "checkpoint exists without its run manifest")
        checkpoint = read_json(checkpoint_path)
        run_manifest = read_json(run_manifest_path)
        validate_checkpoint_structure(checkpoint, condition, ids)
        validate_run_manifest_for_resume(run_manifest, condition)
    else:
        require(not run_manifest_path.exists(), "run manifest exists without a checkpoint; stop for review")

    return {
        "condition": condition,
        "records": records,
        "expected_ids": ids,
        "feature_map": bstar["feature_map"],
        "cache_paths": cache_paths,
        "checkpoint": checkpoint,
        "run_manifest": run_manifest,
        "output_path": output_path,
        "checkpoint_path": checkpoint_path,
        "run_manifest_path": run_manifest_path,
    }


def encode_rng_state(torch: Any) -> dict[str, Any]:
    def encode(tensor: Any) -> str:
        return base64.b64encode(bytes(tensor.detach().cpu().tolist())).decode("ascii")

    cuda_states = torch.cuda.get_rng_state_all()
    return {
        "torch_cpu_b64": encode(torch.get_rng_state()),
        "torch_cuda_all_b64": [encode(state) for state in cuda_states],
        "cuda_device_count": len(cuda_states),
    }


def restore_rng_state(torch: Any, payload: dict[str, Any]) -> None:
    require(torch.cuda.device_count() == payload["cuda_device_count"], "CUDA device count differs from checkpoint")

    def decode(value: str) -> Any:
        raw = base64.b64decode(value, validate=True)
        require(len(raw) > 0, "empty RNG-state payload")
        return torch.tensor(list(raw), dtype=torch.uint8)

    torch.set_rng_state(decode(payload["torch_cpu_b64"]))
    torch.cuda.set_rng_state_all([decode(value) for value in payload["torch_cuda_all_b64"]])


def empty_status(text: str) -> str:
    if text == "":
        return "EMPTY"
    if text.strip() == "":
        return "WHITESPACE_ONLY"
    return "NONEMPTY"


def build_record(
    manifest_record: dict[str, Any],
    condition: dict[str, Any],
    feature_map: dict[str, Any],
    generated_code: str,
    generated_token_count: int | None,
    generation_status: str,
    error: str | None,
) -> dict[str, Any]:
    route = feature_map.get(manifest_record["cwe_id"])
    values = {
        "schema_version": "phase13_generation_record_v1",
        "prompt_id": int(manifest_record["prompt_id"]),
        "source_index": int(manifest_record["source_index"]),
        "population_type": manifest_record["population_type"],
        "prompt_text": manifest_record["source_prompt"]["test_case_prompt"],
        "prompt_text_sha256": manifest_record["prompt_text_sha256"],
        "language": manifest_record["language"],
        "cwe_id": manifest_record["cwe_id"],
        "method": condition["method"],
        "condition_alpha": condition["alpha"],
        "applied_alpha": condition["alpha"] if route is not None else None,
        "run_seed": condition["seed"],
        "steered": route is not None,
        "route_status": "ROUTED" if route is not None else "UNROUTED_NO_FROZEN_BSTAR_ROUTE",
        "layer": int(route["layer"]) if route is not None else None,
        "feature": int(route["feature"]) if route is not None else None,
        "generated_code": generated_code,
        "generated_token_count": generated_token_count,
        "generation_status": generation_status,
        "error": error,
        "empty_status": empty_status(generated_code),
        "fallback_status": "NONE",
        "model_id": condition["model_id"],
        "model_cache_snapshot": condition["model_cache_snapshot"],
        "sae_release": condition["sae_release"] if route is not None else None,
        "sae_id": f"l{int(route['layer'])}r_8x" if route is not None else None,
        "sae_cache_snapshot": condition["sae_cache_snapshot"] if route is not None else None,
        "input_manifest_sha256": condition["input_manifest_sha256"],
        "bstar_config_sha256": condition["bstar_config_sha256"],
        "route_map_sha256": condition["route_map_sha256"],
        "runner_sha256": condition["runner_sha256"],
    }
    require(tuple(values.keys()) == RECORD_KEYS, "internal record schema mismatch")
    return values


def make_hook(torch: Any, sae: Any, feature_id: int, alpha: int, state: dict[str, Any]):
    """Exact audited full-hidden B2 hook with tuple-tail preservation."""
    def hook_fn(module: Any, inputs: Any, output: Any) -> Any:
        state["entered"] = True
        try:
            hidden = output[0] if isinstance(output, tuple) else output
            z = sae.encode(hidden)
            z[..., feature_id] += alpha
            h_steered = sae.decode(z).to(hidden.dtype)
            if isinstance(output, tuple):
                return (h_steered,) + output[1:]
            return h_steered
        except Exception as exc:
            state["exception"] = f"{type(exc).__name__}: {exc}"
            raise

    return hook_fn


def write_checkpoint(
    path: Path,
    condition: dict[str, Any],
    records: list[dict[str, Any]],
    torch: Any,
    elapsed_seconds: float,
    generated_tokens: int,
    events: list[dict[str, Any]],
    status: str,
    append_event: bool = True,
) -> dict[str, Any]:
    event = {
        "completed_record_count": len(records),
        "elapsed_seconds": elapsed_seconds,
        "created_at_utc": utc_now(),
    }
    events = events + [event] if append_event else events
    checkpoint = {
        "schema_version": "phase13_checkpoint_v1",
        "condition": condition,
        "status": status,
        "completed_prompt_ids": [record["prompt_id"] for record in records],
        "records": records,
        "rng_state": encode_rng_state(torch),
        "elapsed_seconds": elapsed_seconds,
        "generated_token_count": generated_tokens,
        "checkpoint_events": events,
        "updated_at_utc": utc_now(),
    }
    atomic_json(path, checkpoint)
    return checkpoint


def counts(records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "record_count": len(records),
        "steered_count": sum(record["steered"] is True for record in records),
        "unsteered_count": sum(record["steered"] is False for record in records),
        "successful_generation_count": sum(record["generation_status"] == "SUCCESS" for record in records),
        "empty_count": sum(record["empty_status"] == "EMPTY" for record in records),
        "whitespace_only_count": sum(record["empty_status"] == "WHITESPACE_ONLY" for record in records),
        "generation_or_hook_failure_count": sum(record["generation_status"] in {"HOOK_EXCEPTION", "GENERATION_EXCEPTION", "MALFORMED_GENERATION"} for record in records),
        "fallback_count": sum(record["fallback_status"] != "NONE" for record in records),
    }


def execute_generation(state: dict[str, Any]) -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    from sae_lens import SAE
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    condition = state["condition"]
    checkpoint = state["checkpoint"]
    resume = checkpoint is not None
    if not resume:
        torch.manual_seed(condition["seed"])

    started_at = utc_now()
    timer = time.perf_counter()
    prior_elapsed = float(checkpoint.get("elapsed_seconds", 0.0)) if resume else 0.0
    records = list(checkpoint.get("records", [])) if resume else []
    generated_tokens = int(checkpoint.get("generated_token_count", 0)) if resume else 0
    checkpoint_events = list(checkpoint.get("checkpoint_events", [])) if resume else []
    previous_run_manifest = state["run_manifest"] or {}
    resume_events = list(previous_run_manifest.get("resume_events", []))
    if resume:
        resume_events.append({"resumed_at_utc": started_at, "completed_record_count": len(records)})

    run_manifest = {
        "schema_version": "phase13_run_manifest_v1",
        "phase": 13,
        "condition": condition,
        "start_state": "RESUME" if resume else "FRESH",
        "started_at_utc": previous_run_manifest.get("started_at_utc", started_at),
        "last_process_started_at_utc": started_at,
        "resume_events": resume_events,
        "generation_status": "IN_PROGRESS",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        },
        "final_record_count": None,
        "counts": None,
        "elapsed_seconds": prior_elapsed,
        "generated_token_count": generated_tokens,
        "checkpoint_events": checkpoint_events,
        "output_sha256": None,
        "completed_at_utc": None,
    }
    require(torch.cuda.is_available(), "CUDA is required for the frozen Phase 13 generation")
    atomic_json(state["run_manifest_path"], run_manifest)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_CACHE_SNAPSHOT, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_CACHE_SNAPSHOT,
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    required_layers = sorted({int(route["layer"]) for route in state["feature_map"].values()})
    saes: dict[int, Any] = {}
    for layer in required_layers:
        sae, _, _ = SAE.from_pretrained(release=SAE_RELEASE, sae_id=f"l{layer}r_8x")
        saes[layer] = sae.to("cuda").eval()

    if resume:
        restore_rng_state(torch, checkpoint["rng_state"])

    for manifest_record in state["records"][len(records):]:
        route = state["feature_map"].get(manifest_record["cwe_id"])
        generated_code = ""
        generated_token_count: int | None = None
        generation_status = "GENERATION_EXCEPTION"
        error: str | None = None
        hook = None
        hook_state: dict[str, Any] = {"entered": False, "exception": None}
        abort_after_record = False
        try:
            prompt_text = manifest_record["source_prompt"]["test_case_prompt"]
            inputs = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=GENERATION_SETTINGS["input_truncation"],
            ).to(model.device)
            if route is not None:
                layer = int(route["layer"])
                try:
                    hook = model.model.layers[layer].register_forward_hook(
                        make_hook(torch, saes[layer], int(route["feature"]), condition["alpha"], hook_state)
                    )
                except Exception as exc:
                    hook_state["exception"] = f"{type(exc).__name__}: {exc}"
                    raise
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=GENERATION_SETTINGS["max_new_tokens"],
                    temperature=GENERATION_SETTINGS["temperature"],
                    top_p=GENERATION_SETTINGS["top_p"],
                    do_sample=GENERATION_SETTINGS["do_sample"],
                    pad_token_id=tokenizer.pad_token_id,
                )
            require(hasattr(generated, "shape") and len(generated.shape) == 2 and generated.shape[0] == 1, "malformed generation tensor")
            prompt_length = int(inputs["input_ids"].shape[1])
            require(int(generated.shape[1]) >= prompt_length, "generated tensor shorter than input")
            new_tokens = generated[0][prompt_length:]
            generated_token_count = int(new_tokens.numel())
            generated_code = tokenizer.decode(new_tokens, skip_special_tokens=True)
            require(isinstance(generated_code, str), "decoded generation is not a string")
            status = empty_status(generated_code)
            generation_status = "SUCCESS" if status == "NONEMPTY" else status
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if hook_state.get("exception") is not None:
                generation_status = "HOOK_EXCEPTION"
                error = hook_state["exception"]
            elif isinstance(exc, RuntimeError) and str(exc).startswith("malformed"):
                generation_status = "MALFORMED_GENERATION"
            else:
                generation_status = "GENERATION_EXCEPTION"
        finally:
            if hook is not None:
                try:
                    hook.remove()
                except Exception as exc:
                    generation_status = "HOOK_EXCEPTION"
                    error = f"hook removal {type(exc).__name__}: {exc}"
                    abort_after_record = True

        record = build_record(
            manifest_record,
            condition,
            state["feature_map"],
            generated_code,
            generated_token_count,
            generation_status,
            error,
        )
        records.append(record)
        generated_tokens += generated_token_count or 0

        if not abort_after_record and (
            len(records) % CHECKPOINT_INTERVAL == 0 or len(records) == len(state["records"])
        ):
            elapsed = prior_elapsed + (time.perf_counter() - timer)
            checkpoint = write_checkpoint(
                state["checkpoint_path"],
                condition,
                records,
                torch,
                elapsed,
                generated_tokens,
                checkpoint_events,
                "IN_PROGRESS",
            )
            checkpoint_events = checkpoint["checkpoint_events"]
            run_manifest["checkpoint_events"] = checkpoint_events
            run_manifest["elapsed_seconds"] = elapsed
            run_manifest["generated_token_count"] = generated_tokens
            run_manifest["counts"] = counts(records)
            atomic_json(state["run_manifest_path"], run_manifest)
            print(f"checkpoint={len(records)}/{len(state['records'])}")
        if abort_after_record:
            elapsed = prior_elapsed + (time.perf_counter() - timer)
            checkpoint = write_checkpoint(
                state["checkpoint_path"],
                condition,
                records,
                torch,
                elapsed,
                generated_tokens,
                checkpoint_events,
                "INTERRUPTED",
            )
            run_manifest.update({
                "generation_status": "INTERRUPTED",
                "counts": counts(records),
                "elapsed_seconds": elapsed,
                "generated_token_count": generated_tokens,
                "checkpoint_events": checkpoint["checkpoint_events"],
            })
            atomic_json(state["run_manifest_path"], run_manifest)
            fail("hook removal failed; progress preserved and the condition stopped")

    require(len(records) == EXPECTED_RECORD_COUNT, "final record count mismatch")
    require([record["prompt_id"] for record in records] == state["expected_ids"], "final prompt order mismatch")
    require(len({record["prompt_id"] for record in records}) == EXPECTED_RECORD_COUNT, "duplicate final prompt IDs")
    require(all(tuple(record.keys()) == RECORD_KEYS for record in records), "nonuniform final schema")
    atomic_json(state["output_path"], records)
    output_hash = sha256_path(state["output_path"])
    elapsed = prior_elapsed + (time.perf_counter() - timer)
    checkpoint = write_checkpoint(
        state["checkpoint_path"],
        condition,
        records,
        torch,
        elapsed,
        generated_tokens,
        checkpoint_events,
        "COMPLETE",
        append_event=False,
    )
    run_manifest.update({
        "generation_status": "COMPLETE",
        "final_record_count": len(records),
        "counts": counts(records),
        "elapsed_seconds": elapsed,
        "generated_token_count": generated_tokens,
        "checkpoint_events": checkpoint["checkpoint_events"],
        "output_sha256": output_hash,
        "peak_cuda_memory_bytes": max(
            (torch.cuda.max_memory_allocated(index) for index in range(torch.cuda.device_count())),
            default=0,
        ),
        "completed_at_utc": utc_now(),
    })
    atomic_json(state["run_manifest_path"], run_manifest)
    print(json.dumps({
        "generation_status": "COMPLETE",
        "output_path": relative(state["output_path"]),
        "output_sha256": output_hash,
        "record_count": len(records),
        "counts": run_manifest["counts"],
        "elapsed_seconds": elapsed,
        "generated_token_count": generated_tokens,
        "resume_event_count": len(resume_events),
    }, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", required=True, type=int, choices=ALLOWED_ALPHAS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = preflight(args)
    resolved = {
        "preflight_status": "PASS",
        "model_loaded": False,
        "generation_executed": False,
        "condition": state["condition"],
        "population": {
            "record_count": len(state["records"]),
            "unique_prompt_ids": len(set(state["expected_ids"])),
            "dev_vuln": sum(record["population_type"] == "DEV_VULN" for record in state["records"]),
            "dev_safe_large": sum(record["population_type"] == "DEV_SAFE_LARGE" for record in state["records"]),
            "safe_cwes": sorted({record["cwe_id"] for record in state["records"] if record["population_type"] == "DEV_SAFE_LARGE"}),
        },
        "resume_checkpoint_present": state["checkpoint"] is not None,
    }
    if args.preflight_only:
        print(json.dumps(resolved, indent=2, sort_keys=True))
        return
    execute_generation(state)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[FAIL] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
