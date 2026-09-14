#!/usr/bin/env python3
"""Run one frozen Phase 16A primary generation condition; never scans or scores."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from phase16_common import *

ALLOWED = {
    "B0": {43, 44}, "B1": {43, 44},
    "B2-alpha20": {42, 43, 44}, "B*": {43, 44},
}
INFORMATION_TIER = {"B0": "NO_INTERVENTION_BASELINE", "B1": "METADATA_FREE", "B2-alpha20": "ORACLE_CWE", "B*": "ORACLE_CWE"}
BSTAR_CONFIG = ROOT / "configs/bstar_config.json"


def make_hook(sae: Any, feature_id: int, alpha: float, state: dict[str, Any]):
    def hook_fn(module: Any, inputs: Any, output: Any) -> Any:
        state["entered"] = True
        try:
            hidden = output[0] if isinstance(output, tuple) else output
            z = sae.encode(hidden)
            z[..., feature_id] += alpha
            edited = sae.decode(z).to(hidden.dtype)
            if isinstance(output, tuple):
                return (edited,) + output[1:]
            return edited
        except Exception as exc:
            state["exception"] = f"{type(exc).__name__}: {exc}"
            raise
    return hook_fn


def record_for(source: dict[str, Any], method: str, seed: int, condition: dict[str, Any], route: dict[str, Any] | None,
               code: str, generated_tokens: int, status: str, failure: str | None, hook_entered: bool) -> dict[str, Any]:
    steered = route is not None and method in ("B2-alpha20", "B*")
    return {
        "schema_version": "phase16_primary_generation_record_v1",
        "prompt_id": source["prompt_id"], "source_index": source["source_index"],
        "prompt_text": source["prompt_text"], "prompt_text_sha256": source["prompt_text_sha256"],
        "language": source["language"], "cwe_id": source["cwe_id"],
        "method": method, "seed": seed, "information_tier": INFORMATION_TIER[method],
        "generated_code": code, "generated_token_count": generated_tokens,
        "generation_status": status, "failure_reason": failure, "empty_status": empty_status(code),
        "intervention_applied": steered, "route_status": (
            "NO_INTERVENTION_BASELINE" if method in ("B0", "B1") else
            "SUPPORTED_ROUTE_APPLIED" if route is not None else "UNSUPPORTED_ROUTE_RAW_FALLBACK"
        ),
        "fallback_status": "RAW_UNMAPPED_ROUTE" if method in ("B2-alpha20", "B*") and route is None else "NONE",
        "alpha": condition["alpha"] if steered else None,
        "layer": int(route["layer"]) if steered else None,
        "feature": int(route["feature"]) if steered else None,
        "hook_entered": hook_entered if steered else None,
        "phase15_freeze_sha256": EXPECTED_HASHES["revision_freeze_manifest.json"],
        "population_sha256": condition["population_sha256"], "prompt_ids_sha256": condition["prompt_ids_sha256"],
        "config_path": condition["config_path"], "config_sha256": condition["config_sha256"],
        "runner_path": condition["runner_path"], "runner_sha256": condition["runner_sha256"],
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "sae_release": SAE_RELEASE if steered else None,
        "sae_id": SAE_IDS[int(route["layer"])] if steered else None,
        "sae_revision": SAE_REVISION if steered else None,
    }


def validate_checkpoint(payload: dict[str, Any], condition: dict[str, Any], population: dict[str, Any]) -> list[dict[str, Any]]:
    require(payload.get("schema_version") == "phase16_primary_checkpoint_v1", "checkpoint schema mismatch")
    require(payload.get("condition") == condition, "checkpoint condition mismatch")
    rows = payload.get("records", [])
    ids = [int(row["prompt_id"]) for row in rows]
    require(ids == population["prompt_ids"][:len(ids)], "checkpoint is not a strict frozen-order prefix")
    require(isinstance(payload.get("rng_state"), dict), "checkpoint RNG state missing")
    return rows


def run(method: str, seed: int, resume: bool) -> None:
    require(method in ALLOWED and seed in ALLOWED[method], "condition is outside the frozen protocol")
    gate = phase15_gate()
    population = load_frozen_population()
    spec = command_spec(method, seed)
    output, checkpoint_path, manifest_path = condition_paths(method, seed)
    require(relative(output) == spec["output_path"], "output path differs from Phase15 command spec")
    require(relative(checkpoint_path) == spec["checkpoint_path"], "checkpoint path differs from Phase15 command spec")
    require(relative(manifest_path) == spec["run_manifest_path"], "run-manifest path differs from Phase15 command spec")
    require(not Path(ROOT / spec["scanner_procedure"]["scan_output_path"]).exists(), "scanner output already exists for new condition")
    runner_path = Path(__file__).resolve()
    runner_hash = sha256_path(runner_path)
    alpha = 20.0 if method == "B2-alpha20" else 40.0 if method == "B*" else None
    config_path = spec["config_path"]
    condition = {
        "method": method, "seed": seed, "information_tier": INFORMATION_TIER[method], "alpha": alpha,
        "population_sha256": population["population_sha256"], "prompt_ids_sha256": population["prompt_ids_sha256"],
        "phase15_freeze_sha256": EXPECTED_HASHES["revision_freeze_manifest.json"],
        "phase15_command_sha256": EXPECTED_HASHES["phase16_command_specifications.json"],
        "config_path": config_path, "config_sha256": spec["config_sha256"],
        "runner_path": relative(runner_path), "runner_sha256": runner_hash,
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "sae_release": SAE_RELEASE if method in ("B2-alpha20", "B*") else None,
        "sae_revision": SAE_REVISION if method in ("B2-alpha20", "B*") else None,
        "generation_settings": GENERATION_SETTINGS,
        "provenance_status": "NEW_REVISION_GENERATION",
    }
    if method == "B2-alpha20" and seed == 42:
        condition["historical_relationship"] = "FRESH_REVISION_RUN_NOT_REPLAY_OR_RECONSTRUCTION_OF_HISTORICAL_SCALAR"

    if output.exists():
        require(manifest_path.is_file(), "immutable output lacks run manifest")
        rows = read_json(output)
        validate_record_order(rows, population, method, seed)
        manifest = read_json(manifest_path)
        require(manifest.get("status") == "COMPLETE" and manifest.get("output_sha256") == sha256_path(output), "existing output manifest invalid")
        print(json.dumps({"status": "ALREADY_COMPLETE_VALIDATED", "method": method, "seed": seed, "output_sha256": sha256_path(output)}))
        return

    checkpoint = None
    rows: list[dict[str, Any]] = []
    if checkpoint_path.exists():
        require(resume, "checkpoint exists; --resume is required")
        checkpoint = read_json(checkpoint_path)
        rows = validate_checkpoint(checkpoint, condition, population)
    else:
        require(not resume or not manifest_path.exists(), "orphan run manifest without checkpoint")

    import torch
    seed_torch(torch, seed)
    start = utc_now()
    prior_manifest = read_json(manifest_path) if manifest_path.exists() else {}
    resume_events = list(prior_manifest.get("resume_events", []))
    if checkpoint is not None:
        resume_events.append({"timestamp_utc": start, "completed_records": len(rows)})
    manifest = {
        "schema_version": "phase16_generation_run_manifest_v1", "phase": 16, "substage": "16A",
        "condition": condition, "status": "IN_PROGRESS", "started_at_utc": prior_manifest.get("started_at_utc", start),
        "last_process_started_at_utc": start, "completed_at_utc": None,
        "resume_events": resume_events, "completion_resume_status": "RESUMED" if checkpoint else "FRESH",
        "completed_records": len(rows), "record_count": EXPECTED_COUNT,
        "environment": None, "counts": None, "output_path": relative(output), "output_sha256": None,
        "checkpoint_path": relative(checkpoint_path), "scan_status": "NOT_STARTED",
    }
    atomic_json(manifest_path, manifest)
    torch, tokenizer, model = load_model()
    manifest["environment"] = environment(torch)
    feature_map = read_json(BSTAR_CONFIG)["feature_map"] if method in ("B2-alpha20", "B*") else {}
    saes = load_saes([16, 19, 23]) if feature_map else {}
    if checkpoint is not None:
        restore_rng_state(torch, checkpoint["rng_state"])
    timer = time.perf_counter()
    prior_elapsed = float(prior_manifest.get("elapsed_seconds", 0.0))
    generated_total = sum(int(row.get("generated_token_count", 0)) for row in rows)
    for source in population["records"][len(rows):]:
        route = feature_map.get(source["cwe_id"])
        prompt = SUBMITTED_B1_PREFIX + source["prompt_text"] if method == "B1" else source["prompt_text"]
        code, token_count, status, failure = "", 0, "FAILED", None
        handle = None
        hook_state = {"entered": False, "exception": None}
        try:
            if route is not None:
                handle = model.model.layers[int(route["layer"])].register_forward_hook(
                    make_hook(saes[int(route["layer"])], int(route["feature"]), float(alpha), hook_state)
                )
            code, token_count, _ = generate_once(torch, model, tokenizer, prompt)
            status = "COMPLETED"
        except Exception as exc:
            failure = hook_state["exception"] or f"{type(exc).__name__}: {exc}"
            status = "HOOK_FAILED" if hook_state["exception"] else "GENERATION_FAILED"
        finally:
            if handle is not None:
                try:
                    handle.remove()
                except Exception as exc:
                    status, failure = "HOOK_FAILED", f"hook removal {type(exc).__name__}: {exc}"
        rows.append(record_for(source, method, seed, condition, route, code, token_count, status, failure, hook_state["entered"]))
        generated_total += token_count
        if len(rows) % CHECKPOINT_INTERVAL == 0 or len(rows) == EXPECTED_COUNT:
            payload = {"schema_version": "phase16_primary_checkpoint_v1", "condition": condition, "records": rows,
                       "rng_state": capture_rng_state(torch), "completed_records": len(rows),
                       "elapsed_seconds": prior_elapsed + time.perf_counter() - timer}
            atomic_json(checkpoint_path, payload)
            manifest["completed_records"] = len(rows)
            manifest["elapsed_seconds"] = payload["elapsed_seconds"]
            manifest["generated_token_count"] = generated_total
            atomic_json(manifest_path, manifest)
            print(f"{method} seed{seed}: checkpoint {len(rows)}/{EXPECTED_COUNT}", flush=True)
    validate_record_order(rows, population, method, seed)
    atomic_json(output, rows)
    counts = summarize_records(rows)
    manifest.update({
        "status": "COMPLETE", "completed_at_utc": utc_now(), "completed_records": EXPECTED_COUNT,
        "counts": counts, "elapsed_seconds": prior_elapsed + time.perf_counter() - timer,
        "generated_token_count": generated_total, "output_sha256": sha256_path(output),
        "structural_validation": "PASS", "scan_status": "NOT_STARTED",
    })
    atomic_json(manifest_path, manifest)
    print(json.dumps({"status": "COMPLETE", "method": method, "seed": seed, "counts": counts,
                      "output_sha256": sha256_path(output), "run_manifest_sha256": sha256_path(manifest_path)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, default=FREEZE)
    parser.add_argument("--method", required=True, choices=list(ALLOWED))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    require(args.freeze.resolve() == FREEZE.resolve(), "only the frozen Phase15 manifest is accepted")
    require(args.method in ALLOWED and args.seed in ALLOWED[args.method], "condition is outside the frozen protocol")
    if args.preflight_only:
        population = load_frozen_population(); spec = command_spec(args.method, args.seed)
        print(json.dumps({"status": "PASS", "method": args.method, "seed": args.seed,
                          "population_count": len(population["records"]), "command_spec": spec}, indent=2))
        return
    run(args.method, args.seed, args.resume)

if __name__ == "__main__":
    main()
