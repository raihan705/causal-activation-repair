#!/usr/bin/env python3
"""Generate matched B0 and active single-route B* utility conditions.

The preflight path uses only the Python standard library. Generation requires
the frozen local model/SAE cache and one CUDA device. Utility tasks never
provide or consume a CWE label; a route is selected only as an experimental
condition.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from routewise_utility_common import (
    ALPHA, APPROVAL, BENCHMARK_ORDER, CHECKPOINT_INTERVAL, EXPECTED_COUNTS,
    EXPECTED_ROUTES, ExperimentError, MODEL_ID, MODEL_REVISION, OUTPUTS, PHASE17_INPUT,
    PROTOCOL, SAE_RELEASE, SEED, atomic_json, baseline_checkpoint,
    baseline_output, baseline_run_manifest, capture_rng_state, empty_status,
    load_protocol, mmlu_prediction, ordered_tasks, read_json, relative,
    render_task, require, restore_rng_state, route_checkpoint, route_output,
    route_run_manifest, route_slug, sha256_path, sha256_text, validate_approval,
    validate_cache, validate_sources,
)


RUNNER = Path(__file__).resolve()
COMMON = Path(__file__).with_name("routewise_utility_common.py")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def runner_condition(mode: str, protocol: dict[str, Any], cwe_id: str | None = None) -> dict[str, Any]:
    condition: dict[str, Any] = {
        "mode": mode,
        "seed": SEED,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "input_manifest_sha256": sha256_path(PHASE17_INPUT),
        "protocol_sha256": sha256_path(PROTOCOL),
        "runner_sha256": sha256_path(RUNNER),
        "common_sha256": sha256_path(COMMON),
        "benchmark_order": list(BENCHMARK_ORDER),
        "generation": protocol["generation"],
        "task_cwe_fields_consumed": [],
    }
    if cwe_id is not None:
        route = protocol["routes"][cwe_id]
        condition.update({
            "scientific_role": "ACTIVE_SINGLE_ROUTE_UTILITY_SENSITIVITY",
            "route_cwe_id": cwe_id,
            "layer": int(route["layer"]),
            "feature": int(route["feature"]),
            "alpha": ALPHA,
            "simultaneous_features": False,
            "route_selection": "EXPERIMENT_CONDITION_ONLY",
        })
    else:
        condition.update({"scientific_role": "FRESH_PAIRED_B0", "intervention": False})
    return condition


def validate_prefix(records: list[dict[str, Any]], tasks: list[dict[str, Any]], label: str) -> None:
    require(len(records) <= len(tasks), f"{label} checkpoint has too many records")
    expected = [str(task["stable_id"]) for task in tasks[:len(records)]]
    actual = [str(record.get("stable_id")) for record in records]
    require(actual == expected, f"{label} checkpoint is not an exact task-order prefix")
    require(len(set(actual)) == len(actual), f"{label} checkpoint contains duplicate IDs")


def load_baseline(tasks: list[dict[str, Any]], protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    output = baseline_output()
    manifest_path = baseline_run_manifest()
    require(output.is_file() and manifest_path.is_file(),
            "Fresh paired B0 is incomplete; run --baseline before any route")
    manifest = read_json(manifest_path)
    require(manifest.get("status") == "COMPLETE", "Fresh paired B0 run manifest is not COMPLETE")
    require(manifest.get("condition") == runner_condition("baseline", protocol),
            "Fresh paired B0 condition mismatch")
    require(manifest.get("output_sha256") == sha256_path(output), "Fresh paired B0 output hash mismatch")
    records = read_json(output)
    require(isinstance(records, list) and len(records) == len(tasks), "Fresh paired B0 count mismatch")
    validate_prefix(records, tasks, "Fresh paired B0")
    require(all(isinstance(row.get("rng_state_before"), dict) for row in records),
            "Fresh paired B0 lacks task-level RNG states")
    return records, sha256_path(output)


def state_for_target(mode: str, protocol: dict[str, Any], tasks: list[dict[str, Any]],
                     resume: bool, cwe_id: str | None = None) -> dict[str, Any]:
    if mode == "baseline":
        output, checkpoint, run_manifest = baseline_output(), baseline_checkpoint(), baseline_run_manifest()
    else:
        require(cwe_id in protocol["routes"], f"Unknown route: {cwe_id}")
        route = protocol["routes"][cwe_id]
        output, checkpoint, run_manifest = (
            route_output(cwe_id, route), route_checkpoint(cwe_id, route), route_run_manifest(cwe_id, route)
        )
    condition = runner_condition(mode, protocol, cwe_id)
    if output.exists():
        require(run_manifest.is_file(), f"{output.name} exists without a run manifest")
        completed = read_json(run_manifest)
        require(completed.get("status") == "COMPLETE", f"{run_manifest.name} is not COMPLETE")
        require(completed.get("condition") == condition, f"{run_manifest.name} condition mismatch")
        require(completed.get("output_sha256") == sha256_path(output), f"{output.name} hash mismatch")
        records = read_json(output)
        require(isinstance(records, list) and len(records) == len(tasks), f"{output.name} count mismatch")
        validate_prefix(records, tasks, output.name)
        return {"status": "COMPLETE", "records": records, "output": output,
                "checkpoint": checkpoint, "run_manifest": run_manifest, "condition": condition}
    records: list[dict[str, Any]] = []
    saved_rng = None
    if checkpoint.exists():
        require(resume, f"{checkpoint.name} exists; --resume is required")
        payload = read_json(checkpoint)
        require(payload.get("condition") == condition, f"{checkpoint.name} condition mismatch")
        records = payload.get("records", [])
        require(isinstance(records, list), f"{checkpoint.name} records are malformed")
        validate_prefix(records, tasks, checkpoint.name)
        if mode == "baseline":
            saved_rng = payload.get("rng_state_after_prefix")
            require(isinstance(saved_rng, dict), "B0 checkpoint RNG state is absent")
    else:
        require(not run_manifest.exists(), f"{run_manifest.name} exists without output/checkpoint")
    return {"status": "PENDING", "records": records, "saved_rng": saved_rng,
            "output": output, "checkpoint": checkpoint, "run_manifest": run_manifest,
            "condition": condition}


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    source, _ = validate_sources()
    protocol = load_protocol()
    validate_approval(RUNNER, COMMON)
    tasks = ordered_tasks(source)
    cache = validate_cache(require_sae=not args.baseline)
    if args.baseline:
        target_states = {"B0": state_for_target("baseline", protocol, tasks, args.resume)}
    else:
        load_baseline(tasks, protocol)
        cwes = list(protocol["routes"]) if args.all else [args.route]
        target_states = {cwe: state_for_target("route", protocol, tasks, args.resume, cwe) for cwe in cwes}
    summary = {
        "schema_version": "routewise_bstar_utility_preflight_v1",
        "status": "PASS",
        "model_loaded": False,
        "generation_run": False,
        "mode": "baseline" if args.baseline else ("all_routes" if args.all else "single_route"),
        "task_count": len(tasks),
        "benchmark_counts": EXPECTED_COUNTS,
        "route_count": 0 if args.baseline else len(target_states),
        "targets": {key: {"status": value["status"], "completed_records": len(value["records"]),
                          "output": relative(value["output"])} for key, value in target_states.items()},
        "protocol_path": relative(PROTOCOL),
        "protocol_sha256": sha256_path(PROTOCOL),
        "approval_path": relative(APPROVAL),
        "approval_sha256": sha256_path(APPROVAL),
        "cache": cache,
        "paired_rng": not args.baseline,
    }
    atomic_json(OUTPUTS / ("baseline_preflight.json" if args.baseline else "route_preflight.json"), summary)
    return {"summary": summary, "source": source, "protocol": protocol,
            "tasks": tasks, "target_states": target_states}


def environment(torch_module: Any) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "pytorch": torch_module.__version__,
        "cuda_runtime": torch_module.version.cuda,
        "cuda_available": torch_module.cuda.is_available(),
        "cuda_device_count": torch_module.cuda.device_count(),
        "cuda_devices": [torch_module.cuda.get_device_name(i)
                         for i in range(torch_module.cuda.device_count())],
    }


def load_model():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    require(torch.cuda.is_available(), "CUDA is required")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True,
        torch_dtype=torch.float16, device_map="auto",
    ).eval()
    return torch, tokenizer, model


def load_sae(layer: int):
    from sae_lens import SAE
    sae, _, _ = SAE.from_pretrained(release=SAE_RELEASE, sae_id=f"l{layer}r_8x")
    return sae.to("cuda").eval()


def make_route_hook(sae: Any, feature: int, layer: int, tracker: dict[str, Any]):
    def hook_fn(module: Any, inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        latent = sae.encode(hidden)
        latent[..., feature] += ALPHA
        edited = sae.decode(latent).to(hidden.dtype)
        signature = {"shape": list(hidden.shape), "dtype": str(hidden.dtype),
                     "device": str(hidden.device), "layer": layer,
                     "feature": feature, "alpha": ALPHA}
        tracker["hook_call_count"] += 1
        tracker["first_hook"] = tracker.get("first_hook") or signature
        tracker["last_hook"] = signature
        return (edited,) + output[1:] if isinstance(output, tuple) else edited
    return hook_fn


def generate(torch_module: Any, tokenizer: Any, model: Any, rendered: str,
             input_limit: int, output_limit: int, eos_token_ids: Any,
             sae: Any = None, layer: int | None = None, feature: int | None = None,
             ) -> tuple[str, int, int, dict[str, Any]]:
    inputs = tokenizer(rendered, return_tensors="pt", truncation=True,
                       max_length=input_limit).to(model.device)
    tracker: dict[str, Any] = {"hook_call_count": 0}
    handle = None
    if sae is not None:
        require(layer is not None and feature is not None, "Incomplete route hook")
        handle = model.model.layers[layer].register_forward_hook(
            make_route_hook(sae, feature, layer, tracker)
        )
    try:
        kwargs: dict[str, Any] = {
            "max_new_tokens": output_limit,
            "temperature": 0.2,
            "top_p": 0.95,
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if eos_token_ids is not None:
            kwargs["eos_token_id"] = eos_token_ids
        with torch_module.inference_mode():
            generated = model.generate(**inputs, **kwargs)
    finally:
        if handle is not None:
            handle.remove()
    new_tokens = generated[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text, int(new_tokens.shape[0]), int(inputs["input_ids"].shape[1]), tracker


def base_record(task: dict[str, Any], rendered: str, text: str, generated_tokens: int,
                input_tokens: int, condition: dict[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": "routewise_bstar_utility_record_v1",
        "stable_id": str(task["stable_id"]),
        "benchmark": task["benchmark"],
        "source_index": int(task["source_index"]),
        "source_prompt_sha256": sha256_text(str(task.get("prompt", task.get("question", "")))),
        "rendered_prompt_sha256": sha256_text(rendered),
        "generated_text": text,
        "input_token_count": input_tokens,
        "generated_token_count": generated_tokens,
        "empty_status": empty_status(text),
        "condition": condition,
    }
    if task["benchmark"] == "mmlu":
        predicted = mmlu_prediction(text)
        record.update({"slice": task["slice"], "answer": task["answer"],
                       "predicted": predicted, "correct": predicted == task["answer"]})
    return record


def write_progress(path: Path, condition: dict[str, Any], records: list[dict[str, Any]],
                   rng_state: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {
        "schema_version": "routewise_bstar_utility_checkpoint_v1",
        "status": "IN_PROGRESS",
        "condition": condition,
        "completed_records": len(records),
        "records": records,
        "updated_at_utc": utc_now(),
    }
    if rng_state is not None:
        payload["rng_state_after_prefix"] = rng_state
    atomic_json(path, payload)


def start_manifest(path: Path, condition: dict[str, Any], torch_module: Any,
                   completed: int) -> dict[str, Any]:
    manifest = {
        "schema_version": "routewise_bstar_utility_run_manifest_v1",
        "status": "IN_PROGRESS",
        "condition": condition,
        "environment": environment(torch_module),
        "started_at_utc": utc_now(),
        "completed_records": completed,
        "output_sha256": None,
    }
    atomic_json(path, manifest)
    return manifest


def finish_target(state: dict[str, Any], records: list[dict[str, Any]], manifest: dict[str, Any],
                  started: float, torch_module: Any) -> None:
    require(len(records) == sum(EXPECTED_COUNTS.values()), "Final generation coverage mismatch")
    atomic_json(state["output"], records)
    manifest.update({
        "status": "COMPLETE",
        "completed_records": len(records),
        "output_sha256": sha256_path(state["output"]),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": max(
            (torch_module.cuda.max_memory_allocated(i) for i in range(torch_module.cuda.device_count())),
            default=0,
        ),
        "completed_at_utc": utc_now(),
    })
    atomic_json(state["run_manifest"], manifest)


def execute_baseline(preflight_state: dict[str, Any]) -> None:
    state = preflight_state["target_states"]["B0"]
    if state["status"] == "COMPLETE":
        print("Fresh paired B0 is already COMPLETE; no generation performed.")
        return
    tasks = preflight_state["tasks"]
    torch, tokenizer, model = load_model()
    if state.get("saved_rng") is None:
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    else:
        restore_rng_state(torch, state["saved_rng"])
    records = list(state["records"])
    run_manifest = start_manifest(state["run_manifest"], state["condition"], torch, len(records))
    started = time.perf_counter()
    try:
        for task in tasks[len(records):]:
            rendered, input_limit, output_limit, eos_token_ids = render_task(tokenizer, task)
            rng_before = capture_rng_state(torch)
            text, generated_tokens, input_tokens, tracker = generate(
                torch, tokenizer, model, rendered, input_limit, output_limit, eos_token_ids,
            )
            require(tracker["hook_call_count"] == 0, "B0 unexpectedly entered a steering hook")
            record = base_record(task, rendered, text, generated_tokens, input_tokens, state["condition"])
            record["rng_state_before"] = rng_before
            records.append(record)
            if len(records) % CHECKPOINT_INTERVAL == 0 or len(records) == len(tasks):
                write_progress(state["checkpoint"], state["condition"], records, capture_rng_state(torch))
                run_manifest["completed_records"] = len(records)
                atomic_json(state["run_manifest"], run_manifest)
                print(f"B0 checkpoint={len(records)}/{len(tasks)}", flush=True)
    except Exception as exc:
        run_manifest.update({"status": "INTERRUPTED", "completed_records": len(records),
                             "error": f"{type(exc).__name__}: {exc}", "updated_at_utc": utc_now()})
        atomic_json(state["run_manifest"], run_manifest)
        raise
    finish_target(state, records, run_manifest, started, torch)
    print(json.dumps({"status": "COMPLETE", "condition": "B0", "records": len(records),
                      "output": relative(state["output"]),
                      "sha256": sha256_path(state["output"])}, indent=2))


def execute_routes(preflight_state: dict[str, Any]) -> None:
    tasks = preflight_state["tasks"]
    protocol = preflight_state["protocol"]
    b0_records, b0_sha256 = load_baseline(tasks, protocol)
    torch, tokenizer, model = load_model()
    for cwe_id, state in preflight_state["target_states"].items():
        if state["status"] == "COMPLETE":
            print(f"{cwe_id} is already COMPLETE; skipping.", flush=True)
            continue
        route = protocol["routes"][cwe_id]
        layer, feature = int(route["layer"]), int(route["feature"])
        sae = load_sae(layer)
        records = list(state["records"])
        run_manifest = start_manifest(state["run_manifest"], state["condition"], torch, len(records))
        run_manifest["paired_b0_sha256"] = b0_sha256
        atomic_json(state["run_manifest"], run_manifest)
        started = time.perf_counter()
        try:
            for index, task in enumerate(tasks[len(records):], start=len(records)):
                rendered, input_limit, output_limit, eos_token_ids = render_task(tokenizer, task)
                b0_record = b0_records[index]
                require(b0_record["stable_id"] == str(task["stable_id"]), "Paired B0 task mismatch")
                require(b0_record["rendered_prompt_sha256"] == sha256_text(rendered),
                        "Paired B0 rendered-prompt mismatch")
                restore_rng_state(torch, b0_record["rng_state_before"])
                text, generated_tokens, input_tokens, tracker = generate(
                    torch, tokenizer, model, rendered, input_limit, output_limit, eos_token_ids,
                    sae=sae, layer=layer, feature=feature,
                )
                require(tracker["hook_call_count"] > 0, f"{cwe_id} route hook was not entered")
                record = base_record(task, rendered, text, generated_tokens, input_tokens, state["condition"])
                record.update({
                    "paired_b0_output_sha256": sha256_text(b0_record["generated_text"]),
                    "paired_rng_state_sha256": sha256_text(json.dumps(
                        b0_record["rng_state_before"], sort_keys=True, separators=(",", ":")
                    )),
                    "hook_evidence": tracker,
                })
                records.append(record)
                if len(records) % CHECKPOINT_INTERVAL == 0 or len(records) == len(tasks):
                    write_progress(state["checkpoint"], state["condition"], records)
                    run_manifest["completed_records"] = len(records)
                    atomic_json(state["run_manifest"], run_manifest)
                    print(f"{cwe_id} checkpoint={len(records)}/{len(tasks)}", flush=True)
        except Exception as exc:
            run_manifest.update({"status": "INTERRUPTED", "completed_records": len(records),
                                 "error": f"{type(exc).__name__}: {exc}", "updated_at_utc": utc_now()})
            atomic_json(state["run_manifest"], run_manifest)
            raise
        finish_target(state, records, run_manifest, started, torch)
        print(json.dumps({"status": "COMPLETE", "route": cwe_id, "layer": layer,
                          "feature": feature, "records": len(records),
                          "output": relative(state["output"]),
                          "sha256": sha256_path(state["output"])}, indent=2), flush=True)
        del sae
        gc.collect()
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--baseline", action="store_true")
    selection.add_argument("--route", choices=list(EXPECTED_ROUTES))
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state = preflight(args)
    print(json.dumps(state["summary"], indent=2, sort_keys=True))
    if args.preflight_only:
        return 0
    if args.baseline:
        execute_baseline(state)
    else:
        execute_routes(state)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExperimentError as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        raise SystemExit(2)
