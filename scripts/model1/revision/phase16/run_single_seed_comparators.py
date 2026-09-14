#!/usr/bin/env python3
"""Run one frozen Phase 16A single-seed comparator; no scanning or metrics."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from phase16_common import *

ALLOWED = {"B3-ungated", "B1-CWE", "RCI-1", "CAA-CWE"}
INFORMATION_TIER = {"B3-ungated": "ORACLE_CWE", "B1-CWE": "ORACLE_CWE", "RCI-1": "METADATA_FREE", "CAA-CWE": "ORACLE_CWE"}
INTERVENTION_LIBRARY = ROOT / "configs/intervention_library.json"
B1_SPEC = ROOT / "revision/model1/phase14/outputs/b1cwe_method_spec.json"
B1_GUIDANCE = ROOT / "revision/model1/phase14/outputs/b1cwe_guidance_map.json"
RCI_SPEC = ROOT / "revision/model1/phase14/outputs/rci_method_spec.json"
RCI_SELECTION = ROOT / "revision/model1/phase14/outputs/rci_selection.json"
CAA_CONFIG = ROOT / "revision/model1/phase14/outputs/caa_cwe_config.json"
CAA_VECTOR_MANIFEST = ROOT / "revision/model1/phase14/outputs/caa_vector_manifest.json"
CAA_MASK_SOURCE = ROOT / "revision/model1/phase14/scripts/caa_position_mask.py"
LANGUAGE_TAGS = {"python": "python", "c": "c", "c++": "cpp", "cpp": "cpp", "java": "java", "javascript": "javascript", "js": "javascript"}


def make_group_hook(sae: Any, features: list[int], alpha: float, state: dict[str, Any]):
    def hook_fn(module: Any, inputs: Any, output: Any) -> Any:
        state["entered"] = True
        try:
            hidden = output[0] if isinstance(output, tuple) else output
            z = sae.encode(hidden)
            for feature in features:
                z[..., feature] += alpha
            edited = sae.decode(z).to(hidden.dtype)
            if isinstance(output, tuple):
                return (edited,) + output[1:]
            return edited
        except Exception as exc:
            state["exception"] = f"{type(exc).__name__}: {exc}"
            raise
    return hook_fn


def selected_groups() -> dict[str, dict[str, Any]]:
    priority = {"mixed_robust": 0, "semantic": 1, "statistical_fallback": 2}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_json(INTERVENTION_LIBRARY)["groups"]:
        grouped[row["cwe_id"]].append(row)
    return {cwe: sorted(rows, key=lambda row: priority.get(row["group_type"], 3))[0] for cwe, rows in grouped.items()}


def language_tag(language: str) -> str:
    return LANGUAGE_TAGS.get(language.strip().lower(), language.strip().lower() or "text")


def extract_fenced_code(text: str) -> tuple[str, str | None]:
    match = re.search(r"```([A-Za-z0-9_+.#-]+)[ \t]*\r?\n([\s\S]*?)```", text)
    if not match or not match.group(2).strip():
        return "", "EXTRACTION_FAILED_INVALID_OR_EMPTY"
    return match.group(2), None


def base_condition(method: str, population: dict[str, Any], spec: dict[str, Any], runner_path: Path) -> dict[str, Any]:
    condition = {
        "method": method, "seed": 42, "information_tier": INFORMATION_TIER[method],
        "population_sha256": population["population_sha256"], "prompt_ids_sha256": population["prompt_ids_sha256"],
        "phase15_freeze_sha256": EXPECTED_HASHES["revision_freeze_manifest.json"],
        "phase15_command_sha256": EXPECTED_HASHES["phase16_command_specifications.json"],
        "config_path": spec["config_path"], "config_sha256": spec["config_sha256"],
        "runner_path": relative(runner_path), "runner_sha256": sha256_path(runner_path),
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "generation_settings": GENERATION_SETTINGS,
        "provenance_status": "NEW_REVISION_GENERATION",
    }
    if method == "B3-ungated":
        condition.update({"sae_release": SAE_RELEASE, "sae_revision": SAE_REVISION, "base_alpha": 20.0, "plausibility_gate": False, "intervention_library_sha256": sha256_path(INTERVENTION_LIBRARY)})
    elif method == "B1-CWE":
        condition.update({"method_spec_sha256": sha256_path(B1_SPEC), "guidance_map_sha256": sha256_path(B1_GUIDANCE), "guidance_available_route_count": 0})
    elif method == "RCI-1":
        condition.update({"rci_variant": "rci1", "cycles": 1, "intended_calls_per_complete_prompt": 3, "method_spec_sha256": sha256_path(RCI_SPEC), "selection_sha256": sha256_path(RCI_SELECTION)})
    elif method == "CAA-CWE":
        condition.update({"layer": 16, "multiplier": 1.0, "caa_config_sha256": sha256_path(CAA_CONFIG), "vector_manifest_sha256": sha256_path(CAA_VECTOR_MANIFEST), "position_mask_source_sha256": sha256_path(CAA_MASK_SOURCE)})
    return condition


def ordinary_record(source: dict[str, Any], method: str, condition: dict[str, Any], code: str, token_count: int,
                    status: str, failure: str | None, route: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "phase16_comparator_generation_record_v1",
        "prompt_id": source["prompt_id"], "source_index": source["source_index"],
        "prompt_text": source["prompt_text"], "prompt_text_sha256": source["prompt_text_sha256"],
        "language": source["language"], "cwe_id": source["cwe_id"],
        "method": method, "seed": 42, "information_tier": INFORMATION_TIER[method],
        "generated_code": code, "generated_token_count": token_count,
        "generation_status": status, "failure_reason": failure, "empty_status": empty_status(code),
        **route,
        "phase15_freeze_sha256": EXPECTED_HASHES["revision_freeze_manifest.json"],
        "population_sha256": condition["population_sha256"], "prompt_ids_sha256": condition["prompt_ids_sha256"],
        "config_path": condition["config_path"], "config_sha256": condition["config_sha256"],
        "runner_path": condition["runner_path"], "runner_sha256": condition["runner_sha256"],
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
    }


def validate_ordinary_checkpoint(payload: dict[str, Any], condition: dict[str, Any], population: dict[str, Any]) -> list[dict[str, Any]]:
    require(payload.get("schema_version") == "phase16_comparator_checkpoint_v1", "checkpoint schema mismatch")
    require(payload.get("condition") == condition, "checkpoint condition mismatch")
    rows = payload.get("records", [])
    require([int(row["prompt_id"]) for row in rows] == population["prompt_ids"][:len(rows)], "checkpoint order mismatch")
    require(isinstance(payload.get("rng_state"), dict), "checkpoint RNG state missing")
    return rows


def load_caa_vectors(torch: Any) -> dict[str, Any]:
    manifest = read_json(CAA_VECTOR_MANIFEST)
    vectors = {}
    for entry in manifest["entries"]:
        if int(entry["layer"]) != 16:
            continue
        path = ROOT / entry["artifact_path"]
        require(sha256_path(path) == entry["artifact_sha256"], f"CAA vector hash mismatch for {entry['cwe_id']}")
        vectors[entry["cwe_id"]] = torch.load(path, map_location="cpu", weights_only=True)["prepared_vector"]
    require(set(vectors) == {"CWE-120", "CWE-327", "CWE-89", "CWE-338"}, "CAA L16 route set mismatch")
    return vectors


def run_ordinary(method: str, resume: bool) -> None:
    population = load_frozen_population(); spec = command_spec(method, 42)
    output, checkpoint_path, manifest_path = condition_paths(method, 42)
    require(relative(output) == spec["output_path"] and relative(checkpoint_path) == spec["checkpoint_path"] and relative(manifest_path) == spec["run_manifest_path"], "condition paths differ from Phase15")
    require(not (ROOT / spec["scanner_procedure"]["scan_output_path"]).exists(), "scanner output already exists")
    condition = base_condition(method, population, spec, Path(__file__).resolve())
    if output.exists():
        rows = read_json(output); validate_record_order(rows, population, method, 42)
        manifest = read_json(manifest_path)
        require(manifest["status"] == "COMPLETE" and manifest["output_sha256"] == sha256_path(output), "existing manifest invalid")
        print(json.dumps({"status": "ALREADY_COMPLETE_VALIDATED", "method": method, "output_sha256": sha256_path(output)})); return
    checkpoint = read_json(checkpoint_path) if checkpoint_path.exists() else None
    if checkpoint is not None: require(resume, "checkpoint exists; --resume required")
    rows = validate_ordinary_checkpoint(checkpoint, condition, population) if checkpoint else []
    import torch
    seed_torch(torch, 42)
    started = utc_now(); prior = read_json(manifest_path) if manifest_path.exists() else {}
    resume_events = list(prior.get("resume_events", []))
    if checkpoint: resume_events.append({"timestamp_utc": started, "completed_records": len(rows)})
    manifest = {"schema_version": "phase16_generation_run_manifest_v1", "phase": 16, "substage": "16A", "condition": condition,
                "status": "IN_PROGRESS", "started_at_utc": prior.get("started_at_utc", started), "last_process_started_at_utc": started,
                "completed_at_utc": None, "resume_events": resume_events, "completion_resume_status": "RESUMED" if checkpoint else "FRESH",
                "completed_records": len(rows), "record_count": EXPECTED_COUNT, "environment": None, "counts": None,
                "elapsed_seconds": float(prior.get("elapsed_seconds", 0.0)), "output_path": relative(output), "output_sha256": None,
                "checkpoint_path": relative(checkpoint_path), "scan_status": "NOT_STARTED"}
    atomic_json(manifest_path, manifest)
    torch, tokenizer, model = load_model(); manifest["environment"] = environment(torch)
    groups = selected_groups() if method == "B3-ungated" else {}
    saes = load_saes([int(group["layer_id"]) for group in groups.values()]) if groups else {}
    guidance = read_json(B1_GUIDANCE) if method == "B1-CWE" else None
    vectors = load_caa_vectors(torch) if method == "CAA-CWE" else {}
    CAAResidualPositionHook = None
    if method == "CAA-CWE":
        sys.path.insert(0, str(CAA_MASK_SOURCE.parent)); from caa_position_mask import CAAResidualPositionHook as HookClass; CAAResidualPositionHook = HookClass
    if checkpoint: restore_rng_state(torch, checkpoint["rng_state"])
    timer = time.perf_counter(); prior_elapsed = float(prior.get("elapsed_seconds", 0.0)); generated_total = sum(int(row.get("generated_token_count", 0)) for row in rows)
    for source in population["records"][len(rows):]:
        code, token_count, status, failure = "", 0, "FAILED", None
        handle = None; hook_state = {"entered": False, "exception": None}; route: dict[str, Any]
        prompt = source["prompt_text"]
        if method == "B3-ungated":
            group = groups.get(source["cwe_id"])
            route = {"route_status": "SUPPORTED_GROUP_APPLIED" if group else "UNSUPPORTED_ROUTE_RAW_FALLBACK", "fallback_status": "NONE" if group else "RAW_UNMAPPED_ROUTE", "intervention_applied": group is not None,
                     "group_id": group["group_id"] if group else None, "layer": int(group["layer_id"]) if group else None, "feature_ids": list(group["feature_ids"]) if group else None, "alpha": float(group["base_alpha"]) if group else None, "hook_entered": None}
        elif method == "B1-CWE":
            route_cfg = guidance["routes"].get(source["cwe_id"])
            route_status = "ROUTE_GUIDANCE_UNAVAILABLE" if route_cfg else "UNSUPPORTED_ROUTE_FALLBACK_B1"
            route = {"route_status": route_status, "guidance_available": False, "fallback_status": "SUBMITTED_B1", "intervention_applied": False}
            prompt = SUBMITTED_B1_PREFIX + prompt
        else:
            vector = vectors.get(source["cwe_id"])
            route = {"route_status": "SUPPORTED_VECTOR_APPLIED" if vector is not None else "UNSUPPORTED_ROUTE_NO_INTERVENTION", "fallback_status": "NONE", "intervention_applied": vector is not None,
                     "no_intervention_reason": None if vector is not None else "NO_SUPPORTED_ROUTE_VECTOR", "layer": 16 if vector is not None else None, "multiplier": 1.0 if vector is not None else None, "hook_call_records": []}
        try:
            if method == "B3-ungated" and group is not None:
                handle = model.model.layers[int(group["layer_id"])].register_forward_hook(make_group_hook(saes[int(group["layer_id"])], list(group["feature_ids"]), float(group["base_alpha"]), hook_state))
            elif method == "CAA-CWE" and vector is not None:
                hook = CAAResidualPositionHook(vector, 1.0, expected_width=4096); handle = model.model.layers[16].register_forward_hook(hook)
            else: hook = None
            code, token_count, _ = generate_once(torch, model, tokenizer, prompt); status = "COMPLETED"
            if method == "B3-ungated" and group is not None: route["hook_entered"] = hook_state["entered"]
            if method == "CAA-CWE" and vector is not None: route["hook_call_records"] = hook.call_records
        except Exception as exc:
            failure = hook_state["exception"] or f"{type(exc).__name__}: {exc}"; status = "HOOK_FAILED" if hook_state["exception"] else "GENERATION_FAILED"
            if method == "CAA-CWE" and 'hook' in locals() and hook is not None: route["hook_call_records"] = hook.call_records
        finally:
            if handle is not None:
                try: handle.remove()
                except Exception as exc: status, failure = "HOOK_FAILED", f"hook removal {type(exc).__name__}: {exc}"
        rows.append(ordinary_record(source, method, condition, code, token_count, status, failure, route)); generated_total += token_count
        if len(rows) % CHECKPOINT_INTERVAL == 0 or len(rows) == EXPECTED_COUNT:
            elapsed = prior_elapsed + time.perf_counter() - timer
            atomic_json(checkpoint_path, {"schema_version": "phase16_comparator_checkpoint_v1", "condition": condition, "records": rows, "rng_state": capture_rng_state(torch), "completed_records": len(rows), "elapsed_seconds": elapsed})
            manifest.update({"completed_records": len(rows), "elapsed_seconds": elapsed, "generated_token_count": generated_total}); atomic_json(manifest_path, manifest)
            print(f"{method} seed42: checkpoint {len(rows)}/{EXPECTED_COUNT}", flush=True)
    validate_record_order(rows, population, method, 42); atomic_json(output, rows); counts = summarize_records(rows)
    manifest.update({"status": "COMPLETE", "completed_at_utc": utc_now(), "completed_records": EXPECTED_COUNT, "counts": counts,
                     "elapsed_seconds": prior_elapsed + time.perf_counter() - timer, "generated_token_count": generated_total,
                     "output_sha256": sha256_path(output), "structural_validation": "PASS", "scan_status": "NOT_STARTED"}); atomic_json(manifest_path, manifest)
    print(json.dumps({"status": "COMPLETE", "method": method, "counts": counts, "output_sha256": sha256_path(output), "run_manifest_sha256": sha256_path(manifest_path)}, indent=2))


def write_rci_cost(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["prompt_id", "variant", "stage_index", "stage_name", "stage_input_token_count", "generated_token_count", "elapsed_seconds", "model_call_index", "completion_status", "failure_reason", "final_output_designation"]
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader()
        for row in rows: writer.writerow({key: row.get(key) for key in fields})
    temporary.replace(path)


def validate_rci_prefix(rows: list[dict[str, Any]], population: dict[str, Any]) -> None:
    stages = ["initial", "critique_1", "improve_1"]
    expected = [(prompt_id, index, stage) for prompt_id in population["prompt_ids"] for index, stage in enumerate(stages)]
    actual = [(int(row["prompt_id"]), int(row["stage_index"]), row["stage_name"]) for row in rows]
    require(actual == expected[:len(actual)], "RCI checkpoint is not a strict stage-order prefix")


def run_rci(resume: bool) -> None:
    method = "RCI-1"; population = load_frozen_population(); spec_cmd = command_spec(method, 42)
    output, checkpoint_path, manifest_path = condition_paths(method, 42); cost_path = OUTPUTS / "rci_1_seed42_cost.csv"
    require(relative(output) == spec_cmd["output_path"] and relative(checkpoint_path) == spec_cmd["checkpoint_path"] and relative(manifest_path) == spec_cmd["run_manifest_path"], "RCI paths differ from Phase15")
    require(not (ROOT / spec_cmd["scanner_procedure"]["scan_output_path"]).exists(), "RCI scanner output already exists")
    condition = base_condition(method, population, spec_cmd, Path(__file__).resolve())
    if output.exists():
        body = read_json(output); finals = body["final_outputs"]; validate_record_order(finals, population, method, 42)
        require(read_json(manifest_path)["output_sha256"] == sha256_path(output), "existing RCI manifest invalid")
        print(json.dumps({"status": "ALREADY_COMPLETE_VALIDATED", "method": method, "output_sha256": sha256_path(output)})); return
    checkpoint = read_json(checkpoint_path) if checkpoint_path.exists() else None
    if checkpoint: require(resume, "RCI checkpoint exists; --resume required")
    stage_rows = checkpoint.get("stage_records", []) if checkpoint else []; validate_rci_prefix(stage_rows, population)
    spec = read_json(RCI_SPEC); stages = ["initial", "critique_1", "improve_1"]
    import torch
    seed_torch(torch, 42); started = utc_now(); prior = read_json(manifest_path) if manifest_path.exists() else {}; resume_events = list(prior.get("resume_events", []))
    if checkpoint: resume_events.append({"timestamp_utc": started, "completed_stage_rows": len(stage_rows)})
    manifest = {"schema_version": "phase16_generation_run_manifest_v1", "phase": 16, "substage": "16A", "condition": condition, "status": "IN_PROGRESS",
                "started_at_utc": prior.get("started_at_utc", started), "last_process_started_at_utc": started, "completed_at_utc": None,
                "resume_events": resume_events, "completion_resume_status": "RESUMED" if checkpoint else "FRESH", "completed_stage_rows": len(stage_rows),
                "record_count": EXPECTED_COUNT, "environment": None, "counts": None, "elapsed_seconds": float(prior.get("elapsed_seconds", 0.0)),
                "output_path": relative(output), "output_sha256": None, "checkpoint_path": relative(checkpoint_path), "cost_path": relative(cost_path), "scan_status": "NOT_STARTED"}
    atomic_json(manifest_path, manifest); torch, tokenizer, model = load_model(); manifest["environment"] = environment(torch)
    if checkpoint: restore_rng_state(torch, checkpoint["rng_state"])
    timer = time.perf_counter(); prior_elapsed = float(prior.get("elapsed_seconds", 0.0)); flat = 0
    for source in population["records"]:
        prompt_existing = [row for row in stage_rows if int(row["prompt_id"]) == source["prompt_id"]]
        current_code, current_critique, prerequisite_failed = "", "", False
        for old in prompt_existing:
            if old["stage_name"] in ("initial", "improve_1"):
                if old["completion_status"] == "COMPLETED": current_code = old["extracted_code"]
                else: prerequisite_failed = True
            elif old["stage_name"] == "critique_1":
                if old["completion_status"] == "COMPLETED": current_critique = old["generated_text"]
                else: prerequisite_failed = True
        for stage_index, stage_name in enumerate(stages):
            if flat < len(stage_rows): flat += 1; continue
            final = stage_name == "improve_1"
            if prerequisite_failed:
                row = {"prompt_id": source["prompt_id"], "variant": "rci1", "stage_index": stage_index, "stage_name": stage_name,
                       "rendered_stage_input": "", "stage_input_token_count": 0, "generated_token_count": 0, "generated_text": "", "extracted_code": "",
                       "elapsed_seconds": 0.0, "model_call_index": None, "completion_status": "SKIPPED_PREREQUISITE_FAILURE",
                       "failure_reason": "A required prior stage failed", "final_output_designation": final}
            else:
                tag = language_tag(source["language"])
                if stage_name == "initial": rendered = spec["templates"]["initial"]["template"].format(language=source["language"], language_tag=tag, coding_task=source["prompt_text"])
                elif stage_name == "critique_1": rendered = spec["templates"]["criticism"]["template"].format(current_code=current_code)
                else: rendered = spec["templates"]["improvement"]["template"].format(language_tag=tag, current_critique=current_critique, current_code=current_code)
                generated_text, extracted, failure, completion, generated_count, input_count = "", "", None, "FAILED", 0, 0
                call_started = time.perf_counter()
                try:
                    generated_text, generated_count, input_count = generate_once(torch, model, tokenizer, rendered)
                    if stage_name in ("initial", "improve_1"):
                        extracted, failure = extract_fenced_code(generated_text); completion = "COMPLETED" if failure is None else "EXTRACTION_FAILED_INVALID_OR_EMPTY"
                    else:
                        completion = "COMPLETED" if generated_text.strip() else "INVALID_OR_EMPTY"; failure = None if completion == "COMPLETED" else "Empty criticism output"
                except Exception as exc: failure = f"{type(exc).__name__}: {exc}"
                row = {"prompt_id": source["prompt_id"], "variant": "rci1", "stage_index": stage_index, "stage_name": stage_name,
                       "rendered_stage_input": rendered, "stage_input_token_count": input_count, "generated_token_count": generated_count,
                       "generated_text": generated_text, "extracted_code": extracted, "elapsed_seconds": time.perf_counter() - call_started,
                       "model_call_index": stage_index + 1, "completion_status": completion, "failure_reason": failure, "final_output_designation": final}
                if completion == "COMPLETED":
                    if stage_name in ("initial", "improve_1"): current_code = extracted
                    else: current_critique = generated_text
                else: prerequisite_failed = True
            stage_rows.append(row); flat += 1
            elapsed = prior_elapsed + time.perf_counter() - timer
            atomic_json(checkpoint_path, {"schema_version": "phase16_rci_checkpoint_v1", "condition": condition, "stage_records": stage_rows, "rng_state": capture_rng_state(torch), "elapsed_seconds": elapsed})
            manifest.update({"completed_stage_rows": len(stage_rows), "elapsed_seconds": elapsed}); atomic_json(manifest_path, manifest)
        print(f"RCI-1 seed42: checkpoint prompt {source['source_index'] + 1}/{EXPECTED_COUNT}", flush=True)
    validate_rci_prefix(stage_rows, population); finals_by_id = {int(row["prompt_id"]): row for row in stage_rows if row["final_output_designation"]}
    finals = []
    for source in population["records"]:
        row = finals_by_id[source["prompt_id"]]
        finals.append({"schema_version": "phase16_rci_final_record_v1", "prompt_id": source["prompt_id"], "source_index": source["source_index"],
                       "prompt_text_sha256": source["prompt_text_sha256"], "language": source["language"], "cwe_id": source["cwe_id"], "method": method,
                       "seed": 42, "information_tier": "METADATA_FREE", "generated_code": row["extracted_code"], "generated_token_count": row["generated_token_count"],
                       "generation_status": row["completion_status"], "failure_reason": row["failure_reason"], "empty_status": empty_status(row["extracted_code"]),
                       "fallback_status": "NONE", "route_status": "METADATA_FREE", "intervention_applied": False,
                       "phase15_freeze_sha256": EXPECTED_HASHES["revision_freeze_manifest.json"], "population_sha256": condition["population_sha256"],
                       "config_path": condition["config_path"], "config_sha256": condition["config_sha256"], "runner_path": condition["runner_path"], "runner_sha256": condition["runner_sha256"],
                       "model_id": MODEL_ID, "model_revision": MODEL_REVISION})
    validate_record_order(finals, population, method, 42); atomic_json(output, {"condition": condition, "stage_records": stage_rows, "final_outputs": finals}); write_rci_cost(cost_path, stage_rows)
    counts = summarize_records(finals); stage_failures = defaultdict(int)
    for row in stage_rows:
        if row["completion_status"] != "COMPLETED": stage_failures[row["stage_name"]] += 1
    rci = {"final_output_count": len(finals), "total_stage_rows": len(stage_rows), "total_model_calls": sum(row["model_call_index"] is not None for row in stage_rows),
           "failure_counts_by_stage": dict(stage_failures), "input_tokens": sum(int(row["stage_input_token_count"]) for row in stage_rows),
           "generated_tokens": sum(int(row["generated_token_count"]) for row in stage_rows), "stage_elapsed_seconds": sum(float(row["elapsed_seconds"]) for row in stage_rows)}
    manifest.update({"status": "COMPLETE", "completed_at_utc": utc_now(), "completed_stage_rows": len(stage_rows), "counts": counts, "rci_structural_cost": rci,
                     "elapsed_seconds": prior_elapsed + time.perf_counter() - timer, "generated_token_count": rci["generated_tokens"], "cost_sha256": sha256_path(cost_path),
                     "output_sha256": sha256_path(output), "structural_validation": "PASS", "scan_status": "NOT_STARTED"}); atomic_json(manifest_path, manifest)
    print(json.dumps({"status": "COMPLETE", "method": method, "counts": counts, "rci": rci, "output_sha256": sha256_path(output), "run_manifest_sha256": sha256_path(manifest_path)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--freeze", type=Path, default=FREEZE); parser.add_argument("--method", required=True, choices=sorted(ALLOWED)); parser.add_argument("--seed", required=True, type=int); parser.add_argument("--resume", action="store_true"); parser.add_argument("--preflight-only", action="store_true"); args = parser.parse_args()
    require(args.freeze.resolve() == FREEZE.resolve(), "only the frozen Phase15 manifest is accepted"); require(args.seed == 42, "single-seed comparators require seed42")
    if args.preflight_only:
        population = load_frozen_population(); print(json.dumps({"status": "PASS", "method": args.method, "seed": 42, "population_count": len(population["records"]), "command_spec": command_spec(args.method, 42)}, indent=2)); return
    if args.method == "RCI-1": run_rci(args.resume)
    else: run_ordinary(args.method, args.resume)

if __name__ == "__main__": main()
