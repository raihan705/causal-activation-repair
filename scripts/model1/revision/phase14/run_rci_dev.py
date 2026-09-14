#!/usr/bin/env python3
"""Run or preflight exactly RCI-1 or RCI-2 on the frozen development subset."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from phase14_common import (
    BASELINE_SUBSET, EXPECTED_BASELINE_SUBSET_SHA256, EXPECTED_RECORD_COUNT,
    EXPECTED_SOURCE_RECOVERY_SHA256, MODEL_CACHE_SNAPSHOT, MODEL_ID, OUTPUTS,
    REQUIRED_SEED, atomic_json, capture_rng_state, read_json, record_source_prompt_metadata_free,
    relative, require, restore_rng_state, sha256_path, validate_preparation,
)


SPEC = OUTPUTS / "rci_method_spec.json"
RUNNER = Path(__file__).resolve()
LANGUAGE_TAGS = {"python": "python", "c": "c", "c++": "cpp", "cpp": "cpp",
                 "java": "java", "javascript": "javascript"}


def targets(variant: str) -> tuple[Path, Path, Path, Path]:
    return (
        OUTPUTS / f"{variant}_dev_outputs.json",
        OUTPUTS / f"{variant}_dev_checkpoint.json",
        OUTPUTS / f"{variant}_dev_run_manifest.json",
        OUTPUTS / f"{variant}_dev_cost.csv",
    )


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    state = validate_preparation()
    require(args.variant in {"rci1", "rci2"}, "variant must be rci1 or rci2")
    require(args.seed == REQUIRED_SEED, "seed must be exactly 42")
    require(args.input.resolve() == BASELINE_SUBSET.resolve(), "only the frozen baseline subset is allowed")
    require(args.spec.resolve() == SPEC.resolve() and args.spec.is_file(), "frozen RCI spec missing")
    expected = targets(args.variant)
    require((args.output, args.checkpoint, args.run_manifest, args.cost_output) == tuple(p.resolve() for p in expected),
            "RCI output/checkpoint path policy mismatch")

    spec = read_json(args.spec)
    candidate = spec.get("candidates", {}).get(args.variant)
    require(spec.get("schema_version") == "phase14_rci_method_spec_v1", "RCI spec schema mismatch")
    require(spec.get("information_tier") == "METADATA_FREE" and spec.get("uses_cwe_metadata") is False,
            "RCI metadata tier mismatch")
    require(spec.get("seed") == 42, "RCI spec seed mismatch")
    require(spec["input"]["sha256"] == EXPECTED_BASELINE_SUBSET_SHA256, "RCI subset hash mismatch")
    require(spec["source_identity"]["source_recovery_sha256"] == EXPECTED_SOURCE_RECOVERY_SHA256,
            "RCI source-recovery hash mismatch")
    require(candidate is not None, "RCI candidate missing")
    expected_calls = 3 if args.variant == "rci1" else 5
    expected_cycles = 1 if args.variant == "rci1" else 2
    require(candidate["total_model_calls"] == expected_calls and candidate["cycles"] == expected_cycles,
            "RCI candidate depth/call count mismatch")
    require(len(candidate["stage_names"]) == expected_calls, "RCI stage count mismatch")
    require(spec["stopping"]["early_stopping"] is False, "early stopping is forbidden")

    ids = [int(record["prompt_id"]) for record in state["records"]]
    if args.checkpoint.exists():
        require(args.resume, "RCI checkpoint exists; --resume is required")
        checkpoint = read_json(args.checkpoint)
        require(checkpoint.get("condition", {}).get("spec_sha256") == sha256_path(args.spec),
                "checkpoint RCI spec hash mismatch")
        require(checkpoint.get("condition", {}).get("variant") == args.variant,
                "checkpoint variant mismatch")
        rows = checkpoint.get("stage_records", [])
        require(isinstance(rows, list), "checkpoint stage records missing")
        require(all(int(row["prompt_id"]) in ids for row in rows), "checkpoint has unknown prompt ID")
        require(isinstance(checkpoint.get("rng_state"), dict), "checkpoint RNG state missing")
    else:
        require(not args.resume, "--resume supplied without RCI checkpoint")

    return {
        "schema_version": "phase14_rci_preflight_v1",
        "status": "PASS", "variant": args.variant, "cycles": expected_cycles,
        "model_calls_per_prompt": expected_calls, "stage_names": candidate["stage_names"],
        "record_count": EXPECTED_RECORD_COUNT, "expected_stage_row_count": 120 * expected_calls,
        "seed": 42, "information_tier": "METADATA_FREE", "uses_cwe_metadata": False,
        "source_order_preserved": True, "generation_executed": False,
        "model_loaded": False, "held_out_accessed": False, "scanner_executed": False,
        "checkpoint_granularity": "every prompt and every stage",
        "resume_reuses_valid_prior_stage": True,
        "input": {"path": relative(args.input), "sha256": sha256_path(args.input)},
        "spec": {"path": relative(args.spec), "sha256": sha256_path(args.spec)},
        "source_recovery_sha256": EXPECTED_SOURCE_RECOVERY_SHA256,
        "runner": {"path": relative(RUNNER), "sha256": sha256_path(RUNNER)},
        "output_path": relative(args.output), "checkpoint_path": relative(args.checkpoint),
        "run_manifest_path": relative(args.run_manifest), "cost_output_path": relative(args.cost_output),
    }


def language_tag(language: str) -> str:
    return LANGUAGE_TAGS.get(language.strip().lower(), language.strip().lower() or "text")


def extract_fenced_code(text: str) -> tuple[str, str | None]:
    match = re.search(r"```([A-Za-z0-9_+.#-]+)[ \t]*\r?\n([\s\S]*?)```", text)
    if not match or not match.group(2).strip():
        return "", "EXTRACTION_FAILED_INVALID_OR_EMPTY"
    return match.group(2), None


def render_stage(spec: dict[str, Any], stage: str, source: dict[str, Any],
                 current_code: str, current_critique: str) -> str:
    tag = language_tag(source.get("language", ""))
    if stage == "initial":
        return spec["templates"]["initial"]["template"].format(
            language=source.get("language", ""), language_tag=tag,
            coding_task=source["test_case_prompt"],
        )
    if stage.startswith("critique_"):
        return spec["templates"]["criticism"]["template"].format(current_code=current_code)
    if stage.startswith("improve_"):
        return spec["templates"]["improvement"]["template"].format(
            language_tag=tag, current_critique=current_critique, current_code=current_code,
        )
    raise RuntimeError(f"unknown RCI stage {stage}")


def write_cost(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "prompt_id", "variant", "stage_index", "stage_name", "stage_input_token_count",
        "generated_token_count", "elapsed_seconds", "model_call_index",
        "completion_status", "failure_reason", "final_output_designation",
    ]
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})
    temporary.replace(path)


def validate_existing_rows(rows: list[dict[str, Any]], records: list[dict[str, Any]], stages: list[str]) -> None:
    expected_sequence = [(int(record["prompt_id"]), index, stage)
                         for record in records for index, stage in enumerate(stages)]
    actual = [(int(row["prompt_id"]), int(row["stage_index"]), row["stage_name"]) for row in rows]
    require(actual == expected_sequence[:len(actual)], "RCI checkpoint is not a strict stage-order prefix")


def run_generation(args: argparse.Namespace, pf: dict[str, Any]) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    state = validate_preparation()
    spec = read_json(args.spec)
    stages = list(spec["candidates"][args.variant]["stage_names"])
    condition = {"method": "RCI", "variant": args.variant, "seed": 42,
                 "input_sha256": pf["input"]["sha256"], "spec_sha256": pf["spec"]["sha256"],
                 "source_recovery_sha256": EXPECTED_SOURCE_RECOVERY_SHA256,
                 "runner_sha256": pf["runner"]["sha256"]}
    stage_rows: list[dict[str, Any]] = []
    checkpoint = None
    if args.checkpoint.exists():
        checkpoint = read_json(args.checkpoint)
        require(checkpoint.get("condition") == condition, "checkpoint condition mismatch")
        stage_rows = checkpoint.get("stage_records", [])
        validate_existing_rows(stage_rows, state["records"], stages)

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_CACHE_SNAPSHOT, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_CACHE_SNAPSHOT, torch_dtype=torch.float16,
        device_map="auto", local_files_only=True,
    ).eval()
    if checkpoint:
        restore_rng_state(torch, checkpoint["rng_state"])

    total_rows = EXPECTED_RECORD_COUNT * len(stages)
    run_manifest = {"schema_version": "phase14_rci_run_manifest_v1", "condition": condition,
                    "status": "IN_PROGRESS", "completed_stage_rows": len(stage_rows)}
    atomic_json(args.run_manifest, run_manifest)

    flat_index = 0
    for record in state["records"]:
        source = record_source_prompt_metadata_free(record)
        prompt_rows = [row for row in stage_rows if int(row["prompt_id"]) == int(record["prompt_id"])]
        current_code = ""
        current_critique = ""
        prerequisite_failed = False
        for old in prompt_rows:
            if old["stage_name"] == "initial" or old["stage_name"].startswith("improve_"):
                if old["completion_status"] == "COMPLETED":
                    current_code = old["extracted_code"]
                else:
                    prerequisite_failed = True
            elif old["stage_name"].startswith("critique_"):
                if old["completion_status"] == "COMPLETED":
                    current_critique = old["generated_text"]
                else:
                    prerequisite_failed = True

        for stage_index, stage_name in enumerate(stages):
            if flat_index < len(stage_rows):
                flat_index += 1
                continue
            is_final = stage_name == stages[-1]
            if prerequisite_failed:
                row = {
                    "prompt_id": int(record["prompt_id"]), "variant": args.variant,
                    "stage_index": stage_index, "stage_name": stage_name,
                    "rendered_stage_input": "", "stage_input_token_count": 0,
                    "generated_token_count": 0, "generated_text": "", "extracted_code": "",
                    "elapsed_seconds": 0.0, "model_call_index": None,
                    "completion_status": "SKIPPED_PREREQUISITE_FAILURE",
                    "failure_reason": "A required prior stage failed",
                    "final_output_designation": is_final,
                }
            else:
                rendered = render_stage(spec, stage_name, source, current_code, current_critique)
                inputs = tokenizer(rendered, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
                generated_text = ""
                extracted_code = ""
                failure_reason = None
                completion_status = "FAILED"
                generated_count = 0
                started = time.perf_counter()
                try:
                    with torch.inference_mode():
                        generated = model.generate(
                            **inputs, max_new_tokens=512, temperature=0.2, top_p=0.95,
                            do_sample=True, pad_token_id=tokenizer.pad_token_id,
                        )
                    new_tokens = generated[0, inputs["input_ids"].shape[1]:]
                    generated_count = int(new_tokens.shape[0])
                    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                    if stage_name == "initial" or stage_name.startswith("improve_"):
                        extracted_code, failure_reason = extract_fenced_code(generated_text)
                        completion_status = "COMPLETED" if failure_reason is None else "EXTRACTION_FAILED_INVALID_OR_EMPTY"
                    else:
                        completion_status = "COMPLETED" if generated_text.strip() else "INVALID_OR_EMPTY"
                        if completion_status != "COMPLETED":
                            failure_reason = "Empty criticism output"
                except Exception as exc:
                    failure_reason = f"{type(exc).__name__}: {exc}"
                elapsed = time.perf_counter() - started
                row = {
                    "prompt_id": int(record["prompt_id"]), "variant": args.variant,
                    "stage_index": stage_index, "stage_name": stage_name,
                    "rendered_stage_input": rendered,
                    "stage_input_token_count": int(inputs["input_ids"].shape[1]),
                    "generated_token_count": generated_count, "generated_text": generated_text,
                    "extracted_code": extracted_code, "elapsed_seconds": elapsed,
                    "model_call_index": stage_index + 1, "completion_status": completion_status,
                    "failure_reason": failure_reason, "final_output_designation": is_final,
                }
                if completion_status == "COMPLETED":
                    if stage_name == "initial" or stage_name.startswith("improve_"):
                        current_code = extracted_code
                    else:
                        current_critique = generated_text
                else:
                    prerequisite_failed = True
            stage_rows.append(row)
            flat_index += 1
            checkpoint = {"schema_version": "phase14_rci_checkpoint_v1", "condition": condition,
                          "stage_records": stage_rows, "rng_state": capture_rng_state(torch)}
            atomic_json(args.checkpoint, checkpoint)
            run_manifest["completed_stage_rows"] = len(stage_rows)
            atomic_json(args.run_manifest, run_manifest)

    require(len(stage_rows) == total_rows, "RCI stage-record count incomplete")
    final_rows = [row for row in stage_rows if row["final_output_designation"]]
    require(len(final_rows) == EXPECTED_RECORD_COUNT, "RCI final-output coverage mismatch")
    outputs = [{
        "schema_version": "phase14_rci_final_record_v1", "prompt_id": row["prompt_id"],
        "variant": args.variant, "method": "RCI", "information_tier": "METADATA_FREE",
        "generated_code": row["extracted_code"], "generation_status": row["completion_status"],
        "failure_reason": row["failure_reason"], "seed": 42,
        "input_manifest_sha256": pf["input"]["sha256"], "method_spec_sha256": pf["spec"]["sha256"],
        "source_recovery_sha256": EXPECTED_SOURCE_RECOVERY_SHA256,
        "runner_sha256": pf["runner"]["sha256"],
    } for row in final_rows]
    atomic_json(args.output, {"condition": condition, "stage_records": stage_rows, "final_outputs": outputs})
    write_cost(args.cost_output, stage_rows)
    run_manifest.update({"status": "COMPLETE", "output_sha256": sha256_path(args.output),
                         "cost_sha256": sha256_path(args.cost_output)})
    atomic_json(args.run_manifest, run_manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=["rci1", "rci2"])
    parser.add_argument("--input", type=Path, default=BASELINE_SUBSET)
    parser.add_argument("--spec", type=Path, default=SPEC)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--run-manifest", type=Path)
    parser.add_argument("--cost-output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-output", type=Path)
    args = parser.parse_args()
    defaults = targets(args.variant)
    for name, default in zip(("output", "checkpoint", "run_manifest", "cost_output"), defaults):
        if getattr(args, name) is None:
            setattr(args, name, default)
    for name in ("input", "spec", "output", "checkpoint", "run_manifest", "cost_output"):
        setattr(args, name, getattr(args, name).resolve())
    if args.preflight_output:
        args.preflight_output = args.preflight_output.resolve()
    return args


def main() -> int:
    args = parse_args()
    pf = preflight(args)
    if args.preflight_output:
        atomic_json(args.preflight_output, pf)
    print(json.dumps(pf, indent=2, sort_keys=True))
    if args.preflight_only:
        return 0
    require(not args.output.exists(), "final RCI output exists; immutable review required")
    run_generation(args, pf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
