#!/usr/bin/env python3
"""Run or preflight the frozen Phase 14 CAA-CWE Stage A/B space."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from caa_position_mask import CAAResidualPositionHook
from phase14_common import (
    ACTIVE_CAA_CWES, BASELINE_SUBSET, CAA_LAYERS, CAA_MULTIPLIERS,
    EXPECTED_BASELINE_SUBSET_SHA256, EXPECTED_SOURCE_RECOVERY_SHA256,
    MODEL_CACHE_SNAPSHOT, MODEL_ID, OUTPUTS, REQUIRED_SEED, atomic_json,
    capture_rng_state, read_json, record_source_prompt, relative, require,
    restore_rng_state, sha256_path, validate_preparation,
)


DENSE_REVISION = OUTPUTS / "dense_hook_readiness_revision.json"
SUPPORT_CSV = Path(__file__).resolve().parents[1].parent / "phase1/caa_pair_support.csv"
VECTOR_MANIFEST = OUTPUTS / "caa_vector_manifest.json"
RUNNER = Path(__file__).resolve()
WRAPPER = RUNNER.with_name("caa_position_mask.py")


def condition_targets(stage: str, layer: int, multiplier: float) -> tuple[Path, Path, Path]:
    mult = str(multiplier).replace(".", "p")
    stem = f"caa_stage_{stage.lower()}_layer{layer}_mult{mult}_seed42"
    return (OUTPUTS / f"{stem}.json", OUTPUTS / f"{stem}_checkpoint.json",
            OUTPUTS / f"{stem}_run_manifest.json")


def support_map() -> dict[str, dict[str, str]]:
    import csv
    with SUPPORT_CSV.open("r", encoding="utf-8", newline="") as handle:
        rows = {row["cwe_id"]: row for row in csv.DictReader(handle)}
    expected = {
        "CWE-120": ("SUPPORTED", "PRIMARY_RAW"),
        "CWE-327": ("SUPPORTED", "PRIMARY_MIXED_RAW_AUGMENTED"),
        "CWE-89": ("SUPPORTED", "PRIMARY_RAW"),
        "CWE-338": ("EXPLORATORY_SUPPORT", "EXPLORATORY_SMALL_N_MIXED"),
    }
    for cwe, values in expected.items():
        require((rows[cwe]["support_status"], rows[cwe]["evidence_tier"]) == values,
                f"CAA support status changed for {cwe}")
    return {cwe: rows[cwe] for cwe in ACTIVE_CAA_CWES}


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    state = validate_preparation()
    require(args.seed == REQUIRED_SEED, "seed must be exactly 42")
    require(args.input.resolve() == BASELINE_SUBSET.resolve(), "only frozen 120-prompt subset is allowed")
    require(args.stage in {"A", "B"}, "stage must be A or B")
    dense = read_json(DENSE_REVISION)
    require(dense.get("verification_status") == "PASS" and dense.get("layers_verified") == list(CAA_LAYERS),
            "CAA mask readiness did not pass layers 16/19/23")
    support_map()

    if args.stage == "A":
        allowed = [{"layer": layer, "multiplier": 1.0, "seed": 42} for layer in CAA_LAYERS]
        if args.layer is not None:
            require(args.layer in CAA_LAYERS and float(args.multiplier) == 1.0,
                    "Stage A requires layer 16/19/23 and multiplier 1")
    else:
        allowed = [{"selected_stage_a_layer": "REQUIRED_AT_EXECUTION", "multiplier": value, "seed": 42}
                   for value in CAA_MULTIPLIERS]
        if args.layer is not None:
            require(args.layer in CAA_LAYERS and float(args.multiplier) in CAA_MULTIPLIERS,
                    "Stage B layer/multiplier is outside frozen space")
            require(args.stage_a_selection is not None and args.stage_a_selection.is_file(),
                    "Stage B execution requires frozen Stage A selection")
            selection = read_json(args.stage_a_selection)
            require(selection.get("stage_a", {}).get("selected_layer") == args.layer,
                    "Stage B layer differs from Stage A selection")

    route_counts = {cwe: 0 for cwe in ACTIVE_CAA_CWES}
    unsupported_counts: dict[str, int] = {}
    for record in state["records"]:
        cwe = record_source_prompt(record)["cwe_identifier"]
        if cwe in route_counts:
            route_counts[cwe] += 1
        else:
            unsupported_counts[cwe] = unsupported_counts.get(cwe, 0) + 1
    supported_count = sum(route_counts.values())
    require(supported_count == 90 and sum(unsupported_counts.values()) == 30,
            "CAA routing coverage changed")
    vector_status = "AVAILABLE" if VECTOR_MANIFEST.is_file() else "NOT_CONSTRUCTED"
    if VECTOR_MANIFEST.is_file():
        manifest = read_json(VECTOR_MANIFEST)
        require(manifest.get("status") == "CONSTRUCTED", "CAA vector manifest not constructed")

    return {
        "schema_version": "phase14_caa_runner_preflight_v1", "status": "PASS",
        "stage": args.stage, "allowed_configurations": allowed,
        "seed": 42, "record_count": 120, "information_tier": "ORACLE_CWE",
        "target_cwe_metadata_field": "source_prompt.cwe_identifier",
        "supported_route_prompt_count": supported_count,
        "unsupported_no_intervention_prompt_count": 30,
        "supported_route_counts": route_counts, "unsupported_route_counts": dict(sorted(unsupported_counts.items())),
        "unsupported_policy": "NO_INTERVENTION_EXPLICIT_REASON; never substitute another CWE vector",
        "vector_manifest_status": vector_status,
        "generation_executed": False, "model_loaded": False, "scanner_executed": False,
        "held_out_accessed": False, "security_outcomes_accessed": False,
        "input": {"path": relative(args.input), "sha256": sha256_path(args.input)},
        "source_recovery_sha256": EXPECTED_SOURCE_RECOVERY_SHA256,
        "dense_readiness": {"path": relative(DENSE_REVISION), "sha256": sha256_path(DENSE_REVISION)},
        "wrapper": {"path": relative(WRAPPER), "sha256": sha256_path(WRAPPER)},
        "runner": {"path": relative(RUNNER), "sha256": sha256_path(RUNNER)},
        "output_policy": "one immutable output/checkpoint/run-manifest triple per stage/layer/multiplier",
    }


def load_vectors(torch: Any, manifest: dict[str, Any], layer: int) -> dict[str, Any]:
    vectors = {}
    entries = [entry for entry in manifest.get("entries", []) if int(entry["layer"]) == layer]
    for entry in entries:
        artifact = (Path(__file__).resolve().parents[4] / entry["artifact_path"]).resolve()
        require(artifact.is_file() and sha256_path(artifact) == entry["artifact_sha256"],
                f"CAA vector artifact mismatch for {entry['cwe_id']}")
        payload = torch.load(artifact, map_location="cpu", weights_only=True)
        vectors[entry["cwe_id"]] = payload["prepared_vector"]
    require(set(vectors) == set(ACTIVE_CAA_CWES), "selected layer lacks a supported route vector")
    return vectors


def run_generation(args: argparse.Namespace, pf: dict[str, Any]) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    require(args.layer is not None and args.multiplier is not None, "execution requires layer and multiplier")
    require(VECTOR_MANIFEST.is_file(), "CAA vectors have not been constructed")
    expected = condition_targets(args.stage, args.layer, float(args.multiplier))
    require((args.output, args.checkpoint, args.run_manifest) == tuple(path.resolve() for path in expected),
            "CAA execution path policy mismatch")
    state = validate_preparation()
    vector_manifest = read_json(VECTOR_MANIFEST)
    vectors = load_vectors(torch, vector_manifest, args.layer)
    condition = {"method": "CAA-CWE", "stage": args.stage, "layer": args.layer,
                 "multiplier": float(args.multiplier), "seed": 42,
                 "input_sha256": EXPECTED_BASELINE_SUBSET_SHA256,
                 "vector_manifest_sha256": sha256_path(VECTOR_MANIFEST),
                 "source_recovery_sha256": EXPECTED_SOURCE_RECOVERY_SHA256,
                 "dense_readiness_sha256": sha256_path(DENSE_REVISION),
                 "wrapper_sha256": sha256_path(WRAPPER), "runner_sha256": sha256_path(RUNNER)}
    records: list[dict[str, Any]] = []
    checkpoint = None
    if args.checkpoint.exists():
        require(args.resume, "CAA checkpoint exists; --resume required")
        checkpoint = read_json(args.checkpoint)
        require(checkpoint.get("condition") == condition, "CAA checkpoint condition mismatch")
        records = checkpoint.get("records", [])
        ids = [int(record["prompt_id"]) for record in state["records"]]
        require([int(row["prompt_id"]) for row in records] == ids[:len(records)], "CAA checkpoint order mismatch")

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
    run_manifest = {"schema_version": "phase14_caa_run_manifest_v1", "condition": condition,
                    "status": "IN_PROGRESS", "completed_records": len(records)}
    atomic_json(args.run_manifest, run_manifest)

    for record in state["records"][len(records):]:
        source = record_source_prompt(record)
        cwe = source["cwe_identifier"]
        vector = vectors.get(cwe)
        hook = None
        handle = None
        route_status = "SUPPORTED_VECTOR_APPLIED" if vector is not None else "UNSUPPORTED_ROUTE_NO_INTERVENTION"
        generated_code, failure = "", None
        token_count = 0
        try:
            inputs = tokenizer(source["test_case_prompt"], return_tensors="pt", truncation=True, max_length=1024).to(model.device)
            if vector is not None:
                hook = CAAResidualPositionHook(vector, float(args.multiplier), expected_width=4096)
                handle = model.model.layers[args.layer].register_forward_hook(hook)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=512, temperature=0.2, top_p=0.95,
                                           do_sample=True, pad_token_id=tokenizer.pad_token_id)
            new_tokens = generated[0, inputs["input_ids"].shape[1]:]
            token_count = int(new_tokens.shape[0])
            generated_code = tokenizer.decode(new_tokens, skip_special_tokens=True)
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
        finally:
            if handle is not None:
                handle.remove()
        records.append({
            "schema_version": "phase14_caa_record_v1", "prompt_id": int(record["prompt_id"]),
            "source_index": int(record["source_index"]), "population_type": record["population_type"],
            "prompt_text": source["test_case_prompt"], "prompt_text_sha256": record["prompt_text_sha256"],
            "language": source.get("language", ""), "target_cwe": cwe, "method": "CAA-CWE",
            "information_tier": "ORACLE_CWE", "route_status": route_status,
            "intervention_applied": vector is not None, "no_intervention_reason": None if vector is not None else "NO_SUPPORTED_ROUTE_VECTOR",
            "layer": args.layer if vector is not None else None,
            "multiplier": float(args.multiplier) if vector is not None else None,
            "hook_call_records": hook.call_records if hook is not None else [],
            "generated_code": generated_code, "generated_token_count": token_count,
            "generation_status": "COMPLETED" if failure is None else "FAILED",
            "failure_reason": failure, "empty_status": "EMPTY" if not generated_code.strip() else "NONEMPTY",
            "seed": 42, "condition": condition,
        })
        atomic_json(args.checkpoint, {"schema_version": "phase14_caa_checkpoint_v1",
                    "condition": condition, "records": records, "rng_state": capture_rng_state(torch)})
        run_manifest["completed_records"] = len(records)
        atomic_json(args.run_manifest, run_manifest)
    require(len(records) == 120, "CAA output coverage incomplete")
    atomic_json(args.output, records)
    run_manifest.update({"status": "COMPLETE", "output_sha256": sha256_path(args.output)})
    atomic_json(args.run_manifest, run_manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["A", "B"])
    parser.add_argument("--layer", type=int)
    parser.add_argument("--multiplier", type=float)
    parser.add_argument("--stage-a-selection", type=Path)
    parser.add_argument("--input", type=Path, default=BASELINE_SUBSET)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--run-manifest", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-output", type=Path)
    args = parser.parse_args()
    args.input = args.input.resolve()
    if args.stage_a_selection:
        args.stage_a_selection = args.stage_a_selection.resolve()
    if args.layer is not None and args.multiplier is None:
        args.multiplier = 1.0 if args.stage == "A" else None
    if args.layer is not None and args.multiplier is not None:
        defaults = condition_targets(args.stage, args.layer, args.multiplier)
        for name, default in zip(("output", "checkpoint", "run_manifest"), defaults):
            if getattr(args, name) is None:
                setattr(args, name, default)
    for name in ("output", "checkpoint", "run_manifest"):
        if getattr(args, name) is not None:
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
    require(args.output is not None and not args.output.exists(), "CAA output path missing or already exists")
    run_generation(args, pf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
