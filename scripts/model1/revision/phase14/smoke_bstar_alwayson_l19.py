#!/usr/bin/env python3
"""Run or preflight the frozen metadata-free B*-AlwaysOn-L19 smoke."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

from phase14_common import (
    GENERATION_SETTINGS, MODEL_CACHE_SNAPSHOT, MODEL_ID, OUTPUTS, REQUIRED_SEED,
    atomic_json, capture_rng_state, read_json, relative, require,
    restore_rng_state, sha256_path, sha256_text, validate_preparation,
)


MANIFEST = OUTPUTS / "bstar_alwayson_l19_smoke_manifest.json"
EXPECTED_MANIFEST_SHA256 = "3ad3dfbd5b034d3e2f4531eb399b87fae96d624b8b3dbacf4a8053425c848000"
EXPECTED_IDS = [1418, 1548, 153, 1632, 74, 855, 1860, 578, 887, 1307,
                779, 1094, 1495, 137, 1405, 1403, 269, 1836, 654, 288]
LAYER = 19
FEATURES = (14193, 16897, 11462, 151)
ALPHA = 40.0
SAE_RELEASE = "llama_scope_lxr_8x"
SAE_ID = "l19r_8x"
SAE_CACHE_SNAPSHOT = "8dbc1d85edfced43081c03c38b05514dbab1368b"
OUTPUT = OUTPUTS / "bstar_alwayson_l19_smoke_outputs.json"
CHECKPOINT = OUTPUTS / "bstar_alwayson_l19_smoke_checkpoint.json"
RUN_MANIFEST = OUTPUTS / "bstar_alwayson_l19_smoke_run_manifest.json"
RUNNER = Path(__file__).resolve()


def consumed_string_fields() -> set[str]:
    """Statically enumerate literal dictionary fields consumed by this runner."""
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"), filename=str(RUNNER))
    fields: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            fields.add(node.slice.value)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get" and node.args
                and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
            fields.add(node.args[0].value)
    return fields


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    validate_preparation()
    require(args.seed == REQUIRED_SEED, "AlwaysOn seed must be exactly 42")
    require(args.manifest.resolve() == MANIFEST.resolve(), "only the frozen AlwaysOn manifest is allowed")
    require(args.output.resolve() == OUTPUT.resolve(), "AlwaysOn output path policy mismatch")
    require(args.checkpoint.resolve() == CHECKPOINT.resolve(), "AlwaysOn checkpoint path policy mismatch")
    require(args.run_manifest.resolve() == RUN_MANIFEST.resolve(), "AlwaysOn run-manifest path policy mismatch")
    require(args.manifest.is_file() and sha256_path(args.manifest) == EXPECTED_MANIFEST_SHA256,
            "AlwaysOn manifest hash mismatch")
    manifest = read_json(args.manifest)
    require(manifest.get("schema_version") == "phase14_bstar_alwayson_smoke_manifest_v1",
            "AlwaysOn manifest schema mismatch")
    require(manifest.get("information_tier") == "METADATA_FREE", "AlwaysOn information tier mismatch")
    require(manifest.get("prompt_ids_execution_order") == EXPECTED_IDS,
            "AlwaysOn exact ID/order mismatch")
    require(manifest.get("selection_size") == 20 and len(set(EXPECTED_IDS)) == 20,
            "AlwaysOn ID cardinality mismatch")
    require(manifest["verification"]["held_out_overlap_count"] == 0,
            "AlwaysOn manifest held-out overlap")
    require(manifest.get("selected_after_alwayson_outcomes") is False,
            "AlwaysOn population was selected after outcomes")
    require(manifest["historical_provenance"]["historical_phase8_generated_outputs"] == "NOT_REUSED",
            "historical Phase 8 outputs were reused")
    require(manifest["historical_provenance"]["historical_phase8_steering_result"] == "NOT_REUSED",
            "historical Phase 8 steering was reused")
    require(manifest["historical_provenance"]["historical_phase8_scanner_result"] == "NOT_USED_AS_ALWAYSON_RESULT",
            "historical Phase 8 scan was reused")
    prompts = manifest.get("prompts")
    require(isinstance(prompts, list) and len(prompts) == 20,
            "AlwaysOn prompt records missing")
    require([int(row["prompt_id"]) for row in prompts] == EXPECTED_IDS,
            "AlwaysOn prompt record order mismatch")
    require(all(sha256_text(row["prompt_text"]) == row["prompt_text_sha256"] for row in prompts),
            "AlwaysOn prompt text hash mismatch")

    fields = consumed_string_fields()
    forbidden = {"cwe_identifier", "cwe_id", "cwe"}
    consumed_forbidden = sorted(fields & forbidden)
    require(consumed_forbidden == [], "AlwaysOn runner consumes a CWE routing field")
    if args.checkpoint.exists():
        require(args.resume, "AlwaysOn checkpoint exists; --resume required")
        checkpoint = read_json(args.checkpoint)
        require(checkpoint.get("condition", {}).get("manifest_sha256") == EXPECTED_MANIFEST_SHA256,
                "AlwaysOn checkpoint manifest mismatch")
        rows = checkpoint.get("records", [])
        require([int(row["prompt_id"]) for row in rows] == EXPECTED_IDS[:len(rows)],
                "AlwaysOn checkpoint is not a frozen-order prefix")
        require(isinstance(checkpoint.get("rng_state"), dict), "AlwaysOn checkpoint RNG missing")
    else:
        require(not args.resume, "--resume supplied without AlwaysOn checkpoint")

    return {
        "schema_version": "phase14_bstar_alwayson_preflight_v1",
        "status": "PASS", "generation_executed": False, "model_loaded": False,
        "scanner_executed": False, "held_out_accessed": False,
        "manifest": {"path": relative(args.manifest), "sha256": EXPECTED_MANIFEST_SHA256},
        "prompt_count": 20, "unique_prompt_count": 20, "held_out_overlap_count": 0,
        "prompt_ids_execution_order": EXPECTED_IDS,
        "condition": {"method": "B*-AlwaysOn-L19", "information_tier": "METADATA_FREE",
                      "layer": LAYER, "features": list(FEATURES),
                      "alpha_by_feature": {str(feature): ALPHA for feature in FEATURES},
                      "simultaneous": True, "seed": 42},
        "routing_proof": {"routing_policy": "CONSTANT_NO_CONDITIONAL_ROUTE",
                          "routing_fields_consumed": [], "forbidden_cwe_field_access_count": 0,
                          "manifest_prompt_fields_consumed": ["prompt_id", "execution_order", "development_source_index",
                                                              "prompt_text", "prompt_text_sha256", "language", "file_path"],
                          "static_literal_field_inventory_sha256": sha256_text("\n".join(sorted(fields)))},
        "runner": {"path": relative(RUNNER), "sha256": sha256_path(RUNNER)},
        "output_path": relative(args.output), "checkpoint_path": relative(args.checkpoint),
        "run_manifest_path": relative(args.run_manifest),
    }


def make_hook(sae: Any, call_records: list[dict[str, Any]]):
    def hook_fn(module: Any, inputs: tuple[Any, ...], output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        latent = sae.encode(hidden)
        for feature in FEATURES:
            latent[..., feature] += ALPHA
        edited = sae.decode(latent).to(hidden.dtype)
        call_records.append({"call_index": len(call_records), "shape": list(hidden.shape),
                             "dtype": str(hidden.dtype), "device": str(hidden.device),
                             "layer": LAYER, "features": list(FEATURES),
                             "alpha_by_feature": {str(feature): ALPHA for feature in FEATURES},
                             "all_features_simultaneous": True})
        if isinstance(output, tuple):
            return (edited,) + output[1:]
        return edited
    return hook_fn


def run_generation(args: argparse.Namespace, pf: dict[str, Any]) -> None:
    import torch
    from sae_lens import SAE
    from transformers import AutoModelForCausalLM, AutoTokenizer

    manifest = read_json(args.manifest)
    condition = {**pf["condition"], "manifest_sha256": EXPECTED_MANIFEST_SHA256,
                 "runner_sha256": pf["runner"]["sha256"],
                 "model_id": MODEL_ID, "model_cache_snapshot": MODEL_CACHE_SNAPSHOT,
                 "sae_release": SAE_RELEASE, "sae_id": SAE_ID,
                 "sae_cache_snapshot": SAE_CACHE_SNAPSHOT,
                 "generation_settings": GENERATION_SETTINGS}
    records: list[dict[str, Any]] = []
    checkpoint = None
    if args.checkpoint.exists():
        checkpoint = read_json(args.checkpoint)
        require(checkpoint.get("condition") == condition, "AlwaysOn checkpoint condition mismatch")
        records = checkpoint.get("records", [])

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    require(torch.cuda.is_available(), "CUDA is required for AlwaysOn smoke generation")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_CACHE_SNAPSHOT, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_CACHE_SNAPSHOT, local_files_only=True,
        torch_dtype=torch.float16, device_map="auto",
    ).eval()
    sae, _, _ = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID)
    sae = sae.to("cuda").eval()
    if checkpoint:
        restore_rng_state(torch, checkpoint["rng_state"])
    run_record = {"schema_version": "phase14_bstar_alwayson_run_manifest_v1",
                  "condition": condition, "status": "IN_PROGRESS",
                  "completed_records": len(records), "routing_fields_consumed": []}
    atomic_json(args.run_manifest, run_record)

    for prompt in manifest["prompts"][len(records):]:
        generated_code, failure, generated_count = "", None, 0
        calls: list[dict[str, Any]] = []
        handle = None
        try:
            inputs = tokenizer(prompt["prompt_text"], return_tensors="pt", truncation=True,
                               max_length=GENERATION_SETTINGS["input_truncation"]).to(model.device)
            handle = model.model.layers[LAYER].register_forward_hook(make_hook(sae, calls))
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, max_new_tokens=GENERATION_SETTINGS["max_new_tokens"],
                    temperature=GENERATION_SETTINGS["temperature"], top_p=GENERATION_SETTINGS["top_p"],
                    do_sample=True, pad_token_id=tokenizer.pad_token_id,
                )
            new_tokens = generated[0, inputs["input_ids"].shape[1]:]
            generated_count = int(new_tokens.shape[0])
            generated_code = tokenizer.decode(new_tokens, skip_special_tokens=True)
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
        finally:
            if handle is not None:
                handle.remove()
        records.append({
            "schema_version": "phase14_bstar_alwayson_record_v1",
            "prompt_id": int(prompt["prompt_id"]), "execution_order": int(prompt["execution_order"]),
            "development_source_index": int(prompt["development_source_index"]),
            "prompt_text": prompt["prompt_text"], "prompt_text_sha256": prompt["prompt_text_sha256"],
            "language": prompt["language"], "method": "B*-AlwaysOn-L19",
            "information_tier": "METADATA_FREE", "routing_policy": "CONSTANT_NO_CONDITIONAL_ROUTE",
            "routing_fields_consumed": [], "layer": LAYER, "features": list(FEATURES),
            "alpha_by_feature": {str(feature): ALPHA for feature in FEATURES},
            "all_features_simultaneous": True, "hook_calls": calls,
            "generated_code": generated_code, "generated_token_count": generated_count,
            "generation_status": "COMPLETED" if failure is None else "FAILED",
            "failure_reason": failure, "empty_status": "EMPTY" if not generated_code.strip() else "NONEMPTY",
            "seed": 42, "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "runner_sha256": pf["runner"]["sha256"],
        })
        atomic_json(args.checkpoint, {"schema_version": "phase14_bstar_alwayson_checkpoint_v1",
                    "condition": condition, "records": records, "rng_state": capture_rng_state(torch)})
        run_record["completed_records"] = len(records)
        atomic_json(args.run_manifest, run_record)
    require(len(records) == 20, "AlwaysOn smoke output coverage incomplete")
    atomic_json(args.output, records)
    run_record.update({"status": "COMPLETE", "output_sha256": sha256_path(args.output)})
    atomic_json(args.run_manifest, run_record)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--run-manifest", type=Path, default=RUN_MANIFEST)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-output", type=Path)
    args = parser.parse_args()
    for name in ("manifest", "output", "checkpoint", "run_manifest"):
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
    require(not args.output.exists(), "AlwaysOn final output exists; immutable review required")
    run_generation(args, pf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
