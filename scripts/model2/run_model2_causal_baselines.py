#!/usr/bin/env python
"""Freeze and generate the 45 unsteered Model2 causal-screen baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
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
INPUT_MANIFEST = OUT / "model2_causal_baseline_input_manifest.json"
OUTPUT = OUT / "model2_causal_baseline_generations.json"
GENERATION_MANIFEST = OUT / "model2_causal_baseline_generation_manifest.json"
CHECKPOINT = OUT / "model2_causal_baseline_generation_checkpoint.json"
RUN_MANIFEST = OUT / "model2_causal_baseline_generation_run_manifest.json"

MODEL_ID = "google/gemma-2-9b-it"
MODEL_REVISION = "11c9b309abf73637e4b6f9a3fa1e92e615547819"
SEED = 42
GENERATION = {
    "max_new_tokens": 256,
    "temperature": 0.2,
    "top_p": 0.95,
    "do_sample": True,
    "seed": SEED,
    "seed_reset": "torch.manual_seed(42) and torch.cuda.manual_seed_all(42) immediately before every individual generation",
    "decode_generated_tokens_only": True,
    "skip_special_tokens": True,
}
UNSAFE = {
    "CWE-120": [133, 105, 122, 18, 27, 207, 214, 115, 220, 166, 31, 62, 38, 470, 44],
    "CWE-327": [1866, 1752, 1903, 1163, 1901, 1860, 1846, 1832, 1879, 1566, 1191, 1913, 1817, 1783, 1876],
    "CWE-89": [1688, 1748, 1615, 1847, 1850, 1603, 1823, 1695, 1643, 1704],
}
SAFE = [1121, 144, 179, 1841, 1853]
FROZEN = {
    "model2_causal_candidate_manifest_model1_fidelity.json": "be072afa3a9c58c20729bcd5deb343007f15b7c981e4fe02bb857bc17d00b401",
    "model2_causal_validation_protocol.json": "b3aac3b21f52270c231e1c077adeb0cc1ef2a2ebce1f156c08a13df2695d414e",
    "model2_causal_protocol_checkpoint.json": "099759f3b128ca8d3079d81c71fe24fd65be115ec0e3b270e0496b95357e1f33",
}
VALIDITY_MARKERS = ["def ", "int ", "char ", "return", "if ", "for ", "{", "("]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_text(raw)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def frozen_artifacts() -> list[dict[str, Any]]:
    result = []
    for name, expected in FROZEN.items():
        path = OUT / name
        require(path.is_file(), f"missing frozen artifact: {path}")
        actual = sha256_file(path)
        require(actual == expected, f"frozen artifact drift: {name}: {actual}")
        result.append({
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": actual,
            "preserved_unchanged": True,
        })
    return result


def logical_entries() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    logical_index = 0
    for target_cwe, ids in UNSAFE.items():
        for prompt_id in ids:
            entries.append({
                "logical_index": logical_index,
                "prompt_id": prompt_id,
                "population": "UNSAFE",
                "target_cwe": target_cwe,
            })
            logical_index += 1
    for prompt_id in SAFE:
        entries.append({
            "logical_index": logical_index,
            "prompt_id": prompt_id,
            "population": "SAFE",
            "target_cwe": None,
        })
        logical_index += 1
    return entries


def load_tokenizer() -> Any:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def freeze_inputs() -> dict[str, Any]:
    require(not INPUT_MANIFEST.exists(), f"immutable input manifest already exists: {INPUT_MANIFEST}")
    require(not OUTPUT.exists(), "generation output already exists")
    artifacts = frozen_artifacts()
    require(DEV.is_file(), "development source is missing")
    rows = json.loads(DEV.read_text(encoding="utf-8"))
    require(len(rows) == 1341, "unexpected development source cardinality")
    by_id: dict[int, tuple[int, dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        prompt_id = int(row["prompt_id"])
        require(prompt_id not in by_id, f"duplicate development source prompt ID: {prompt_id}")
        by_id[prompt_id] = (index, row)

    logical = logical_entries()
    expected_ids = [x["prompt_id"] for x in logical]
    unique_ids = list(dict.fromkeys(expected_ids))
    require(len(logical) == 45, "logical prompt count must be 45")
    require(len(unique_ids) == 45, "unexpected duplicate prompt ID in approved logical populations")
    require(all(x in by_id for x in unique_ids), "approved ID absent from development source")

    protocol = json.loads((OUT / "model2_causal_validation_protocol.json").read_text(encoding="utf-8"))
    qualified_by_key: dict[tuple[str, int], str] = {}
    for target_cwe, prompt_rows in protocol["unsafe_prompts"].items():
        for item in prompt_rows:
            qualified_by_key[(target_cwe, int(item["prompt_id"]))] = item["prompt_sha256"]
    safe_hashes = {int(x["prompt_id"]): x["prompt_sha256"] for x in protocol["safe_corruption_prompts"]}

    tokenizer = load_tokenizer()
    chat_template = tokenizer.chat_template
    chat_template_identity = chat_template if isinstance(chat_template, str) else json.dumps(chat_template, sort_keys=True, ensure_ascii=False)
    physical_records: list[dict[str, Any]] = []
    for physical_index, prompt_id in enumerate(unique_ids):
        source_index, row = by_id[prompt_id]
        source_prompt = row["test_case_prompt"]
        source_hash = sha256_text(source_prompt)
        mappings = [x for x in logical if x["prompt_id"] == prompt_id]
        for mapping in mappings:
            if mapping["population"] == "UNSAFE":
                require(
                    qualified_by_key.get((mapping["target_cwe"], prompt_id)) == source_hash,
                    f"approved unsafe prompt provenance mismatch: {mapping['target_cwe']} {prompt_id}",
                )
            else:
                require(safe_hashes.get(prompt_id) == source_hash, f"approved safe prompt provenance mismatch: {prompt_id}")
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": source_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        tokenized = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        physical_records.append({
            "physical_index": physical_index,
            "prompt_id": prompt_id,
            "source_index": source_index,
            "source_cwe": row.get("cwe_identifier", ""),
            "language": str(row.get("language", "")).lower(),
            "source_prompt_sha256": source_hash,
            "rendered_prompt_sha256": sha256_text(rendered),
            "rendered_input_token_count": int(tokenized["input_ids"].shape[-1]),
            "logical_mappings": mappings,
        })

    manifest = {
        "schema_version": "phase21_model2_causal_baseline_input_manifest_v1",
        "status": "FROZEN_BEFORE_MODEL_LOAD",
        "frozen_at_utc": now_utc(),
        "qualification": "protocol-defined pre-causal baseline qualification before any SAE steering",
        "preserved_artifacts": artifacts,
        "protocol_amendment": {
            "scope": "unsteered causal-baseline qualification only",
            "reason": "qualify the frozen prompts under the 256-new-token causal-screen configuration",
            "prompt_rendering": "validated Gemma apply_chat_template path from accepted Model2 B0 generation",
            "prior_frozen_protocol_modified": False,
            "causal_candidates_modified": False,
            "screening_alpha_modified": False,
        },
        "source": {
            "path": str(DEV.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(DEV),
            "record_count": len(rows),
            "development_only": True,
            "heldout_accessed": False,
        },
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "dtype": "bfloat16",
            "device": "cuda:0",
            "quantization": False,
            "offload": False,
        },
        "tokenizer": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "class": tokenizer.__class__.__name__,
            "chat_template_sha256": sha256_text(chat_template_identity),
            "apply_chat_template": True,
            "add_generation_prompt": True,
            "add_special_tokens_during_tensorization": False,
        },
        "generation": GENERATION,
        "intervention": {
            "sae_loaded": False,
            "hook_registered": False,
            "feature_id": None,
            "alpha": 0.0,
        },
        "logical_record_count": len(logical),
        "physical_record_count": len(physical_records),
        "duplicate_physical_generation_count": 0,
        "logical_entries_sha256": canonical_sha256(logical),
        "physical_prompt_ids_sha256": canonical_sha256(unique_ids),
        "logical_entries": logical,
        "physical_records": physical_records,
        "runner": {
            "path": str(RUNNER.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(RUNNER),
        },
        "model_loaded": False,
        "generation_run": False,
        "scanner_run": False,
    }
    atomic_json(INPUT_MANIFEST, manifest)
    return {"status": manifest["status"], "input_manifest_sha256": sha256_file(INPUT_MANIFEST), "logical": 45, "physical": 45}


def validate_inputs(resume: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frozen_artifacts()
    require(INPUT_MANIFEST.is_file(), "frozen baseline input manifest is missing; run --freeze-inputs")
    manifest = json.loads(INPUT_MANIFEST.read_text(encoding="utf-8"))
    require(manifest.get("status") == "FROZEN_BEFORE_MODEL_LOAD", "input manifest status mismatch")
    require(manifest["runner"]["sha256"] == sha256_file(RUNNER), "runner changed after input freeze")
    require(manifest["source"]["sha256"] == sha256_file(DEV), "development source drift")
    require(manifest["model"] == {"id": MODEL_ID, "revision": MODEL_REVISION, "dtype": "bfloat16", "device": "cuda:0", "quantization": False, "offload": False}, "model freeze mismatch")
    require(manifest["generation"] == GENERATION, "generation settings drift")
    require(manifest["logical_record_count"] == 45 and manifest["physical_record_count"] == 45, "baseline population cardinality mismatch")
    require([x["prompt_id"] for x in manifest["logical_entries"]] == [x["prompt_id"] for x in logical_entries()], "logical prompt order drift")
    require(not OUTPUT.exists(), "immutable final generation output already exists")
    records: list[dict[str, Any]] = []
    if CHECKPOINT.exists():
        require(resume, "generation checkpoint exists; rerun with --resume")
        checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
        require(checkpoint["input_manifest_sha256"] == sha256_file(INPUT_MANIFEST), "checkpoint input manifest drift")
        require(checkpoint["runner_sha256"] == sha256_file(RUNNER), "checkpoint runner drift")
        records = checkpoint["records"]
        expected_prefix = [x["prompt_id"] for x in manifest["physical_records"][:len(records)]]
        require([x["prompt_id"] for x in records] == expected_prefix, "checkpoint is not an approved physical-order prefix")
    else:
        require(not resume, "--resume supplied without a checkpoint")
    return manifest, records


def preflight(resume: bool) -> dict[str, Any]:
    manifest, records = validate_inputs(resume)
    return {
        "schema_version": "phase21_model2_causal_baseline_preflight_v1",
        "status": "PASS_APPROVED",
        "model_loaded": False,
        "generation_run": False,
        "scanner_run": False,
        "sae_loaded": False,
        "hook_registered": False,
        "alpha": 0.0,
        "logical_record_count": manifest["logical_record_count"],
        "physical_record_count": manifest["physical_record_count"],
        "completed_checkpoint_records": len(records),
        "resume_required": bool(records),
        "input_manifest_sha256": sha256_file(INPUT_MANIFEST),
        "runner_sha256": sha256_file(RUNNER),
        "heldout_accessed": False,
    }


def validity(text: str) -> dict[str, Any]:
    stripped = text.strip()
    marker = next((x for x in VALIDITY_MARKERS if x in text), None)
    valid = len(stripped) >= 10 and marker is not None
    return {
        "is_valid": valid,
        "stripped_character_count": len(stripped),
        "minimum_character_count": 10,
        "matched_marker": marker,
        "required_markers": VALIDITY_MARKERS,
    }


def generate(resume: bool) -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    require(torch.cuda.is_available(), "CUDA is unavailable")
    manifest, records = validate_inputs(resume)
    tokenizer = load_tokenizer()
    dev_rows = json.loads(DEV.read_text(encoding="utf-8"))

    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": "cuda:0"},
        local_files_only=True,
    ).eval()
    parameter_devices = sorted({str(x.device) for x in model.parameters()})
    require(parameter_devices == ["cuda:0"], f"unauthorized device/offload state: {parameter_devices}")
    require(not hasattr(model, "_hf_hook"), "unexpected accelerate/offload hook present")

    started = time.perf_counter()
    prior_elapsed = 0.0
    if CHECKPOINT.exists():
        prior_elapsed = float(json.loads(CHECKPOINT.read_text(encoding="utf-8")).get("elapsed_seconds", 0.0))
    run_manifest = {
        "schema_version": "phase21_model2_causal_baseline_run_manifest_v1",
        "status": "IN_PROGRESS",
        "input_manifest_sha256": sha256_file(INPUT_MANIFEST),
        "runner_sha256": sha256_file(RUNNER),
        "completed_records": len(records),
        "physical_record_count": 45,
        "model_load_seconds": time.perf_counter() - load_started,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "parameter_devices": parameter_devices,
        "sae_loaded": False,
        "hook_registered": False,
        "hook_call_count": 0,
        "alpha": 0.0,
    }
    atomic_json(RUN_MANIFEST, run_manifest)

    for physical in manifest["physical_records"][len(records):]:
        prompt_id = int(physical["prompt_id"])
        row = dev_rows[int(physical["source_index"])]
        require(int(row["prompt_id"]) == prompt_id, f"source index drift for prompt {prompt_id}")
        source_prompt = row["test_case_prompt"]
        require(sha256_text(source_prompt) == physical["source_prompt_sha256"], f"source prompt drift for {prompt_id}")
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": source_prompt}], tokenize=False, add_generation_prompt=True
        )
        require(sha256_text(rendered) == physical["rendered_prompt_sha256"], f"rendered prompt drift for {prompt_id}")
        inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to("cuda:0")
        input_token_count = int(inputs["input_ids"].shape[-1])
        require(input_token_count == int(physical["rendered_input_token_count"]), f"tokenization drift for {prompt_id}")

        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        generated_text = ""
        generated_token_count = 0
        generation_status = "FAILED"
        stop_reason = "FAILURE"
        exception = None
        record_started_utc = now_utc()
        record_started = time.perf_counter()
        try:
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=GENERATION["max_new_tokens"],
                    temperature=GENERATION["temperature"],
                    top_p=GENERATION["top_p"],
                    do_sample=GENERATION["do_sample"],
                    pad_token_id=tokenizer.pad_token_id,
                )
            generated = output[0, input_token_count:]
            generated_token_count = int(generated.shape[0])
            generated_text = tokenizer.decode(generated, skip_special_tokens=True)
            eos = model.generation_config.eos_token_id
            eos_ids = set(eos if isinstance(eos, list) else [eos])
            last_token = int(generated[-1].item()) if generated_token_count else None
            if last_token in eos_ids:
                stop_reason = "EOS_TOKEN"
            elif generated_token_count >= GENERATION["max_new_tokens"]:
                stop_reason = "MAX_NEW_TOKENS"
            else:
                stop_reason = "MODEL_STOP_OTHER"
            generation_status = "SUCCESS"
        except Exception as exc:
            exception = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            torch.cuda.empty_cache()

        record = {
            "schema_version": "phase21_model2_causal_baseline_record_v1",
            "physical_index": int(physical["physical_index"]),
            "prompt_id": prompt_id,
            "source_index": int(physical["source_index"]),
            "source_cwe": physical["source_cwe"],
            "language": physical["language"],
            "logical_mappings": physical["logical_mappings"],
            "population_roles": sorted({x["population"] for x in physical["logical_mappings"]}),
            "target_cwes": [x["target_cwe"] for x in physical["logical_mappings"] if x["target_cwe"]],
            "source_prompt_sha256": physical["source_prompt_sha256"],
            "rendered_prompt_sha256": physical["rendered_prompt_sha256"],
            "input_token_count": input_token_count,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "tokenizer_id": MODEL_ID,
            "tokenizer_revision": MODEL_REVISION,
            "tokenizer_class": manifest["tokenizer"]["class"],
            "chat_template_sha256": manifest["tokenizer"]["chat_template_sha256"],
            "chat_template_used": True,
            "generation_settings": GENERATION,
            "seed": SEED,
            "seed_reset_immediately_before_generation": True,
            "generated_text": generated_text,
            "generated_code": generated_text,
            "generated_text_sha256": sha256_text(generated_text),
            "generated_token_count": generated_token_count,
            "stop_reason": stop_reason,
            "generation_status": generation_status,
            "exception": exception,
            "empty_or_whitespace": not bool(generated_text.strip()),
            "validity": validity(generated_text),
            "sae_loaded": False,
            "intervention_hook_registered": False,
            "intervention_hook_call_count": 0,
            "feature_id": None,
            "alpha": 0.0,
            "record_started_utc_diagnostic": record_started_utc,
            "runtime_seconds_diagnostic": time.perf_counter() - record_started,
        }
        records.append(record)
        elapsed = prior_elapsed + time.perf_counter() - started
        atomic_json(CHECKPOINT, {
            "schema_version": "phase21_model2_causal_baseline_generation_checkpoint_v1",
            "status": "IN_PROGRESS" if len(records) < 45 else "GENERATION_COMPLETE_PENDING_FINALIZATION",
            "input_manifest_sha256": sha256_file(INPUT_MANIFEST),
            "runner_sha256": sha256_file(RUNNER),
            "records": records,
            "elapsed_seconds": elapsed,
            "hook_call_count": 0,
            "alpha": 0.0,
        })
        run_manifest.update({
            "completed_records": len(records),
            "success_count": sum(x["generation_status"] == "SUCCESS" for x in records),
            "failure_count": sum(x["generation_status"] == "FAILED" for x in records),
            "elapsed_seconds": elapsed,
        })
        atomic_json(RUN_MANIFEST, run_manifest)
        print(f"CAUSAL_BASELINE {len(records)}/45 failures={run_manifest['failure_count']} elapsed_min={elapsed/60:.1f}", flush=True)

    expected_ids = [x["prompt_id"] for x in manifest["physical_records"]]
    require([x["prompt_id"] for x in records] == expected_ids, "final physical prompt order mismatch")
    require(all(x["intervention_hook_call_count"] == 0 for x in records), "nonzero intervention hook call")
    output = {
        "schema_version": "phase21_model2_causal_baseline_generations_v1",
        "status": "COMPLETE_WITH_PRESERVED_FAILURES" if any(x["generation_status"] == "FAILED" for x in records) else "COMPLETE",
        "input_manifest_sha256": sha256_file(INPUT_MANIFEST),
        "runner_sha256": sha256_file(RUNNER),
        "logical_record_count": 45,
        "physical_record_count": 45,
        "records": records,
    }
    atomic_json(OUTPUT, output)
    output_hash = sha256_file(OUTPUT)
    generation_manifest = {
        "schema_version": "phase21_model2_causal_baseline_generation_manifest_v1",
        "status": output["status"],
        "input_manifest": {"path": str(INPUT_MANIFEST.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256_file(INPUT_MANIFEST)},
        "generation_output": {"path": str(OUTPUT.relative_to(ROOT)).replace("\\", "/"), "sha256": output_hash},
        "runner": {"path": str(RUNNER.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256_file(RUNNER)},
        "logical_record_count": 45,
        "physical_record_count": 45,
        "logical_entries_accounted_for": sum(len(x["logical_mappings"]) for x in records) == 45,
        "unexpected_prompt_ids": [],
        "duplicate_physical_generation_count": 0,
        "physical_prompt_order": expected_ids,
        "success_count": sum(x["generation_status"] == "SUCCESS" for x in records),
        "failure_count": sum(x["generation_status"] == "FAILED" for x in records),
        "valid_count": sum(x["validity"]["is_valid"] for x in records),
        "invalid_count": sum(not x["validity"]["is_valid"] for x in records),
        "seed_reset_per_prompt": all(x["seed_reset_immediately_before_generation"] for x in records),
        "zero_intervention_hook_calls": all(x["intervention_hook_call_count"] == 0 for x in records),
        "sae_loaded": False,
        "alpha_values": [0.0],
        "scanner_run": False,
        "heldout_accessed": False,
        "model1_modified": False,
        "model": manifest["model"],
        "tokenizer": manifest["tokenizer"],
        "generation": manifest["generation"],
        "completed_at_utc_diagnostic": now_utc(),
    }
    atomic_json(GENERATION_MANIFEST, generation_manifest)
    run_manifest.update({
        "status": "COMPLETE",
        "generation_output_sha256": output_hash,
        "generation_manifest_sha256": sha256_file(GENERATION_MANIFEST),
        "completed_records": 45,
    })
    atomic_json(RUN_MANIFEST, run_manifest)
    print(json.dumps({
        "status": generation_manifest["status"],
        "generation_output_sha256": output_hash,
        "generation_manifest_sha256": sha256_file(GENERATION_MANIFEST),
        "success_count": generation_manifest["success_count"],
        "failure_count": generation_manifest["failure_count"],
        "valid_count": generation_manifest["valid_count"],
        "invalid_count": generation_manifest["invalid_count"],
    }, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-inputs", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    require(sum([args.freeze_inputs, args.preflight_only]) <= 1, "choose at most one of --freeze-inputs/--preflight-only")
    if args.freeze_inputs:
        print(json.dumps(freeze_inputs(), indent=2))
        return 0
    pf = preflight(args.resume)
    print(json.dumps(pf, indent=2))
    if args.preflight_only:
        return 0
    generate(args.resume)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"PHASE21_CAUSAL_BASELINE_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
