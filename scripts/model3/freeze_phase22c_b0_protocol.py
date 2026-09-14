#!/usr/bin/env python
"""Freeze the approved source-matched Model3 22C B0 protocol."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[4]
DEV = ROOT / "data/cyberseceval/dev_prompts.json"
OUT = ROOT / "revision/model3/phase22/outputs"
POPULATION = OUT / "phase22c_source_population.json"
PROTOCOL = OUT / "phase22c_b0_protocol.json"
RUNNER = Path(__file__).with_name("run_phase22c_b0.py")
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
ELIGIBLE = {("CWE-120", "c"), ("CWE-120", "cpp"), ("CWE-327", "java"), ("CWE-89", "python")}
EXPECTED_STRATA = {"CWE-120|c": 49, "CWE-120|cpp": 61, "CWE-327|java": 46, "CWE-89|python": 24}


def model_snapshot_path() -> Path:
    path = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots" / MODEL_REVISION
    if not path.is_dir() or not (path / "config.json").is_file() or not (path / "tokenizer_config.json").is_file():
        raise RuntimeError(f"pinned local model snapshot is incomplete: {path}")
    return path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    if PROTOCOL.exists():
        raise RuntimeError("22C protocol already frozen")
    if not RUNNER.is_file():
        raise RuntimeError(f"runner missing: {RUNNER}")
    rows = json.loads(DEV.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != 1341:
        raise RuntimeError("development source must contain exactly 1341 prompts")
    if len({int(x["prompt_id"]) for x in rows}) != 1341:
        raise RuntimeError("development prompt IDs are not unique")
    selected = []
    excluded_matching_cwe = Counter()
    for source_index, row in enumerate(rows):
        cwe = row.get("cwe_identifier", "")
        language = row.get("language", "").lower()
        if cwe not in {"CWE-120", "CWE-327", "CWE-89"}:
            continue
        if (cwe, language) not in ELIGIBLE:
            excluded_matching_cwe[f"{cwe}|{language}"] += 1
            continue
        prompt = row["test_case_prompt"]
        selected.append({
            "population_index": len(selected), "source_index": source_index,
            "prompt_id": int(row["prompt_id"]), "cwe_identifier": cwe,
            "language": language, "test_case_prompt": prompt,
            "source_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "scanner_eligible_by_frozen_language_policy": True,
        })
    observed = Counter(f"{x['cwe_identifier']}|{x['language']}" for x in selected)
    if dict(sorted(observed.items())) != EXPECTED_STRATA:
        raise RuntimeError(f"eligible stratum mismatch: {dict(observed)}")
    if len(selected) != 180 or len({x["prompt_id"] for x in selected}) != 180:
        raise RuntimeError("source-matched eligible population must contain 180 unique records")
    population = {
        "schema_version": "phase22c_model3_source_population_v1",
        "status": "FROZEN_BEFORE_GENERATION", "split": "DEVELOPMENT",
        "selection_rule": {
            "source_cwes": ["CWE-120", "CWE-327", "CWE-89"],
            "scanner_eligible_cwe_language_pairs": [
                {"cwe_identifier": cwe, "language": language} for cwe, language in sorted(ELIGIBLE)
            ],
            "source_order_preserved": True, "sampling_used": False,
            "outcome_labels_used": False, "heldout_used": False,
        },
        "source": {"path": "data/cyberseceval/dev_prompts.json", "sha256": sha256_file(DEV), "record_count": 1341},
        "counts": {
            "eligible_total": 180, "eligible_by_cwe_language": dict(sorted(observed.items())),
            "matching_cwe_but_scanner_language_ineligible_total": sum(excluded_matching_cwe.values()),
            "excluded_by_cwe_language": dict(sorted(excluded_matching_cwe.items())),
        },
        "prompt_ids_source_order": [x["prompt_id"] for x in selected],
        "prompt_ids_source_order_canonical_sha256": canonical_hash([x["prompt_id"] for x in selected]),
        "records": selected,
        "model2_outcome_labels_imported": False, "heldout_accessed": False,
    }
    if POPULATION.exists():
        existing_population = json.loads(POPULATION.read_text(encoding="utf-8"))
        if existing_population != population:
            raise RuntimeError("existing interrupted-freeze population differs from deterministic reconstruction")
    else:
        atomic_json(POPULATION, population)

    snapshot = model_snapshot_path()
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    rendered_records = []
    for row in selected:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["test_case_prompt"]}], tokenize=False,
            add_generation_prompt=True,
        )
        token_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        rendered_records.append({
            "population_index": row["population_index"], "source_index": row["source_index"],
            "prompt_id": row["prompt_id"], "source_prompt_sha256": row["source_prompt_sha256"],
            "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
            "rendered_input_token_count": len(token_ids),
        })
    protocol = {
        "schema_version": "phase22c_model3_b0_protocol_v1",
        "status": "FROZEN_BEFORE_GENERATION", "method": "B0", "split": "DEVELOPMENT",
        "source": {"path": "data/cyberseceval/dev_prompts.json", "sha256": sha256_file(DEV), "record_count": 1341},
        "source_population": {"path": "revision/model3/phase22/outputs/phase22c_source_population.json", "sha256": sha256_file(POPULATION), "record_count": 180},
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "dtype": "bfloat16", "device": "cuda:0", "quantization": False, "offload": False},
        "tokenizer": {"id": MODEL_ID, "revision": MODEL_REVISION, "class": type(tokenizer).__name__, "padding_side": tokenizer.padding_side, "pad_token_id": tokenizer.pad_token_id, "local_snapshot_resolved": True},
        "prompt_interface": {
            "chat_template_used": True, "messages": [{"role": "user", "content": "<test_case_prompt>"}],
            "add_generation_prompt": True, "tokenize_template_call": False,
            "tokenizer_add_special_tokens_after_render": False, "input_truncation": False,
            "rendered_prompt_aggregate_sha256": canonical_hash([{"prompt_id": x["prompt_id"], "rendered_prompt_sha256": x["rendered_prompt_sha256"]} for x in rendered_records]),
        },
        "generation": {"seed": 42, "temperature": 0.2, "top_p": 0.95, "max_new_tokens": 512, "do_sample": True, "decode_generated_tokens_only": True, "skip_special_tokens": True, "batch_size": 1},
        "checkpoint": {"source_order_prefix_required": True, "rng_state_saved": True, "interval_records": 10, "failure_preservation": True, "changed_setting_retry": False},
        "scanner": {"script_path": "phases/phase9/colab_scan_phase9.py", "script_sha256": sha256_file(ROOT / "phases/phase9/colab_scan_phase9.py"), "semgrep_version_required": "1.175.0", "registry_config": "p/security-audit", "rules_changed": False},
        "runner": {"path": "revision/model3/phase22/scripts/run_phase22c_b0.py", "sha256": sha256_file(RUNNER)},
        "records": rendered_records, "heldout_accessed": False, "sae_loaded": False,
        "security_intervention": False, "model2_outcome_labels_imported": False,
    }
    atomic_json(PROTOCOL, protocol)
    print(json.dumps({
        "status": protocol["status"], "population_count": 180,
        "population_sha256": sha256_file(POPULATION), "protocol_sha256": sha256_file(PROTOCOL),
        "runner_sha256": sha256_file(RUNNER), "max_input_tokens": max(x["rendered_input_token_count"] for x in rendered_records),
        "eligible_by_cwe_language": dict(sorted(observed.items())),
        "excluded_matching_cwe": dict(sorted(excluded_matching_cwe.items())),
    }, indent=2))


if __name__ == "__main__":
    main()
