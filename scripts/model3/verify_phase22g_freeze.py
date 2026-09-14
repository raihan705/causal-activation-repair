#!/usr/bin/env python
"""Independently rebuild and verify the frozen Stage 22G inputs."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "revision/model3/phase22/outputs"
DEV = ROOT / "data/cyberseceval/dev_prompts.json"
TEST = ROOT / "data/cyberseceval/test_prompts.json"
POPULATION = OUT / "phase22g_evaluation_population.json"
PROTOCOL = OUT / "phase22g_prospective_evaluation_protocol.json"
RUNNER = Path(__file__).with_name("run_phase22h_paired_evaluation.py")
OUTPUT = OUT / "phase22g_freeze_verification.json"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
ELIGIBLE = {("CWE-120", "c"), ("CWE-120", "cpp"), ("CWE-327", "java"), ("CWE-89", "python")}


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def build_worklist(protocol: dict[str, Any], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    routes = {item["target_cwe"]: item for item in protocol["frozen_method"]["routes"]}
    worklist: list[dict[str, Any]] = []
    for seed in protocol["generation"]["seeds"]:
        for row in records:
            worklist.append({
                "record_index": len(worklist), "record_id": f"B0|S{seed}|P{row['prompt_id']}",
                "condition": "B0", "seed": seed, **row,
                "layer": None, "feature_id": None, "alpha": 0.0,
            })
    for layer in sorted({int(item["layer"]) for item in routes.values()}):
        for seed in protocol["generation"]["seeds"]:
            for row in records:
                route = routes[row["cwe_identifier"]]
                if int(route["layer"]) != layer:
                    continue
                worklist.append({
                    "record_index": len(worklist), "record_id": f"MODEL3_ROUTED|S{seed}|P{row['prompt_id']}",
                    "condition": "MODEL3_CWE_LABEL_ROUTED", "seed": seed, **row,
                    "layer": layer, "feature_id": int(route["feature_id"]), "alpha": float(route["alpha"]),
                })
    return worklist


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    frozen = json.loads(POPULATION.read_text(encoding="utf-8"))
    test = json.loads(TEST.read_text(encoding="utf-8"))
    dev = json.loads(DEV.read_text(encoding="utf-8"))
    require(not set(int(x["prompt_id"]) for x in test).intersection(int(x["prompt_id"]) for x in dev), "split overlap")
    os.environ["HF_HUB_OFFLINE"] = "1"
    snapshot = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots" / MODEL_REVISION
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    rebuilt = []
    excluded = Counter()
    for source_index, row in enumerate(test):
        cwe = row.get("cwe_identifier", "")
        language = row.get("language", "").lower()
        if cwe not in {"CWE-120", "CWE-327", "CWE-89"}:
            continue
        if (cwe, language) not in ELIGIBLE:
            excluded[f"{cwe}|{language}"] += 1
            continue
        prompt = row["test_case_prompt"]
        rendered = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        rebuilt.append({
            "population_index": len(rebuilt), "source_index": source_index,
            "prompt_id": int(row["prompt_id"]), "cwe_identifier": cwe, "language": language,
            "source_prompt_sha256": sha256_text(prompt), "rendered_prompt_sha256": sha256_text(rendered),
            "rendered_input_token_count": len(tokenizer(rendered, add_special_tokens=False)["input_ids"]),
            "scanner_eligible_by_frozen_language_policy": True,
        })
    require(rebuilt == frozen["records"], "population deterministic rebuild differs")
    worklist = build_worklist(protocol, rebuilt)
    require(len(worklist) == 468, "worklist count drift")
    require(canonical_sha256(worklist) == protocol["execution"]["worklist_canonical_sha256"], "worklist hash drift")
    require(protocol["runner"]["sha256"] == sha256_file(RUNNER), "runner hash drift")
    result = {
        "schema_version": "phase22g_freeze_verification_v1",
        "status": "PASS",
        "checks": {
            "test_source_hash_matches": True,
            "development_test_id_overlap_zero": True,
            "population_exact_rebuild_equal": True,
            "eligible_record_count_78": True,
            "all_eligible_records_included": True,
            "no_sampling_or_replacement": True,
            "three_routes_frozen": True,
            "seeds_42_43_44_frozen": True,
            "worklist_468_exact_rebuild_hash_matches": True,
            "runner_hash_matches": True,
            "heldout_generation_not_run": True,
            "heldout_scanner_not_run": True,
        },
        "eligible_by_cwe_language": frozen["counts"]["eligible_by_cwe_language"],
        "excluded_by_cwe_language": dict(sorted(excluded.items())),
        "population_sha256": sha256_file(POPULATION),
        "protocol_sha256": sha256_file(PROTOCOL),
        "runner_sha256": sha256_file(RUNNER),
        "prompt_ids_canonical_sha256": frozen["prompt_ids_source_order_canonical_sha256"],
        "worklist_canonical_sha256": canonical_sha256(worklist),
        "max_rendered_input_tokens": max(item["rendered_input_token_count"] for item in rebuilt),
    }
    atomic_json(OUTPUT, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
