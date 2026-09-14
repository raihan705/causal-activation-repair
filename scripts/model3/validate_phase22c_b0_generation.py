#!/usr/bin/env python
"""Validate the immutable Model3 22C B0 generation before scanning."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "revision/model3/phase22/outputs"
POPULATION = OUT / "phase22c_source_population.json"
PROTOCOL = OUT / "phase22c_b0_protocol.json"
GENERATION = OUT / "phase22c_b0_dev_outputs.json"
MANIFEST = OUT / "phase22c_b0_generation_manifest.json"
RUN_MANIFEST = OUT / "phase22c_b0_run_manifest.json"
VALIDATION = OUT / "phase22c_b0_local_validation.json"
EXPECTED_COUNT = 180


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    for path in (POPULATION, PROTOCOL, GENERATION, MANIFEST, RUN_MANIFEST):
        require(path.is_file(), f"missing required artifact: {path}")
    population = json.loads(POPULATION.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    rows = json.loads(GENERATION.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    run = json.loads(RUN_MANIFEST.read_text(encoding="utf-8"))
    expected_ids = [int(x["prompt_id"]) for x in population["records"]]
    require(len(expected_ids) == EXPECTED_COUNT and len(set(expected_ids)) == EXPECTED_COUNT, "population ID failure")
    require(isinstance(rows, list) and len(rows) == EXPECTED_COUNT, "generation count mismatch")
    ids = [int(x["prompt_id"]) for x in rows]
    require(ids == expected_ids and len(set(ids)) == EXPECTED_COUNT, "generation IDs/order mismatch")
    require(manifest.get("status") in {"COMPLETE", "COMPLETE_WITH_PRESERVED_FAILURES"}, "generation manifest incomplete")
    require(run.get("status") == "COMPLETE" and run.get("completed_records") == EXPECTED_COUNT, "run manifest incomplete")
    generation_hash = sha256_file(GENERATION)
    require(manifest["generation_output"]["sha256"] == generation_hash == run["output_sha256"], "generation hash mismatch")
    require(manifest["source_population"]["sha256"] == sha256_file(POPULATION), "population provenance mismatch")
    require(manifest["protocol"]["sha256"] == sha256_file(PROTOCOL), "protocol provenance mismatch")
    failures = 0
    empties = 0
    for source, row, fixed in zip(population["records"], rows, protocol["records"], strict=True):
        pid = int(row["prompt_id"])
        require(int(source["prompt_id"]) == int(fixed["prompt_id"]) == pid, f"record alignment mismatch: {pid}")
        require(row["source_index"] == source["source_index"] and row["population_index"] == source["population_index"], f"source index mismatch: {pid}")
        require(row["source_cwe"] == row["cwe_id"] == source["cwe_identifier"], f"CWE metadata mismatch: {pid}")
        require(row["language"].lower() == source["language"], f"language mismatch: {pid}")
        require(row["source_prompt_sha256"] == source["source_prompt_sha256"], f"source prompt hash mismatch: {pid}")
        require(row["rendered_prompt_sha256"] == fixed["rendered_prompt_sha256"], f"rendered prompt hash mismatch: {pid}")
        require(row["method"] == "B0" and row["seed"] == 42, f"condition mismatch: {pid}")
        require(row["sae_intervention_applied"] is False and row["causal_steering_applied"] is False, f"intervention contamination: {pid}")
        require(row["generated_text"] == row["generated_code"], f"scanner input mismatch: {pid}")
        failures += row["generation_status"] == "FAILED"
        empties += not row["generated_code"].strip()
    require(failures == manifest["failure_count"] and empties == manifest["empty_or_whitespace_count"], "failure/empty summary mismatch")
    value = {
        "schema_version": "phase22c_model3_b0_local_validation_v1", "validation_status": "PASS",
        "generation_sha256": generation_hash, "generation_manifest_sha256": sha256_file(MANIFEST),
        "run_manifest_sha256": sha256_file(RUN_MANIFEST), "protocol_sha256": sha256_file(PROTOCOL),
        "source_population_sha256": sha256_file(POPULATION), "record_count": EXPECTED_COUNT,
        "unique_prompt_id_count": len(set(ids)), "source_order_preserved": ids == expected_ids,
        "success_count": EXPECTED_COUNT - failures, "failure_count": failures,
        "empty_or_whitespace_count": empties, "scanner_run": False, "heldout_accessed": False,
    }
    atomic_json(VALIDATION, value)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
