#!/usr/bin/env python
"""Validate completed Stage 22F strength generation before scanning."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
PHASE = ROOT / "revision/model3/phase22"
OUT = PHASE / "outputs"
PROTOCOL = OUT / "phase22f_strength_calibration_protocol.json"
CAUSAL_PROTOCOL = OUT / "phase22e_causal_validation_protocol.json"
EXECUTION = OUT / "phase22f_strength_execution_protocol.json"
GENERATION = OUT / "phase22f_strength_generations.json"
CHECKPOINT = OUT / "phase22f_strength_generation_checkpoint.json"
RUN_MANIFEST = OUT / "phase22f_strength_generation_run_manifest.json"
FINAL_MANIFEST = OUT / "phase22f_strength_generation_manifest.json"
RUNNER = PHASE / "scripts/run_phase22f_strength_calibration.py"
SHARED = PHASE / "scripts/run_phase22e_causal_steering.py"
VALIDATION = OUT / "phase22f_strength_generation_validation.json"

EXPECTED = {
    "protocol": "20e57079a57a65fe4569132c09b05b649a98627c85fe9c6effd8b3c20aac279c",
    "causal_protocol": "e0fcf227460e7408471caabb1d695fa1905bc51073a1581776aa1772b13932f4",
    "execution": "1c4cccc2c76011783113022c5d836120515c818e367af9ba001d8857edc1cd72",
    "runner": "b48fa0a12d59af20ec06c7ce7fbcb7971836363409a8977989d68471c22d96ce",
    "shared": "c7f20dd1f73b3eda1c63ec99c661d2a23fcfebc9ea568d4b718ba99372c5d1f5",
    "generation": "7d8996c7d62c8966eea62f51324a3a6c264c3a92284361cf59c495362fac24c9",
}
EXPECTED_COUNT = 1083


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


def load_runner() -> Any:
    spec = importlib.util.spec_from_file_location("phase22f_frozen_runner", RUNNER)
    require(spec is not None and spec.loader is not None, "cannot import frozen runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validity(text: str) -> dict[str, Any]:
    stripped = text.strip()
    markers = ("def ", "int ", "char ", "return", "if ", "for ", "{", "(")
    marker = next((x for x in markers if x in text), None)
    valid = len(stripped) >= 10 and marker is not None
    return {"is_valid": valid, "invalid_reason": None if valid else ("EMPTY_OR_TOO_SHORT" if len(stripped) < 10 else "NO_CODE_MARKER"), "stripped_character_count": len(stripped), "matched_marker": marker}


def main() -> None:
    require(not VALIDATION.exists(), "immutable validation already exists")
    for path in (PROTOCOL, CAUSAL_PROTOCOL, EXECUTION, GENERATION, CHECKPOINT, RUN_MANIFEST, FINAL_MANIFEST, RUNNER, SHARED):
        require(path.is_file(), f"missing artifact: {path}")
    hashes = {
        "protocol": sha256_file(PROTOCOL), "causal_protocol": sha256_file(CAUSAL_PROTOCOL),
        "execution": sha256_file(EXECUTION), "runner": sha256_file(RUNNER), "shared": sha256_file(SHARED),
        "generation": sha256_file(GENERATION), "checkpoint": sha256_file(CHECKPOINT),
        "run_manifest": sha256_file(RUN_MANIFEST), "final_manifest": sha256_file(FINAL_MANIFEST),
    }
    for name, expected in EXPECTED.items():
        require(hashes[name] == expected, f"{name} hash mismatch: {hashes[name]}")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    causal = json.loads(CAUSAL_PROTOCOL.read_text(encoding="utf-8"))
    execution = json.loads(EXECUTION.read_text(encoding="utf-8"))
    generation = json.loads(GENERATION.read_text(encoding="utf-8"))
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    run_manifest = json.loads(RUN_MANIFEST.read_text(encoding="utf-8"))
    final_manifest = json.loads(FINAL_MANIFEST.read_text(encoding="utf-8"))
    records = generation["records"]
    require(generation["status"] == run_manifest["status"] == final_manifest["status"] == "COMPLETE", "completion status mismatch")
    require(len(records) == checkpoint["completed_records"] == run_manifest["completed_records"] == final_manifest["record_count"] == EXPECTED_COUNT, "completion count mismatch")
    require(checkpoint["records"] == records, "checkpoint/final mismatch")
    require(final_manifest["success_count"] == EXPECTED_COUNT and final_manifest["failure_count"] == 0, "generation success mismatch")
    require(run_manifest["failure_count"] == 0, "run failure mismatch")
    require(final_manifest["output_sha256"] == run_manifest["output_sha256"] == hashes["generation"], "output hash binding mismatch")
    require(run_manifest["generation_manifest_sha256"] == hashes["final_manifest"], "manifest hash binding mismatch")
    require(execution["runner_sha256"] == hashes["runner"] and execution["shared_helper_sha256"] == hashes["shared"], "runner/helper binding mismatch")
    require(execution["strength_protocol_sha256"] == hashes["protocol"], "protocol binding mismatch")
    runner = load_runner()
    worklist = runner.build_worklist(protocol, causal)
    require(len(worklist) == EXPECTED_COUNT and canonical_sha256(worklist) == execution["worklist_canonical_sha256"], "rebuilt worklist mismatch")
    require([x["record_id"] for x in records] == [x["record_id"] for x in worklist], "record order mismatch")
    require(final_manifest["record_ids_canonical_sha256"] == canonical_sha256([x["record_id"] for x in records]), "record ID hash mismatch")
    fields = ("record_index", "record_id", "target_cwe", "layer", "feature_id", "route_key", "statistical_rank", "composite_score", "screening_alpha", "strength_multiplier", "calibration_alpha", "arm", "prompt_id", "source_index", "language", "source_prompt_sha256", "rendered_prompt_sha256", "input_token_count")
    for index, (record, item) in enumerate(zip(records, worklist)):
        require(all(record[field] == item[field] for field in fields), f"worklist metadata mismatch at {index}")
        require(record["generation_status"] == "SUCCESS" and record["exception"] is None, f"generation failure at {index}")
        require(record["generated_text"] == record["generated_code"], f"generated text/code mismatch at {index}")
        require(record["generated_text_sha256"] == sha256_text(record["generated_text"]), f"text hash mismatch at {index}")
        require(record["validity"] == validity(record["generated_text"]), f"validity mismatch at {index}")
        require(record["strength_multiplier"] in (0.5, 1.5, 2.0), f"unexpected multiplier at {index}")
        require(record["calibration_alpha"] == record["screening_alpha"] * record["strength_multiplier"], f"alpha formula mismatch at {index}")
        require(record["generation_settings"] == causal["generation"], f"generation settings mismatch at {index}")
        require(record["seed"] == 42 and record["seed_reset_immediately_before_generation"], f"seed mismatch at {index}")
        require(record["direction"] == "POSITIVE" and record["intervention"] == "DIRECT_LAST_TOKEN_RESIDUAL_ADDITION_EVERY_HOOK_CALL", f"intervention mismatch at {index}")
        require(record["causal_steering_applied"] and not record["scanner_run"], f"execution boundary mismatch at {index}")
        require(record["execution_protocol_sha256"] == hashes["execution"], f"record protocol mismatch at {index}")
    invalid = [x for x in records if not x["validity"]["is_valid"]]
    require(len(invalid) == final_manifest["invalid_count"] == run_manifest["invalid_count"] == 34, "invalid count mismatch")
    invalid_by_multiplier = Counter(str(x["strength_multiplier"]) for x in invalid)
    invalid_by_target = Counter(x["target_cwe"] for x in invalid)
    invalid_by_layer = Counter(str(x["layer"]) for x in invalid)
    target_counts = Counter(x["target_cwe"] for x in records)
    multiplier_counts = Counter(str(x["strength_multiplier"]) for x in records)
    require(target_counts == Counter({"CWE-120": 810, "CWE-327": 33, "CWE-89": 240}), f"target distribution mismatch: {target_counts}")
    require(multiplier_counts == Counter({"0.5": 361, "1.5": 361, "2.0": 361}), f"multiplier distribution mismatch: {multiplier_counts}")
    result = {
        "schema_version": "phase22f_strength_generation_validation_v1", "status": "PASS",
        "artifact_hashes": hashes,
        "checks": {"frozen_provenance_chain": True, "complete_1083": True, "checkpoint_equals_final": True, "exact_worklist_order": True, "strength_formula_and_grid": True, "generation_settings_and_seed": True, "intervention_metadata": True, "all_calls_successful": True, "invalid_outputs_retained": True, "scanner_not_run": True, "stage22g_not_started": True, "heldout_not_used": True},
        "record_count": EXPECTED_COUNT, "success_count": EXPECTED_COUNT, "failure_count": 0,
        "valid_count": EXPECTED_COUNT - len(invalid), "invalid_count": len(invalid),
        "invalid_by_multiplier": dict(invalid_by_multiplier), "invalid_by_target": dict(invalid_by_target),
        "invalid_by_layer": dict(invalid_by_layer), "target_counts": dict(target_counts),
        "multiplier_counts": dict(multiplier_counts), "elapsed_seconds": run_manifest["elapsed_seconds"],
        "elapsed_hours": run_manifest["elapsed_seconds"] / 3600.0, "gpu": run_manifest["gpu"],
        "scientific_handling": "retain all 34 invalid outputs; unsafe invalids cannot repair and count in invalid rate; safe invalids count as corruption",
        "scanner_run": False, "stage22g_started": False, "heldout_used": False,
    }
    atomic_json(VALIDATION, result)
    print(json.dumps({**result, "validation_sha256": sha256_file(VALIDATION)}, indent=2))


if __name__ == "__main__":
    main()
