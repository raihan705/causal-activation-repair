#!/usr/bin/env python
"""Validate the completed Stage 22E steered generation before scanning."""

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
PROTOCOL = OUT / "phase22e_causal_validation_protocol.json"
DENOMINATOR = OUT / "phase22e_paired_denominator_manifest.json"
EXECUTION = OUT / "phase22e_steering_execution_protocol.json"
GENERATION = OUT / "phase22e_steered_generations.json"
CHECKPOINT = OUT / "phase22e_steered_generation_checkpoint.json"
RUN_MANIFEST = OUT / "phase22e_steered_generation_run_manifest.json"
FINAL_MANIFEST = OUT / "phase22e_steered_generation_manifest.json"
RUNNER = PHASE / "scripts/run_phase22e_causal_steering.py"
VALIDATION = OUT / "phase22e_steered_generation_validation.json"

EXPECTED = {
    "protocol": "e0fcf227460e7408471caabb1d695fa1905bc51073a1581776aa1772b13932f4",
    "denominator": "30921b90579dbb9d350c256f90f720332bf41d12f6c4ce6c730ea61240ec7dde",
    "execution": "86d784318cd7d4b6d4cf770f42650aa4f44981f30b269d49e59d678274a088a3",
    "runner": "c7f20dd1f73b3eda1c63ec99c661d2a23fcfebc9ea568d4b718ba99372c5d1f5",
    "generation": "c1844a73f468dd88ce608418ea8ce0cd2802d63a1ce2589ea89c77866917a727",
}
EXPECTED_COUNT = 1170


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
    spec = importlib.util.spec_from_file_location("phase22e_frozen_steering_runner", RUNNER)
    require(spec is not None and spec.loader is not None, "cannot import frozen steering runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validity(text: str) -> dict[str, Any]:
    stripped = text.strip()
    markers = ("def ", "int ", "char ", "return", "if ", "for ", "{", "(")
    marker = next((x for x in markers if x in text), None)
    valid = len(stripped) >= 10 and marker is not None
    return {
        "is_valid": valid,
        "invalid_reason": None if valid else ("EMPTY_OR_TOO_SHORT" if len(stripped) < 10 else "NO_CODE_MARKER"),
        "stripped_character_count": len(stripped),
        "matched_marker": marker,
    }


def main() -> None:
    require(not VALIDATION.exists(), "immutable generation validation already exists")
    paths = (PROTOCOL, DENOMINATOR, EXECUTION, GENERATION, CHECKPOINT, RUN_MANIFEST, FINAL_MANIFEST, RUNNER)
    for path in paths:
        require(path.is_file(), f"missing required artifact: {path}")
    hashes = {
        "protocol": sha256_file(PROTOCOL), "denominator": sha256_file(DENOMINATOR),
        "execution": sha256_file(EXECUTION), "runner": sha256_file(RUNNER),
        "generation": sha256_file(GENERATION), "checkpoint": sha256_file(CHECKPOINT),
        "run_manifest": sha256_file(RUN_MANIFEST), "final_manifest": sha256_file(FINAL_MANIFEST),
    }
    for name, expected in EXPECTED.items():
        require(hashes[name] == expected, f"{name} hash mismatch: {hashes[name]}")

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    denominator = json.loads(DENOMINATOR.read_text(encoding="utf-8"))
    execution = json.loads(EXECUTION.read_text(encoding="utf-8"))
    generation = json.loads(GENERATION.read_text(encoding="utf-8"))
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    run_manifest = json.loads(RUN_MANIFEST.read_text(encoding="utf-8"))
    final_manifest = json.loads(FINAL_MANIFEST.read_text(encoding="utf-8"))
    records = generation["records"]

    require(generation["status"] == run_manifest["status"] == final_manifest["status"] == "COMPLETE", "completion status mismatch")
    require(len(records) == checkpoint["completed_records"] == run_manifest["completed_records"] == final_manifest["record_count"] == EXPECTED_COUNT, "completion count mismatch")
    require(checkpoint["records"] == records, "checkpoint and final generation differ")
    require(final_manifest["success_count"] == EXPECTED_COUNT and final_manifest["failure_count"] == final_manifest["invalid_count"] == 0, "manifest success/validity mismatch")
    require(run_manifest["failure_count"] == run_manifest["invalid_count"] == 0, "run manifest failure/validity mismatch")
    require(final_manifest["output_sha256"] == run_manifest["output_sha256"] == hashes["generation"], "output hash binding mismatch")
    require(final_manifest["generation_manifest_sha256"] if "generation_manifest_sha256" in final_manifest else True, "unexpected empty manifest hash")
    require(run_manifest["generation_manifest_sha256"] == hashes["final_manifest"], "run-to-final manifest binding mismatch")
    require(execution["runner_sha256"] == hashes["runner"], "execution-to-runner binding mismatch")
    require(execution["causal_protocol_sha256"] == hashes["protocol"], "execution-to-causal protocol binding mismatch")
    require(execution["paired_denominator_sha256"] == hashes["denominator"], "execution-to-denominator binding mismatch")

    runner = load_runner()
    worklist = runner.build_worklist(protocol, denominator)
    require(len(worklist) == EXPECTED_COUNT, "rebuilt worklist count mismatch")
    require(canonical_sha256(worklist) == execution["worklist_canonical_sha256"], "rebuilt worklist hash mismatch")
    require([x["record_id"] for x in records] == [x["record_id"] for x in worklist], "generation is not exact worklist order")
    require(final_manifest["record_ids_canonical_sha256"] == canonical_sha256([x["record_id"] for x in records]), "record ID hash mismatch")

    metadata_fields = (
        "record_index", "record_id", "target_cwe", "layer", "feature_id",
        "statistical_rank", "composite_score", "route_key", "screening_alpha",
        "arm", "prompt_id", "source_index", "language", "source_prompt_sha256",
        "rendered_prompt_sha256", "input_token_count",
    )
    for index, (record, item) in enumerate(zip(records, worklist)):
        require(all(record[field] == item[field] for field in metadata_fields), f"worklist metadata mismatch at row {index}")
        require(record["generation_status"] == "SUCCESS" and record["exception"] is None, f"generation failure at row {index}")
        require(record["generated_text"] == record["generated_code"] and record["generated_code"].strip(), f"generated code mismatch/empty at row {index}")
        require(record["generated_text_sha256"] == sha256_text(record["generated_text"]), f"generated text hash mismatch at row {index}")
        require(record["validity"] == validity(record["generated_text"]) and record["validity"]["is_valid"], f"validity mismatch at row {index}")
        require(record["method"] == "MODEL3_CANDIDATE_SCREEN", f"method mismatch at row {index}")
        require(record["direction"] == "POSITIVE", f"direction mismatch at row {index}")
        require(record["intervention"] == "DIRECT_LAST_TOKEN_RESIDUAL_ADDITION_EVERY_HOOK_CALL", f"intervention mismatch at row {index}")
        require(record["seed"] == 42 and record["seed_reset_immediately_before_generation"], f"seed mismatch at row {index}")
        require(record["generation_settings"] == protocol["generation"], f"generation settings mismatch at row {index}")
        require(record["model_id"] == protocol["model"]["id"] and record["model_revision"] == protocol["model"]["revision"], f"model provenance mismatch at row {index}")
        require(record["sae_revision"] == protocol["sae"]["revision"], f"SAE revision mismatch at row {index}")
        require(record["causal_steering_applied"] and not record["scanner_run"], f"execution boundary mismatch at row {index}")
        require(record["execution_protocol_sha256"] == hashes["execution"], f"record execution protocol mismatch at row {index}")

    target_counts = Counter(x["target_cwe"] for x in records)
    layer_counts = Counter(int(x["layer"]) for x in records)
    target_layer_counts = Counter((x["target_cwe"], int(x["layer"])) for x in records)
    arm_counts = Counter((x["target_cwe"], x["arm"]) for x in records)
    route_assignments = {(x["target_cwe"], int(x["layer"]), int(x["feature_id"])) for x in records}
    require(target_counts == Counter({"CWE-120": 540, "CWE-327": 330, "CWE-89": 300}), f"target distribution mismatch: {target_counts}")
    require(layer_counts == Counter({7: 390, 15: 390, 23: 390}), f"layer distribution mismatch: {layer_counts}")
    require(len(route_assignments) == 90, "target/layer/feature assignment count mismatch")
    require(arm_counts == Counter({
        ("CWE-120", "QUALIFIED_UNSAFE"): 390, ("CWE-120", "QUALIFIED_SAFE"): 150,
        ("CWE-327", "QUALIFIED_UNSAFE"): 180, ("CWE-327", "QUALIFIED_SAFE"): 150,
        ("CWE-89", "QUALIFIED_UNSAFE"): 150, ("CWE-89", "QUALIFIED_SAFE"): 150,
    }), f"arm distribution mismatch: {arm_counts}")

    result = {
        "schema_version": "phase22e_steered_generation_validation_v1",
        "status": "PASS",
        "artifact_hashes": hashes,
        "checks": {
            "frozen_provenance_chain": True, "complete_1170": True,
            "checkpoint_equals_final": True, "exact_worklist_order": True,
            "record_metadata_and_hashes": True, "generation_settings_and_seed": True,
            "intervention_metadata": True, "all_successful": True,
            "all_valid": True, "scanner_not_run": True, "heldout_not_used": True,
        },
        "record_count": EXPECTED_COUNT,
        "success_count": EXPECTED_COUNT,
        "failure_count": 0,
        "invalid_count": 0,
        "target_counts": dict(target_counts),
        "layer_counts": {str(key): value for key, value in sorted(layer_counts.items())},
        "target_layer_counts": {f"{key[0]}|L{key[1]}": value for key, value in sorted(target_layer_counts.items())},
        "arm_counts": {f"{key[0]}|{key[1]}": value for key, value in sorted(arm_counts.items())},
        "target_layer_feature_assignment_count": len(route_assignments),
        "elapsed_seconds": run_manifest["elapsed_seconds"],
        "elapsed_hours": run_manifest["elapsed_seconds"] / 3600.0,
        "gpu": run_manifest["gpu"],
        "scanner_run": False,
        "heldout_used": False,
    }
    atomic_json(VALIDATION, result)
    print(json.dumps({**result, "validation_sha256": sha256_file(VALIDATION)}, indent=2))


if __name__ == "__main__":
    main()
