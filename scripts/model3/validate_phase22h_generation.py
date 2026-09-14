#!/usr/bin/env python
"""Validate the complete frozen Stage 22H generation without scanning it."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "revision/model3/phase22/outputs"
PROTOCOL = OUT / "phase22g_prospective_evaluation_protocol.json"
POPULATION = OUT / "phase22g_evaluation_population.json"
GENERATION = OUT / "phase22h_paired_generations.json"
CHECKPOINT = OUT / "phase22h_paired_generation_checkpoint.json"
RUN_MANIFEST = OUT / "phase22h_paired_generation_run_manifest.json"
GENERATION_MANIFEST = OUT / "phase22h_paired_generation_manifest.json"
RUNNER = Path(__file__).with_name("run_phase22h_paired_evaluation.py")
OUTPUT = OUT / "phase22h_generation_validation.json"
SAE_SHA256 = {
    15: "1339c52258e95bc64535a90e45067187722e659e0ee03c976db813614b29ab5b",
    23: "a593d0da4a10cde4674b48749bf573b14e40119c1cff34cf2d499f334940ef4b",
}


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


def validity(text: str) -> dict[str, Any]:
    stripped = text.strip()
    markers = ("def ", "int ", "char ", "return", "if ", "for ", "{", "(")
    marker = next((item for item in markers if item in text), None)
    valid = len(stripped) >= 10 and marker is not None
    return {
        "is_valid": valid,
        "invalid_reason": None if valid else ("EMPTY_OR_TOO_SHORT" if len(stripped) < 10 else "NO_CODE_MARKER"),
        "stripped_character_count": len(stripped),
        "matched_marker": marker,
    }


def build_worklist(protocol: dict[str, Any], population: dict[str, Any]) -> list[dict[str, Any]]:
    rows = population["records"]
    routes = {item["target_cwe"]: item for item in protocol["frozen_method"]["routes"]}
    result: list[dict[str, Any]] = []
    for seed in protocol["generation"]["seeds"]:
        for row in rows:
            result.append({
                "record_index": len(result), "record_id": f"B0|S{seed}|P{row['prompt_id']}",
                "condition": "B0", "seed": seed, **row,
                "layer": None, "feature_id": None, "alpha": 0.0,
            })
    for layer in sorted({int(item["layer"]) for item in routes.values()}):
        for seed in protocol["generation"]["seeds"]:
            for row in rows:
                route = routes[row["cwe_identifier"]]
                if int(route["layer"]) != layer:
                    continue
                result.append({
                    "record_index": len(result), "record_id": f"MODEL3_ROUTED|S{seed}|P{row['prompt_id']}",
                    "condition": "MODEL3_CWE_LABEL_ROUTED", "seed": seed, **row,
                    "layer": layer, "feature_id": int(route["feature_id"]), "alpha": float(route["alpha"]),
                })
    return result


def main() -> None:
    require(not OUTPUT.exists(), "immutable Stage 22H generation validation already exists")
    for path in (PROTOCOL, POPULATION, GENERATION, CHECKPOINT, RUN_MANIFEST, GENERATION_MANIFEST, RUNNER):
        require(path.is_file(), f"missing generation artifact: {path}")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    population = json.loads(POPULATION.read_text(encoding="utf-8"))
    generation = json.loads(GENERATION.read_text(encoding="utf-8"))
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    run_manifest = json.loads(RUN_MANIFEST.read_text(encoding="utf-8"))
    manifest = json.loads(GENERATION_MANIFEST.read_text(encoding="utf-8"))
    protocol_hash = sha256_file(PROTOCOL)
    population_hash = sha256_file(POPULATION)
    runner_hash = sha256_file(RUNNER)
    generation_hash = sha256_file(GENERATION)

    require(protocol_hash == "ed400877e33e784f33247f82d4186f673caee0d861d5f0b5717bfc4a9afed904", "protocol drift")
    require(population_hash == "1a8ce89e42a075fd09ecc626f51b95561621c70f8c4a6942b113b9213869e2b8", "population drift")
    require(runner_hash == "c42fa2fc76f7f2b663435f660d06fb7e70221aa1b0fc91bd1db4513bc78748cc", "runner drift")
    require(generation["status"] == "COMPLETE", "generation not complete")
    records = generation["records"]
    require(len(records) == 468, "generation record count mismatch")
    worklist = build_worklist(protocol, population)
    require(len(worklist) == 468, "rebuilt worklist count mismatch")
    require(canonical_sha256(worklist) == protocol["execution"]["worklist_canonical_sha256"], "rebuilt worklist hash mismatch")
    require([row["record_id"] for row in records] == [row["record_id"] for row in worklist], "record order/worklist mismatch")

    routes = {item["target_cwe"]: item for item in protocol["frozen_method"]["routes"]}
    for index, (record, expected) in enumerate(zip(records, worklist)):
        require(record["record_index"] == expected["record_index"] == index, f"record index mismatch at {index}")
        for field in ("record_id", "condition", "seed", "prompt_id", "source_index", "cwe_identifier", "language", "source_prompt_sha256", "rendered_prompt_sha256", "rendered_input_token_count", "layer", "feature_id"):
            require(record[field] == expected[field], f"{field} mismatch at {index}")
        require(float(record["alpha"]) == float(expected["alpha"]), f"alpha mismatch at {index}")
        require(record["protocol_sha256"] == protocol_hash, f"record protocol hash mismatch at {index}")
        require(record["model_id"] == protocol["model"]["id"] and record["model_revision"] == protocol["model"]["revision"], f"model identity mismatch at {index}")
        require(record["seed_reset_immediately_before_generation"] is True, f"seed reset flag missing at {index}")
        require(record["generation_status"] == "SUCCESS" and record["exception"] is None, f"failed generation at {index}")
        require(record["generated_text"] == record["generated_code"], f"text/code mismatch at {index}")
        require(sha256_text(record["generated_text"]) == record["generated_text_sha256"], f"generated text hash mismatch at {index}")
        require(record["generated_token_count"] > 0, f"empty token suffix at {index}")
        require(record["validity"] == validity(record["generated_text"]), f"validity recomputation mismatch at {index}")
        require(record["validity"]["is_valid"], f"frozen-invalid generation at {index}")
        if record["condition"] == "B0":
            require(record["method"] == "B0" and record["causal_steering_applied"] is False, f"B0 intervention mismatch at {index}")
            require(record["layer"] is None and record["feature_id"] is None and record["alpha"] == 0.0, f"B0 route fields mismatch at {index}")
            require(record["routing_fields_consumed"] == [] and record["sae_revision"] is None and record["sae_weight_sha256"] is None, f"B0 routing/SAE mismatch at {index}")
        else:
            route = routes[record["cwe_identifier"]]
            require(record["method"] == "MODEL3_CWE_LABEL_ROUTED" and record["causal_steering_applied"] is True, f"routed intervention mismatch at {index}")
            require(record["routing_fields_consumed"] == ["cwe_identifier"], f"routed fields mismatch at {index}")
            require(record["layer"] == route["layer"] and record["feature_id"] == route["feature_id"] and float(record["alpha"]) == float(route["alpha"]), f"frozen route mismatch at {index}")
            require(record["sae_revision"] == protocol["sae"]["revision"] and record["sae_weight_sha256"] == SAE_SHA256[int(record["layer"])], f"SAE mismatch at {index}")

    require(checkpoint["completed_records"] == 468 and checkpoint["records"] == records, "final checkpoint differs from final records")
    require(checkpoint["protocol_sha256"] == protocol_hash and checkpoint["runner_sha256"] == runner_hash, "checkpoint lineage mismatch")
    require(run_manifest["status"] == "COMPLETE" and run_manifest["completed_records"] == 468, "run manifest incomplete")
    require(run_manifest["protocol_sha256"] == protocol_hash and run_manifest["runner_sha256"] == runner_hash, "run manifest lineage mismatch")
    require(run_manifest["parameter_devices"] == ["cuda:0"], "model was not wholly on cuda:0")
    require(manifest["status"] == "COMPLETE" and manifest["record_count"] == 468, "generation manifest incomplete")
    require(manifest["output_sha256"] == generation_hash and manifest["protocol_sha256"] == protocol_hash, "generation manifest lineage mismatch")
    require(manifest["record_ids_canonical_sha256"] == canonical_sha256([row["record_id"] for row in records]), "record-ID canonical hash mismatch")

    condition_counts = Counter(row["condition"] for row in records)
    seed_condition_counts = Counter(f"S{row['seed']}|{row['condition']}" for row in records)
    cwe_condition_counts = Counter(f"{row['cwe_identifier']}|{row['condition']}" for row in records)
    paired = {(int(row["seed"]), int(row["prompt_id"])): {} for row in records}
    for row in records:
        paired[(int(row["seed"]), int(row["prompt_id"]))][row["condition"]] = row
    require(len(paired) == 234 and all(set(value) == {"B0", "MODEL3_CWE_LABEL_ROUTED"} for value in paired.values()), "paired coverage mismatch")
    identical_pairs = sum(value["B0"]["generated_text"] == value["MODEL3_CWE_LABEL_ROUTED"]["generated_text"] for value in paired.values())

    result = {
        "schema_version": "phase22h_generation_validation_v1",
        "status": "PASS",
        "checks": {
            "frozen_protocol_population_runner_hashes_match": True,
            "generation_complete_468": True,
            "worklist_exact_rebuild_and_order_match": True,
            "all_records_successful": True,
            "all_records_frozen_valid": True,
            "all_generated_text_hashes_match": True,
            "all_b0_records_unsteered": True,
            "all_routed_records_match_frozen_cwe_route": True,
            "all_234_prompt_seed_pairs_complete": True,
            "checkpoint_equals_final_records": True,
            "model_wholly_on_cuda0": True,
            "scanner_not_run": True,
        },
        "protocol_sha256": protocol_hash,
        "population_sha256": population_hash,
        "runner_sha256": runner_hash,
        "generation_sha256": generation_hash,
        "generation_manifest_sha256": sha256_file(GENERATION_MANIFEST),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "run_manifest_sha256": sha256_file(RUN_MANIFEST),
        "record_count": len(records),
        "condition_counts": dict(sorted(condition_counts.items())),
        "seed_condition_counts": dict(sorted(seed_condition_counts.items())),
        "cwe_condition_counts": dict(sorted(cwe_condition_counts.items())),
        "prompt_seed_pair_count": len(paired),
        "paired_generated_text_identical_count": identical_pairs,
        "paired_generated_text_changed_count": len(paired) - identical_pairs,
        "success_count": sum(row["generation_status"] == "SUCCESS" for row in records),
        "failure_count": sum(row["generation_status"] != "SUCCESS" for row in records),
        "invalid_count": sum(not row["validity"]["is_valid"] for row in records),
        "elapsed_seconds": run_manifest["elapsed_seconds"],
        "gpu": run_manifest["gpu"],
        "scanner_run": False,
        "security_metrics_computed": False,
    }
    atomic_json(OUTPUT, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
