#!/usr/bin/env python
"""Aggregate the three clean-process Phase 22B technical records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


LAYERS = (7, 15, 23)
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
SAE_REPOSITORY = "andyrdt/saes-qwen2.5-7b-instruct"
SAE_REVISION = "c37e53c4bb07127ad17ab88f28b93d4e87142e59"


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(output_dir: Path) -> int:
    records: list[dict[str, Any]] = []
    record_files: list[dict[str, str]] = []
    for layer in LAYERS:
        path = output_dir / f"phase22b_layer_{layer}_technical.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing layer record: {path}")
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("layer") != layer:
            raise ValueError(f"Layer identity mismatch in {path}")
        if record.get("model_revision") != MODEL_REVISION or record.get("sae_revision") != SAE_REVISION:
            raise ValueError(f"Frozen revision mismatch in {path}")
        records.append(record)
        record_files.append({"path": path.name, "sha256": sha256_file(path)})

    compatible = [int(record["layer"]) for record in records if record.get("status") == "COMPATIBLE"]
    failed = [
        {
            "layer": int(record["layer"]),
            "status": record.get("status"),
            "blocker": record.get("blocker"),
        }
        for record in records
        if record.get("status") != "COMPATIBLE"
    ]
    matrix_rows: list[dict[str, Any]] = []
    reconstruction_rows: list[dict[str, Any]] = []
    memory_rows: list[dict[str, Any]] = []
    for record in records:
        reconstruction = record.get("reconstruction", {})
        generation = record.get("generation", {})
        memory = record.get("memory", {}).get("final_and_peak", {})
        matrix_rows.append(
            {
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "sae_repository": SAE_REPOSITORY,
                "sae_revision": SAE_REVISION,
                "layer": record["layer"],
                "sae_id": record["sae_id"],
                "status": record["status"],
                "hashes_match": record.get("file_verification", {}).get("all_hashes_match", False),
                "model_load": "PASS" if record.get("model_loading") else "FAIL_OR_NOT_RUN",
                "sae_load": "PASS" if record.get("sae_loading") else "FAIL_OR_NOT_RUN",
                "hook_calls": record.get("hook", {}).get("reconstruction_hook_calls", 0),
                "explained_variance": reconstruction.get("explained_variance", ""),
                "fraction_loss_recovered": reconstruction.get("fraction_loss_recovered", ""),
                "observed_mean_l0": reconstruction.get("mean_l0", ""),
                "reconstructed_generated_tokens": generation.get("reconstructed_generated_tokens", 0),
                "peak_allocated_vram_bytes": memory.get("peak_allocated_vram_bytes", ""),
                "blocker": record.get("blocker") or "",
            }
        )
        reconstruction_rows.append(
            {
                "layer": record["layer"],
                "status": record["status"],
                "mse": reconstruction.get("mse", ""),
                "normalized_mse": reconstruction.get("normalized_mse", ""),
                "explained_variance": reconstruction.get("explained_variance", ""),
                "mean_cosine_similarity": reconstruction.get("mean_cosine_similarity", ""),
                "mean_l0": reconstruction.get("mean_l0", ""),
                "baseline_cross_entropy": reconstruction.get("baseline_cross_entropy", ""),
                "reconstructed_cross_entropy": reconstruction.get("reconstructed_cross_entropy", ""),
                "zero_ablation_cross_entropy": reconstruction.get("zero_ablation_cross_entropy", ""),
                "fraction_loss_recovered": reconstruction.get("fraction_loss_recovered", ""),
                "gate_status": reconstruction.get("gate_status", "NOT_RUN"),
            }
        )
        for scope in ("model_only", "after_sae_load", "final_and_peak"):
            values = record.get("memory", {}).get(scope)
            if values:
                memory_rows.append({"layer": record["layer"], "scope": scope, **values})

    atomic_csv(output_dir / "phase22b_compatibility_matrix.csv", list(matrix_rows[0]), matrix_rows)
    atomic_csv(output_dir / "phase22b_reconstruction.csv", list(reconstruction_rows[0]), reconstruction_rows)
    if memory_rows:
        atomic_csv(output_dir / "phase22b_memory.csv", list(memory_rows[0]), memory_rows)

    if not compatible:
        checkpoint_status = "BLOCKED_RECONSTRUCTION_OR_TECHNICAL_COMPATIBILITY"
        checkpoint = "FAIL_NO_COMPATIBLE_LAYER"
    elif failed:
        checkpoint_status = "PASS_PARTIAL_COMPATIBILITY"
        checkpoint = "22B_PASS_WITH_COMPATIBLE_LAYERS"
    else:
        checkpoint_status = "PASS"
        checkpoint = "22B_PASS_ALL_FROZEN_LAYERS"
    result = {
        "schema_version": "phase22b_checkpoint_v1",
        "phase": 22,
        "stage": "22B",
        "status": checkpoint_status,
        "checkpoint": checkpoint,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "sae_repository": SAE_REPOSITORY,
        "sae_revision": SAE_REVISION,
        "tested_layers": list(LAYERS),
        "compatible_layers": compatible,
        "failed_layers": failed,
        "compatible_layer_count": len(compatible),
        "technical_gate_rule": "At least one prospectively frozen layer must pass all pre-outcome technical and positive-reconstruction criteria.",
        "record_files": record_files,
        "benchmark_generation": False,
        "scanner_run": False,
        "feature_discovery_run": False,
        "heldout_used": False,
        "quantization_used": False,
        "cpu_offload_used": False,
        "next_stage": "22C_DEVELOPMENT_B0_POPULATION_AND_SCANNER_BASELINE" if compatible else None,
        "next_stage_ready": True,
    }
    atomic_json(output_dir / "phase22b_checkpoint.json", result)
    print(json.dumps(result, indent=2))
    return 0 if compatible else 4


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("revision/model3/phase22/outputs"))
    args = parser.parse_args()
    return run(args.output_dir.resolve())


if __name__ == "__main__":
    sys.exit(main())
