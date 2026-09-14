#!/usr/bin/env python
"""Aggregate source-matched reconstruction metrics and freeze layer viability."""

import csv
import hashlib
import json
import math
import os
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "revision/model2/phase21/outputs"
ACT = OUT / "activations"
LAYERS = (9, 20, 31)
SOURCES = {"CVE_VULNERABLE": 128, "CVE_FIXED": 128, "MODEL2_B0": 261}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def write_json(path, value):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def relative(path):
    return str(Path(path).relative_to(ROOT)).replace("\\", "/")


def main():
    source_manifest = OUT / "model2_activation_source_manifest.json"
    activation_manifest = OUT / "model2_activation_manifest.csv"
    activation_run = OUT / "model2_activation_run_manifest.json"
    scanner_provenance = OUT / "model2_scanner_provenance.json"
    if json.loads(source_manifest.read_text(encoding="utf-8"))["record_count"] != 517:
        raise RuntimeError("source population mismatch")
    if json.loads(activation_run.read_text(encoding="utf-8"))["status"] != "COMPLETE":
        raise RuntimeError("activation extraction incomplete")
    activation_rows = read_csv(activation_manifest)
    if len(activation_rows) != 1551:
        raise RuntimeError("activation manifest must contain 517 rows for each of three layers")
    for layer in LAYERS:
        subset = [row for row in activation_rows if int(row["layer"]) == layer]
        if len(subset) != 517 or Counter(row["source"] for row in subset) != Counter(SOURCES):
            raise RuntimeError(f"activation population mismatch at layer {layer}")
        if [int(row["activation_artifact_row"]) for row in subset] != list(range(517)):
            raise RuntimeError(f"activation artifact row order mismatch at layer {layer}")

    aggregate = []
    layer_results = []
    all_artifacts = {}
    for layer in LAYERS:
        detail_path = OUT / f"model2_layer_{layer}_source_reconstruction.json"
        detail = json.loads(detail_path.read_text(encoding="utf-8"))
        rows = read_csv(ACT / f"model2_layer_{layer}_reconstruction_rows.csv")
        if detail["status"] != "COMPLETE" or detail["nan_or_inf_found"] or len(rows) != 517:
            raise RuntimeError(f"incomplete/nonfinite reconstruction at layer {layer}")
        source_metrics = []
        for source, expected_n in SOURCES.items():
            chosen = [row for row in rows if row["source"] == source]
            if len(chosen) != expected_n:
                raise RuntimeError(f"source mismatch for {source} at layer {layer}")
            rel = [float(row["relative_l2"]) for row in chosen]
            mse = [float(row["raw_mse"]) for row in chosen]
            l0 = [float(row["l0"]) for row in chosen]
            active = [float(row["active_fraction"]) for row in chosen]
            reported = rel + mse + l0 + active
            metrics = {
                "layer": layer, "hook": f"blocks.{layer}.hook_resid_post", "source": source, "n": len(chosen),
                "relative_l2_mean": statistics.fmean(rel), "relative_l2_median": statistics.median(rel),
                "relative_l2_population_sd": statistics.pstdev(rel), "relative_l2_min": min(rel), "relative_l2_max": max(rel),
                "raw_mse_mean": statistics.fmean(mse), "mean_l0": statistics.fmean(l0),
                "mean_active_feature_fraction": statistics.fmean(active),
                "nan_count_all_reported_metrics": sum(math.isnan(value) for value in reported),
                "inf_count_all_reported_metrics": sum(math.isinf(value) for value in reported),
                "relative_l2_gate_strict_less_than": 0.65, "active_fraction_gate_less_than_or_equal": 0.20,
                "relative_l2_pass": statistics.fmean(rel) < 0.65,
                "active_fraction_pass": statistics.fmean(active) <= 0.20,
            }
            aggregate.append(metrics)
            source_metrics.append(metrics)
        technical = any(row["nan_count_all_reported_metrics"] or row["inf_count_all_reported_metrics"] for row in source_metrics)
        reconstruction_fail = any(not row["relative_l2_pass"] for row in source_metrics)
        sparsity_fail = any(not row["active_fraction_pass"] for row in source_metrics)
        status = "TECHNICAL_FAILURE" if technical else "NON_VIABLE_RECONSTRUCTION" if reconstruction_fail else "NON_VIABLE_SPARSITY" if sparsity_fail else "VIABLE"
        artifacts = {}
        for name, item in detail["artifacts"].items():
            verified = file_hash(ROOT / item["path"])
            if verified != item["sha256"]:
                raise RuntimeError(f"artifact hash mismatch: layer {layer} {name}")
            artifacts[name] = {**item, "verified_sha256": verified}
        all_artifacts[str(layer)] = artifacts
        layer_results.append({
            "layer": layer, "hook": f"blocks.{layer}.hook_resid_post", "sae_id": detail["sae_id"],
            "params_sha256": detail["params_sha256"], "status": status, "required_sources": source_metrics,
            "all_required_source_reconstruction_pass": not reconstruction_fail,
            "all_required_source_sparsity_pass": not sparsity_fail, "technical_failure": technical,
        })

    aggregate_path = OUT / "model2_source_reconstruction.csv"
    write_csv(aggregate_path, aggregate)
    viable = [row["layer"] for row in layer_results if row["status"] == "VIABLE"]
    status = "PASS" if viable else "NO_VIABLE_LAYER"
    state = "LAYER_VIABILITY_COMPLETE" if viable else "STOPPED_NO_VIABLE_LAYER"
    viability = {
        "schema_version": "phase21_model2_layer_viability_v1", "status": status, "state": state,
        "decision_scope": "SAE reconstruction and sparsity viability only; no layer ranking or security feature discovery",
        "frozen_model": {"id": "google/gemma-2-9b-it", "revision": "11c9b309abf73637e4b6f9a3fa1e92e615547819"},
        "frozen_sae_repository": {"id": "google/gemma-scope-9b-it-res", "revision": "e86af97a5b6fbbccca28ab654f2fda1b0768f770"},
        "gates": {"mean_relative_l2_per_required_source": "strictly less than 0.65", "mean_active_feature_fraction_per_required_source": "less than or equal to 0.20", "relative_l2_row_formula": "||h - SAE.decode(SAE.encode(h))||_2 / ||h||_2", "population_standard_deviation_ddof": 0},
        "source_populations": {
            "CVE_VULNERABLE": {"n": 128, "authorization": "development train+validation vulnerable code; planned CWE-120/CWE-327/CWE-89 only"},
            "CVE_FIXED": {"n": 128, "authorization": "exactly paired development train+validation fixed code"},
            "MODEL2_B0": {"n": 261, "authorization": "frozen Model2 B0 development generations for planned source CWEs; reconstruction is not conditioned on scanner eligibility"},
        },
        "layers": layer_results, "viable_layers": viable, "viable_layer_ranking": None,
        "cwe_327_support_note": "DISCOVERY_USABLE_SOURCE_MATCHED_SUPPORT_LIMITED: B0 source n=97; scanner-eligible n=46; target-positive n=3; target-safe n=43. This does not alter reconstruction viability and must constrain later discovery interpretation.",
        "artifacts": all_artifacts,
        "provenance": {
            "source_manifest": {"path": relative(source_manifest), "sha256": file_hash(source_manifest)},
            "activation_manifest": {"path": relative(activation_manifest), "sha256": file_hash(activation_manifest)},
            "activation_run_manifest": {"path": relative(activation_run), "sha256": file_hash(activation_run)},
            "scanner_provenance": {"path": relative(scanner_provenance), "sha256": file_hash(scanner_provenance)},
            "source_reconstruction_csv": {"path": relative(aggregate_path), "sha256": file_hash(aggregate_path)},
        },
        "security_feature_ranking_run": False, "top200_created": False, "causal_intervention_run": False,
        "steered_generation_run": False, "scanner_run_during_this_substage": False, "heldout_used": False, "model1_modified": False,
    }
    viability_path = OUT / "model2_layer_viability.json"
    write_json(viability_path, viability)
    checkpoint = {
        "schema_version": "phase21_model2_layer_viability_checkpoint_v1", "status": status, "state": state,
        "source_records": 517, "completed_layers": list(LAYERS), "viable_layers": viable,
        "layer_statuses": {str(row["layer"]): row["status"] for row in layer_results},
        "source_manifest_sha256": file_hash(source_manifest), "activation_manifest_sha256": file_hash(activation_manifest),
        "source_reconstruction_sha256": file_hash(aggregate_path), "layer_viability_sha256": file_hash(viability_path),
        "next_stage": "feature discovery under a separately frozen protocol",
        "feature_ranking_run": False, "causal_intervention_run": False, "scanner_run": False, "heldout_used": False,
    }
    checkpoint_path = OUT / "model2_layer_viability_checkpoint.json"
    write_json(checkpoint_path, checkpoint)
    print(json.dumps({"status": status, "state": state, "viable_layers": viable, "layer_statuses": checkpoint["layer_statuses"], "source_reconstruction_sha256": checkpoint["source_reconstruction_sha256"], "layer_viability_sha256": checkpoint["layer_viability_sha256"], "checkpoint_sha256": file_hash(checkpoint_path)}, indent=2))


if __name__ == "__main__":
    main()
