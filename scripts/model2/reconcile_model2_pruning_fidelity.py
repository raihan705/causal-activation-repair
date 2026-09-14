#!/usr/bin/env python
"""Reconcile Model2 pre-causal pruning with the executed Model1 semantics."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "revision/model2/phase21/outputs"
RANKINGS = OUT / "model2_statistical_feature_rankings.json"
STRICT_PRUNED = OUT / "model2_pruned_candidates.json"
STRICT_CAUSAL = OUT / "model2_causal_candidate_manifest.json"
PARTITION = OUT / "model2_feature_partition_manifest.json"
B0 = OUT / "model2_b0_dev_outputs.json"
SCAN = OUT / "model2_b0_dev_scan.json"
MODEL1_PRUNING = ROOT / "phases/phase4/prune_feature_candidates.py"
MODEL1_CAUSAL = ROOT / "phases/phase4/validate_feature_causality_multilayer.py"
SCANNER = ROOT / "phases/phase9/colab_scan_phase9.py"
AUDIT_CSV = OUT / "model2_pruning_fidelity_audit.csv"
SUMMARY_JSON = OUT / "model2_pruning_fidelity_summary.json"
FIDELITY_CAUSAL = OUT / "model2_causal_candidate_manifest_model1_fidelity.json"
PROTOCOL = OUT / "model2_causal_validation_protocol.json"
CHECKPOINT = OUT / "model2_causal_protocol_checkpoint.json"
TARGETS = ("CWE-120", "CWE-327", "CWE-89")
LAYERS = (9, 20, 31)

PRESERVED = {
    "model2_feature_partition_manifest.json": "4aefb5aa35ee7845c556bac03a4e64a29d3b41422824a18767e36a5800e710c2",
    "model2_feature_latent_manifest.csv": "4501130a9e11d0f9f4f8318ff214107f6d4c561839079cc5b7547e12a8e75268",
    "model2_safe_unsafe_summary.csv": "2c3e3ab83264a8b1a05b87a7abbc79ddb87f090a2eeda70a777d83818764d1f9",
    "model2_statistical_feature_rankings.json": "3a2e5eae10f0cce381853024b3ad8ef3523b0cac87531cdb296d127c20bc4b78",
    "model2_statistical_feature_rankings_summary.csv": "c34a844af217d4138ca2b21c288dafd7e2f703e44f7cd38b9a527cf099033931",
    "model2_pruned_candidates.json": "fba7608cd442d73036ad162bf1d0b83f19573b129c986035b1ff519a35a9c5e4",
    "model2_causal_candidate_manifest.json": "026aab1dcdea57e641c940aa012bf7b41111bdc863ddb22fa771650e4f37ba42",
    "model2_feature_discovery_checkpoint.json": "cb189cfbaef1fa5f34621e13d76b06903fc3dc20033248281f5a2faa8c989007",
}


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()


def canonical_id_hash(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def atomic_bytes(path: Path, payload: bytes) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_bytes(path, canonical_bytes(value))


def atomic_csv(path: Path, rows: list[dict]) -> None:
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def recover_model1_boolean() -> tuple[str, str]:
    text = MODEL1_PRUNING.read_text(encoding="utf-8")
    line = next(line.strip() for line in text.splitlines() if "if fu > FREQ_BOTH_MAX and fs > FREQ_BOTH_MAX:" in line)
    tree = ast.parse(text)
    matches = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.BoolOp) and isinstance(node.test.op, ast.And):
            source = ast.get_source_segment(text, node.test)
            if source and "FREQ_BOTH_MAX" in source and "fu" in source and "fs" in source:
                matches.append(source)
    require(matches == ["fu > FREQ_BOTH_MAX and fs > FREQ_BOTH_MAX"], "could not uniquely recover executed Model1 both-high Boolean")
    return line, matches[0]


def pruning_reasons(record: dict, rule: str) -> list[str]:
    fu = record["unsafe_activation_frequency"]
    fs = record["safe_activation_frequency"]
    score = record["composite_score"]
    magnitude = record["unsafe_mean_magnitude"]
    reasons = []
    if fu < 0.02:
        reasons.append("UNSAFE_FREQUENCY_BELOW_0.02")
    if rule == "RULE_STRICT_OR":
        if fu > 0.95:
            reasons.append("UNSAFE_FREQUENCY_ABOVE_0.95")
        if fs > 0.95:
            reasons.append("SAFE_FREQUENCY_ABOVE_0.95")
    elif rule == "RULE_MODEL1_EXECUTED":
        if fu > 0.95 and fs > 0.95:
            reasons.append("BOTH_FREQUENCIES_ABOVE_0.95")
    else:
        raise RuntimeError(f"unknown rule: {rule}")
    if score < 0.15:
        reasons.append("COMPOSITE_SCORE_BELOW_0.15")
    if math.isnan(score) or math.isinf(score):
        reasons.append("NON_FINITE_SCORE")
    if math.isnan(magnitude) or math.isinf(magnitude) or magnitude > 1e4:
        reasons.append("NON_FINITE_OR_EXTREME_UNSAFE_MEAN")
    return reasons


def freeze_prompts() -> tuple[dict, list[dict], int]:
    scan_rows = json.loads(SCAN.read_text(encoding="utf-8"))
    b0_rows = json.loads(B0.read_text(encoding="utf-8"))
    require([int(row["prompt_id"]) for row in scan_rows] == [int(row["prompt_id"]) for row in b0_rows], "B0 scan/output order mismatch")
    b0 = {int(row["prompt_id"]): row for row in b0_rows}
    unsafe = {}
    caps = {"CWE-120": 15, "CWE-327": 15, "CWE-89": 15}
    for target in TARGETS:
        eligible = [row for row in scan_rows if not row["skipped"] and target in row["vulnerable_cwes"]]
        selected = eligible[:min(caps[target], len(eligible))]
        unsafe[target] = [{
            "prompt_id": int(row["prompt_id"]), "source_index": int(b0[int(row["prompt_id"])]["source_index"]),
            "source_cwe": b0[int(row["prompt_id"])]["cwe_id"], "language": row["language"],
            "prompt_sha256": hashlib.sha256(b0[int(row["prompt_id"])]["prompt_text"].encode()).hexdigest(),
            "verified_target_finding": True,
        } for row in selected]
    safe_population = [row for row in scan_rows if not row["skipped"] and not row["findings"]]
    rng = np.random.default_rng(42)
    indices = rng.choice(len(safe_population), min(5, len(safe_population)), replace=False)
    safe_selected = []
    for index in indices:
        row = safe_population[int(index)]
        source = b0[int(row["prompt_id"])]
        safe_selected.append({
            "prompt_id": int(row["prompt_id"]), "source_index": int(source["source_index"]),
            "source_cwe": source["cwe_id"], "language": row["language"],
            "prompt_sha256": hashlib.sha256(source["prompt_text"].encode()).hexdigest(),
            "scanner_eligible": True, "scanner_findings": [],
        })
    return unsafe, safe_selected, len(safe_population)


def build_outputs() -> tuple[list[dict], dict, dict, dict, dict]:
    historical_line, historical_expression = recover_model1_boolean()
    rankings = json.loads(RANKINGS.read_text(encoding="utf-8"))
    strict_existing = json.loads(STRICT_PRUNED.read_text(encoding="utf-8"))
    audit_rows = []
    cells = {}
    corrected_cells = {}
    strict_labeled, model1_labeled = set(), set()
    changed_top10 = 0
    for target in TARGETS:
        cells[target] = {}
        corrected_cells[target] = {}
        for layer in LAYERS:
            key = f"L{layer}"
            records = rankings["cells"][target][key]
            require(len(records) == 200 and [row["rank"] for row in records] == list(range(1, 201)), f"ranking input drift {target}/{key}")
            strict = [row for row in records if not pruning_reasons(row, "RULE_STRICT_OR")]
            model1 = [row for row in records if not pruning_reasons(row, "RULE_MODEL1_EXECUTED")]
            require([row["feature_id"] for row in strict] == [row["feature_id"] for row in strict_existing["cells"][target][key]["candidates"]], f"STRICT_OR reconstruction mismatch {target}/{key}")
            strict_ids = [row["feature_id"] for row in strict]
            model1_ids = [row["feature_id"] for row in model1]
            added = [feature for feature in model1_ids if feature not in set(strict_ids)]
            removed = [feature for feature in strict_ids if feature not in set(model1_ids)]
            symmetric = sorted(set(strict_ids) ^ set(model1_ids))
            strict_top10 = strict[:10]
            model1_top10 = model1[:10]
            identical = [row["feature_id"] for row in strict_top10] == [row["feature_id"] for row in model1_top10]
            changed_top10 += int(not identical)
            strict_labeled.update((target, layer, feature) for feature in strict_ids)
            model1_labeled.update((target, layer, feature) for feature in model1_ids)
            audit_rows.append({
                "cwe_id": target, "layer": layer, "top200_input_count": 200,
                "strict_or_survivor_count": len(strict), "model1_executed_survivor_count": len(model1),
                "added_by_model1_executed_count": len(added), "removed_by_model1_executed_count": len(removed),
                "symmetric_difference_count": len(symmetric),
                "symmetric_difference_feature_ids": json.dumps(symmetric, separators=(",", ":")),
                "first10_identical": identical,
                "strict_or_first10_feature_ids": json.dumps([row["feature_id"] for row in strict_top10], separators=(",", ":")),
                "model1_executed_first10_feature_ids": json.dumps([row["feature_id"] for row in model1_top10], separators=(",", ":")),
            })
            cells[target][key] = {
                "top200_input_count": 200,
                "RULE_STRICT_OR": {"survivor_count": len(strict), "survivor_feature_ids": strict_ids, "first10": strict_top10},
                "RULE_MODEL1_EXECUTED": {"survivor_count": len(model1), "survivor_feature_ids": model1_ids, "first10": model1_top10},
                "added_by_model1_executed": added, "removed_by_model1_executed": removed,
                "symmetric_difference_feature_ids": symmetric, "first10_identical": identical,
            }
            corrected_cells[target][key] = {
                "status": "CAUSAL_VALIDATION_CANDIDATES_FROZEN",
                "pruning_rule": "RULE_MODEL1_EXECUTED_PRIMARY",
                "candidate_count": len(model1_top10),
                "candidates": [{
                    "cwe_id": target, "layer": layer, "feature_id": row["feature_id"],
                    "original_statistical_rank": row["rank"], "composite_score": row["composite_score"],
                    "unsafe_activation_frequency": row["unsafe_activation_frequency"],
                    "safe_activation_frequency": row["safe_activation_frequency"],
                    "pruning_status": "SURVIVED_RULE_MODEL1_EXECUTED",
                } for row in model1_top10],
            }
    intersection = len(strict_labeled & model1_labeled)
    union = len(strict_labeled | model1_labeled)
    summary = {
        "schema_version": "phase21_model2_pruning_fidelity_summary_v1", "status": "PASS",
        "source_ranking_sha256": file_hash(RANKINGS),
        "source_strict_pruned_sha256": file_hash(STRICT_PRUNED),
        "historical_model1_source": {"path": "phases/phase4/prune_feature_candidates.py", "sha256": file_hash(MODEL1_PRUNING)},
        "historical_model1_high_frequency_line_verbatim": historical_line,
        "historical_model1_high_frequency_boolean_expression_verbatim": historical_expression,
        "rules": {
            "RULE_STRICT_OR": "reject if unsafe_frequency > 0.95 OR safe_frequency > 0.95",
            "RULE_MODEL1_EXECUTED": "reject if unsafe_frequency > 0.95 AND safe_frequency > 0.95",
            "primary_replication_rule": "RULE_MODEL1_EXECUTED",
            "preserved_diagnostic_rule": "RULE_STRICT_OR",
        },
        "cells": cells,
        "aggregate": {"strict_or_labeled_candidate_count": len(strict_labeled), "model1_executed_labeled_candidate_count": len(model1_labeled), "intersection_count": intersection, "union_count": union, "total_candidate_set_jaccard": intersection / union, "cells_with_changed_first10": changed_top10},
        "causal_outcome_observed_before_correction": False,
    }
    corrected = {
        "schema_version": "phase21_model2_causal_candidate_manifest_model1_fidelity_v1",
        "status": "FROZEN_BEFORE_CAUSAL_INTERVENTION",
        "primary_pruning_rule": "RULE_MODEL1_EXECUTED",
        "historical_boolean_expression": historical_expression,
        "source_ranking_path": "revision/model2/phase21/outputs/model2_statistical_feature_rankings.json",
        "source_ranking_sha256": file_hash(RANKINGS),
        "pruning_fidelity_summary_path": "revision/model2/phase21/outputs/model2_pruning_fidelity_summary.json",
        "cells": corrected_cells,
        "candidate_meaning": "CAUSAL_VALIDATION_CANDIDATES_ONLY_NOT_VALIDATED_SECURITY_FEATURES",
        "causal_intervention_run": False, "alpha_selected_as_final_strength": False,
    }
    unsafe, safe, safe_population_count = freeze_prompts()
    candidate_total = sum(corrected_cells[target][f"L{layer}"]["candidate_count"] for target in TARGETS for layer in LAYERS)
    expected_by_target = {
        target: sum(corrected_cells[target][f"L{layer}"]["candidate_count"] for layer in LAYERS) * (len(unsafe[target]) + len(safe))
        for target in TARGETS
    }
    expected_total = sum(expected_by_target.values())
    protocol = {
        "schema_version": "phase21_model2_causal_validation_protocol_v1",
        "status": "FROZEN_BEFORE_CAUSAL_EXECUTION",
        "candidate_manifest_path": "revision/model2/phase21/outputs/model2_causal_candidate_manifest_model1_fidelity.json",
        "candidate_population": "first 10 RULE_MODEL1_EXECUTED survivors per CWE x layer",
        "targets": list(TARGETS), "layers": list(LAYERS), "candidate_count_total_across_cells": candidate_total,
        "screening_alpha": 20.0, "alpha_role": "MODEL1_MATCHED_CAUSAL_SCREENING_ONLY_NOT_FINAL_MODEL2_STRENGTH",
        "unsafe_prompts": unsafe,
        "unsafe_prompt_selection": "first min(15,N) verified target-positive, scanner-eligible B0 development records in frozen scan/output order",
        "safe_corruption_population_count": safe_population_count,
        "safe_corruption_prompts": safe,
        "safe_prompt_ids_sha256": canonical_id_hash([row["prompt_id"] for row in safe]),
        "safe_prompt_selection": "eligible, findings=[], non-skipped verified B0 development rows; np.random.default_rng(42).choice(population,5,replace=False)",
        "model": {"id": "google/gemma-2-9b-it", "revision": "11c9b309abf73637e4b6f9a3fa1e92e615547819"},
        "sae": {"repository": "google/gemma-scope-9b-it-res", "revision": "e86af97a5b6fbbccca28ab654f2fda1b0768f770", "layer_ids": {"9": "layer_9/width_16k/average_l0_47", "20": "layer_20/width_16k/average_l0_47", "31": "layer_31/width_16k/average_l0_43"}},
        "input_rendering": {"source_field": "prompt_text", "chat_template": False, "truncation": True, "max_input_tokens": 512, "model1_matched_replication": True},
        "hook": {"hook_points": {"9": "blocks.9.hook_resid_post", "20": "blocks.20.hook_resid_post", "31": "blocks.31.hook_resid_post"}, "semantics": "at every forward-hook call encode only hidden[:, -1:, :] as float32, add +20 to selected latent, decode, cast to hidden dtype, and replace only that current last hook position", "feature_direction": "positive additive", "no_model1_feature_or_layer_transfer": True},
        "generation": {"max_new_tokens": 256, "temperature": 0.2, "top_p": 0.95, "do_sample": True, "seed": 42, "seed_reset": "torch.manual_seed(42) immediately before every individual generation", "one_steered_generation_per_candidate_prompt": True, "unsteered_generation_in_screen": False},
        "validity": {"function": "nonempty stripped length >=10 and contains one of: def , int , char , return, if , for , {, (", "source": "phases/phase4/validate_feature_causality_multilayer.py"},
        "metric_semantics": {"unsafe_repair": "valid steered output whose verified scan omits target CWE; unsafe prompts are baseline vulnerable by frozen verified scan", "unsafe_invalid": "invalid unsafe steered output increments n_invalid and cannot repair", "safe_corruption": "invalid safe steered output OR any verified scanner finding", "repair_rate": "n_repair / n_unsafe_tested", "corruption_rate": "n_corrupt / n_safe_tested", "invalid_rate": "n_invalid_unsafe / n_unsafe_tested"},
        "retention": {"validated_iff": "repair_rate > 0 AND corruption_rate < 0.20 AND invalid_rate < 0.50", "repair_rate_strict_min": 0.0, "corruption_rate_strict_max": 0.20, "invalid_rate_strict_max": 0.50},
        "expected_generation_count_by_target": expected_by_target, "expected_generation_count_total": expected_total,
        "causal_generation_output_schema": {"container": "JSON object with schema/version, immutable provenance, records, completion summary", "record_required_fields": ["record_id", "cwe_id", "layer", "feature_id", "original_statistical_rank", "screening_alpha", "prompt_id", "prompt_population", "source_index", "language", "prompt_sha256", "seed", "generation_settings", "generated_text", "generated_code", "generated_token_count", "validity", "generation_status", "exception", "hook_call_count", "hook_input_shapes", "hook_finite", "model_id", "model_revision", "sae_id", "sae_params_sha256", "candidate_manifest_sha256", "protocol_sha256"]},
        "immutable_colab_scanner_manifest": {"required_semgrep_version": "1.175.0", "scanner_script_path": "phases/phase9/colab_scan_phase9.py", "scanner_script_sha256": file_hash(SCANNER), "scanner_provenance_path": "revision/model2/phase21/outputs/model2_scanner_provenance.json", "scanner_provenance_sha256": file_hash(OUT / "model2_scanner_provenance.json"), "registry_config": "p/security-audit", "rules_changed": False, "language_and_skip_policy_changed": False, "input_requirement": "completed causal generation artifact with exact frozen hash", "output_requirement": "one scan result per generation record keyed by record_id plus immutable return manifest", "scanner_execution_in_protocol_freeze": False},
        "causal_generation_run": False, "scanner_run": False, "heldout_used": False,
    }
    checkpoint = {
        "schema_version": "phase21_model2_causal_protocol_checkpoint_v1", "status": "PASS",
        "state": "PRUNING_FIDELITY_COMPLETE",
        "actual_model1_boolean_recovered": True, "both_pruning_rules_reproduced": True,
        "primary_pruning_rule": "RULE_MODEL1_EXECUTED", "strict_or_preserved_as_diagnostic": True,
        "corrected_candidate_manifest_sha256": None, "causal_protocol_sha256": None,
        "expected_generation_count": expected_total,
        "screening_alpha": 20.0, "screening_alpha_is_final_model2_strength": False,
        "causal_generation_run": False, "scanner_run": False, "heldout_used": False, "model1_modified": False,
        "next_execution_authorized": False,
    }
    return audit_rows, summary, corrected, protocol, checkpoint


def main() -> None:
    for name, expected in PRESERVED.items():
        require(file_hash(OUT / name) == expected, f"preserved Model2 artifact hash drift: {name}")
    for path in (AUDIT_CSV, SUMMARY_JSON, FIDELITY_CAUSAL, PROTOCOL, CHECKPOINT):
        require(not path.exists(), f"new reconciliation output already exists: {path.name}")
    first = build_outputs()
    second = build_outputs()
    require(canonical_bytes(first[1]) == canonical_bytes(second[1]), "summary deterministic rebuild failed")
    require(canonical_bytes(first[2]) == canonical_bytes(second[2]), "corrected candidate deterministic rebuild failed")
    require(canonical_bytes(first[3]) == canonical_bytes(second[3]), "protocol deterministic rebuild failed")
    audit_rows, summary, corrected, protocol, checkpoint = first
    atomic_csv(AUDIT_CSV, audit_rows)
    summary["audit_csv_sha256"] = file_hash(AUDIT_CSV)
    atomic_json(SUMMARY_JSON, summary)
    corrected["pruning_fidelity_summary_sha256"] = file_hash(SUMMARY_JSON)
    corrected_bytes = canonical_bytes(corrected)
    require(corrected_bytes == canonical_bytes({**corrected}), "corrected candidate byte rebuild failed")
    atomic_bytes(FIDELITY_CAUSAL, corrected_bytes)
    protocol["candidate_manifest_sha256"] = file_hash(FIDELITY_CAUSAL)
    protocol_bytes = canonical_bytes(protocol)
    require(protocol_bytes == canonical_bytes({**protocol}), "protocol byte rebuild failed")
    atomic_bytes(PROTOCOL, protocol_bytes)
    checkpoint["corrected_candidate_manifest_sha256"] = file_hash(FIDELITY_CAUSAL)
    checkpoint["causal_protocol_sha256"] = file_hash(PROTOCOL)
    checkpoint["pruning_fidelity_summary_sha256"] = file_hash(SUMMARY_JSON)
    checkpoint["pruning_fidelity_audit_sha256"] = file_hash(AUDIT_CSV)
    atomic_json(CHECKPOINT, checkpoint)
    for name, expected in PRESERVED.items():
        require(file_hash(OUT / name) == expected, f"preserved artifact changed during reconciliation: {name}")
    print(json.dumps({
        "status": "PASS", "state": checkpoint["state"],
        "historical_expression": summary["historical_model1_high_frequency_boolean_expression_verbatim"],
        "aggregate": summary["aggregate"],
        "audit_rows": audit_rows,
        "corrected_candidate_manifest_sha256": checkpoint["corrected_candidate_manifest_sha256"],
        "unsafe_ids": {target: [row["prompt_id"] for row in protocol["unsafe_prompts"][target]] for target in TARGETS},
        "safe_ids": [row["prompt_id"] for row in protocol["safe_corruption_prompts"]],
        "expected_generation_count": checkpoint["expected_generation_count"],
        "protocol_sha256": checkpoint["causal_protocol_sha256"],
        "checkpoint_sha256": file_hash(CHECKPOINT),
    }, indent=2))


if __name__ == "__main__":
    main()
