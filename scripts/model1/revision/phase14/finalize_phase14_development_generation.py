#!/usr/bin/env python3
"""Freeze Phase 14 development-generation summary and scan-transfer manifest."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "revision/model1/phase14/outputs"
SUBSET = ROOT / "revision/model1/phase13/outputs/baseline_selection_subset.json"
SCANNER = ROOT / "phases/phase9/colab_scan_phase9.py"
SCANNER_SHA = "60cedd3d6ec2d19e5e7e5ba4d709230efdd5d9478047492fefda4dd88d9afbb7"
SUBSET_SHA = "3f023091c601fd6b4ffe8c074ad927f4bd8e32d211e0a98d81d7f486aa355a0d"
SUMMARY = OUT / "phase14_development_generation_summary.json"
TRANSFER = OUT / "phase14_development_scan_transfer_manifest.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def file_ref(path: Path, expected: str | None = None) -> dict[str, Any]:
    require(path.is_file(), f"missing file: {rel(path)}")
    digest = sha(path)
    if expected:
        require(digest == expected, f"hash mismatch: {rel(path)}")
    return {"path": rel(path), "sha256": digest}


CONDITIONS = [
    {
        "key": "b1cwe", "method": "B1-CWE", "information_tier": "ORACLE_CWE", "seed": 42,
        "output": OUT / "b1cwe_dev_outputs.json", "output_sha": "c824c975f6e4fc14ce12e97a5a3576cfc5865f3b13b3635611952ccddab852dc",
        "run": OUT / "b1cwe_dev_run_manifest.json", "run_sha": "f985f8fab23556d93629d79a4d07a02725e82326ceff398f77899b86af586f84",
        "validation": OUT / "b1cwe_structural_validation.json", "runner": ROOT / "revision/model1/phase14/scripts/run_b1cwe_dev.py",
        "config_refs": [OUT / "b1cwe_method_spec.json", OUT / "b1cwe_guidance_map.json"],
    },
    {
        "key": "rci1", "method": "RCI-1", "information_tier": "METADATA_FREE", "seed": 42,
        "output": OUT / "rci1_dev_outputs.json", "output_sha": "619511be08d42fa9b00b8b4fe60e0ea926b6764cda81618a116f4a5bb47b8c21",
        "run": OUT / "rci1_dev_run_manifest.json", "run_sha": "c3e1b0977d9645dd22267a1e3d17945c410c7ea1b1027adf0f476ffffa243ecc",
        "validation": OUT / "rci1_structural_validation.json", "runner": ROOT / "revision/model1/phase14/scripts/run_rci_dev.py",
        "config_refs": [OUT / "rci_method_spec.json"],
    },
    {
        "key": "rci2", "method": "RCI-2", "information_tier": "METADATA_FREE", "seed": 42,
        "output": OUT / "rci2_dev_outputs.json", "output_sha": "32a068c3326f10df68e1a15f2d2a9a2f60d0747c0e0cf3242c99d4c85be6cd7e",
        "run": OUT / "rci2_dev_run_manifest.json", "run_sha": "32e3822c1f0cfe267f431daeda0e38d26cc27dc44d9e6680586591df423f8482",
        "validation": OUT / "rci2_structural_validation.json", "runner": ROOT / "revision/model1/phase14/scripts/run_rci_dev.py",
        "config_refs": [OUT / "rci_method_spec.json"],
    },
    *[
        {
            "key": f"caa_l{layer}", "method": "CAA-CWE-StageA", "information_tier": "ORACLE_CWE", "seed": 42,
            "layer": layer, "multiplier": 1.0,
            "output": OUT / f"caa_stage_a_layer{layer}_mult1p0_seed42.json",
            "output_sha": {16:"75557e7841cc5d252551df36ffd32db76525418a9f2082ebca4b1990d55745e0",19:"b34271c0086248050c7fcfa38c9d4d25ddcc2fecf0dd590fab0ebdc2bfafc408",23:"d8e03698d948d8575ae86b0b3c85cdd5d46a8cf938cdf2106d5271575d2240af"}[layer],
            "run": OUT / f"caa_stage_a_layer{layer}_mult1p0_seed42_run_manifest.json",
            "run_sha": {16:"40b3db73fc8dc1f6a180db4e3e00cd008b21fe0ca45890c55565b63d1cdc8ea9",19:"a97faf06e5281a73d3b378d71e8b6512a71bf87b18e71fdc6273759735f95ab4",23:"c9893a8668109fa09f99ca513fdbbd292e44e75929ca88b09c70a3b55fed0108"}[layer],
            "validation": OUT / f"caa_stage_a_layer{layer}_structural_validation.json",
            "runner": ROOT / "revision/model1/phase14/scripts/run_caa_cwe_dev.py",
            "config_refs": [OUT / "caa_vector_manifest.json", OUT / "dense_hook_readiness_revision.json", ROOT / "revision/model1/phase14/scripts/caa_position_mask.py"],
        } for layer in (16, 19, 23)
    ],
    {
        "key": "alwayson", "method": "B*-AlwaysOn-L19", "information_tier": "METADATA_FREE", "seed": 42,
        "layer": 19, "alpha_by_feature": {"14193":40.0,"16897":40.0,"11462":40.0,"151":40.0},
        "output": OUT / "bstar_alwayson_l19_smoke_outputs.json", "output_sha": "96da7faf1b7655f45342290d46e1bf58d437fe67e7a266967909cc782ae66db8",
        "run": OUT / "bstar_alwayson_l19_smoke_run_manifest.json", "run_sha": "cabe828bb466eb153ca46597fa7c681c64bec4559298868d95429347ea05fcda",
        "validation": OUT / "bstar_alwayson_l19_structural_validation.json", "runner": ROOT / "revision/model1/phase14/scripts/smoke_bstar_alwayson_l19.py",
        "prompt_manifest": OUT / "bstar_alwayson_l19_smoke_manifest.json",
        "config_refs": [OUT / "bstar_alwayson_l19_smoke_manifest.json"],
    },
]


def final_records(key: str, path: Path) -> list[dict[str, Any]]:
    data = read(path)
    return data["final_outputs"] if key.startswith("rci") else data


def main() -> None:
    require(not SUMMARY.exists() and not TRANSFER.exists(), "finalization artifacts already exist; immutable review required")
    file_ref(SUBSET, SUBSET_SHA)
    file_ref(SCANNER, SCANNER_SHA)
    summary_rows = []
    transfer_rows = []

    for item in CONDITIONS:
        output_ref = file_ref(item["output"], item["output_sha"])
        run_ref = file_ref(item["run"], item["run_sha"])
        validation_ref = file_ref(item["validation"])
        runner_ref = file_ref(item["runner"])
        records = final_records(item["key"], item["output"])
        require(len(records) == (20 if item["key"] == "alwayson" else 120), f"record count mismatch: {item['key']}")
        statuses: dict[str, int] = {}
        for row in records:
            status = str(row.get("generation_status", "UNKNOWN"))
            statuses[status] = statuses.get(status, 0) + 1
        strict_empty = sum(row.get("generated_code", "") == "" for row in records)
        whitespace = sum(row.get("generated_code", "") != "" and not row.get("generated_code", "").strip() for row in records)
        active = sum(bool(row.get("intervention_applied", False)) for row in records)
        if item["key"] == "alwayson":
            active = len(records)
        fallback = sum(bool(row.get("fallback_status")) for row in records)
        tokens = sum(int(row.get("generated_token_count", 0) or 0) for row in records)
        validation = read(item["validation"])
        model_calls = validation.get("actual_model_call_count")
        if item["key"].startswith("rci"):
            tokens = int(validation["generated_token_count"])
        summary_rows.append({
            "method": item["method"], "variant_or_layer": item.get("layer", item["key"]),
            "seed": item["seed"], "information_tier": item["information_tier"],
            "record_count": len(records), "routed_or_intervention_active_count": active,
            "no_intervention_or_fallback_count": len(records)-active if item["key"].startswith("caa") else fallback,
            "successful_count": statuses.get("COMPLETED", 0), "strict_empty_count": strict_empty,
            "whitespace_only_count": whitespace,
            "generation_failures": sum(v for k,v in statuses.items() if k not in {"COMPLETED","EXTRACTION_FAILED_INVALID_OR_EMPTY","SKIPPED_PREREQUISITE_FAILURE"}),
            "preserved_extraction_or_prerequisite_failure_finals": statuses.get("EXTRACTION_FAILED_INVALID_OR_EMPTY",0)+statuses.get("SKIPPED_PREREQUISITE_FAILURE",0),
            "hook_failures": int(validation.get("hook_failure_count", validation.get("hook_contract_failure_count", 0))),
            "generated_token_count": tokens, "model_call_count": model_calls,
            "checkpoint_count": int(validation.get("checkpoint_count", len(records))), "resume_count": int(validation.get("resume_count", 0)),
            "elapsed_seconds": validation.get("elapsed_seconds"),
            "elapsed_time_warning": validation.get("elapsed_time_warning"),
            "output_sha256": output_ref["sha256"], "run_manifest_sha256": run_ref["sha256"],
            "structural_validation_sha256": validation_ref["sha256"], "structural_validation_status": validation["status"],
        })

        prompt_manifest = item.get("prompt_manifest", SUBSET)
        prompt_ref = file_ref(prompt_manifest, "3ad3dfbd5b034d3e2f4531eb399b87fae96d624b8b3dbacf4a8053425c848000" if item["key"] == "alwayson" else SUBSET_SHA)
        transfer = {
            "method": item["method"], "information_tier": item["information_tier"], "seed": item["seed"],
            "layer": item.get("layer"), "multiplier": item.get("multiplier"),
            "output_path": output_ref["path"], "output_sha256": output_ref["sha256"], "record_count": len(records),
            "prompt_manifest_path": prompt_ref["path"], "prompt_manifest_sha256": prompt_ref["sha256"],
            "run_manifest_path": run_ref["path"], "run_manifest_sha256": run_ref["sha256"],
            "runner_path": runner_ref["path"], "runner_sha256": runner_ref["sha256"],
            "structural_validation_path": validation_ref["path"], "structural_validation_sha256": validation_ref["sha256"],
            "method_config_and_vector_refs": [file_ref(path) for path in item["config_refs"]],
            "frozen_scanner_source_path": rel(SCANNER), "frozen_scanner_source_sha256": SCANNER_SHA,
            "scan_status": "NOT_STARTED",
        }
        if "alpha_by_feature" in item:
            transfer["alpha_by_feature"] = item["alpha_by_feature"]
        transfer_rows.append(transfer)

    summary_doc = {
        "schema_version": "phase14_development_generation_summary_v1", "phase": 14,
        "status": "DEVELOPMENT_GENERATION_COMPLETE_AWAITING_SCANS_AND_SELECTION",
        "condition_count": 7, "conditions": summary_rows,
        "rci_audit_sheet": file_ref(OUT / "rci_human_audit_sheet.json", "fbfc190316406567fc480a26853c37903b7c985f31e2281100f4f8d116ccb2fa"),
        "held_out_prompt_or_result_file_accessed": False, "scanner_executed": False,
        "security_metrics_computed": False, "rci_selection_performed": False,
        "caa_stage_a_selection_performed": False, "caa_stage_b_executed": False,
    }
    SUMMARY.write_text(json.dumps(summary_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    transfer_doc = {
        "schema_version": "phase14_development_scan_transfer_manifest_v1", "phase": 14,
        "status": "FROZEN_AWAITING_COLAB_SCANS", "entry_count": 7, "entries": transfer_rows,
        "all_scan_statuses": "NOT_STARTED", "generation_files_immutable_after_inclusion": True,
        "held_out_prompt_or_result_file_accessed": False, "scanner_executed": False,
    }
    TRANSFER.write_text(json.dumps(transfer_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status":"PASS","summary_sha256":sha(SUMMARY),"transfer_sha256":sha(TRANSFER),"entries":7}, indent=2))


if __name__ == "__main__":
    main()
