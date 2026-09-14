#!/usr/bin/env python3
"""Phase 14 development-scan orchestration for the frozen Phase 9 scanner.

This wrapper validates provenance, materializes deterministic RCI final-output
adapters, invokes the unchanged ``scan_file`` function sequentially, and
maintains an atomic resume manifest.  It does not implement scanner rules or
compute security metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


EXPECTED_TRANSFER_SHA256 = (
    "818d3dbd69ddaf55760e12466fc353312303ff0976937ad302c9b454c1ee4ddc"
)
EXPECTED_GENERATION_SUMMARY_SHA256 = (
    "71d80b573b021ab1ca9fc575651a463e1cb796bb1389b646222ae1f2baa8afbd"
)
EXPECTED_SCANNER_SHA256 = (
    "60cedd3d6ec2d19e5e7e5ba4d709230efdd5d9478047492fefda4dd88d9afbb7"
)
EXPECTED_SUBSET_SHA256 = (
    "3f023091c601fd6b4ffe8c074ad927f4bd8e32d211e0a98d81d7f486aa355a0d"
)
EXPECTED_ALWAYSON_MANIFEST_SHA256 = (
    "3ad3dfbd5b034d3e2f4531eb399b87fae96d624b8b3dbacf4a8053425c848000"
)

CONDITIONS = [
    {
        "key": "b1cwe",
        "method": "B1-CWE",
        "layer": None,
        "multiplier": None,
        "tier": "ORACLE_CWE",
        "records": 120,
        "icd_name": "b1cwe_dev_icd.json",
    },
    {
        "key": "rci1",
        "method": "RCI-1",
        "layer": None,
        "multiplier": None,
        "tier": "METADATA_FREE",
        "records": 120,
        "icd_name": "rci1_dev_icd.json",
        "adapter_name": "rci1_scanner_input.json",
        "variant": "rci1",
    },
    {
        "key": "rci2",
        "method": "RCI-2",
        "layer": None,
        "multiplier": None,
        "tier": "METADATA_FREE",
        "records": 120,
        "icd_name": "rci2_dev_icd.json",
        "adapter_name": "rci2_scanner_input.json",
        "variant": "rci2",
    },
    {
        "key": "caa_l16_m1",
        "method": "CAA-CWE-StageA",
        "layer": 16,
        "multiplier": 1.0,
        "tier": "ORACLE_CWE",
        "records": 120,
        "icd_name": "caa_stage_a_layer16_mult1p0_seed42_icd.json",
    },
    {
        "key": "caa_l19_m1",
        "method": "CAA-CWE-StageA",
        "layer": 19,
        "multiplier": 1.0,
        "tier": "ORACLE_CWE",
        "records": 120,
        "icd_name": "caa_stage_a_layer19_mult1p0_seed42_icd.json",
    },
    {
        "key": "caa_l23_m1",
        "method": "CAA-CWE-StageA",
        "layer": 23,
        "multiplier": 1.0,
        "tier": "ORACLE_CWE",
        "records": 120,
        "icd_name": "caa_stage_a_layer23_mult1p0_seed42_icd.json",
    },
    {
        "key": "alwayson_l19",
        "method": "B*-AlwaysOn-L19",
        "layer": 19,
        "multiplier": None,
        "tier": "METADATA_FREE",
        "records": 20,
        "icd_name": "bstar_alwayson_l19_smoke_icd.json",
    },
]


class ValidationError(RuntimeError):
    """Raised when a frozen input or completed result fails validation."""


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, indent=2, ensure_ascii=False).encode("utf-8")


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValidationError(f"Cannot parse JSON {path}: {exc}") from exc


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise ValidationError(f"Unsafe stale atomic-write file exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, json_bytes(value))


def resolve_path(workdir: Path, recorded: str) -> Path:
    candidate = Path(recorded)
    choices = []
    if candidate.is_absolute():
        choices.append(candidate)
    else:
        choices.extend((Path.cwd() / candidate, workdir / candidate, workdir / candidate.name))
    for choice in choices:
        if choice.exists():
            return choice.resolve()
    raise ValidationError(
        f"Required file not found for {recorded}; checked: "
        + ", ".join(str(choice) for choice in choices)
    )


def require_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256_path(path)
    if actual != expected:
        raise ValidationError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")


def unique_ids(records: list[dict[str, Any]], label: str) -> list[Any]:
    try:
        ids = [record["prompt_id"] for record in records]
    except Exception as exc:
        raise ValidationError(f"{label} is missing prompt_id") from exc
    if len(ids) != len(set(ids)):
        raise ValidationError(f"{label} contains duplicate prompt IDs")
    return ids


def load_scanner(scanner_path: Path):
    spec = importlib.util.spec_from_file_location("phase14_frozen_phase9_scanner", scanner_path)
    if spec is None or spec.loader is None:
        raise ValidationError(f"Cannot import frozen scanner: {scanner_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scan_file = getattr(module, "scan_file", None)
    if not callable(scan_file):
        raise ValidationError("Frozen scanner has no callable scan_file")
    parameters = list(inspect.signature(scan_file).parameters)
    if parameters != ["out_path", "icd_path"]:
        raise ValidationError(f"Unexpected frozen scan_file signature: {parameters}")
    return scan_file


def semgrep_version(required: bool) -> str:
    try:
        result = subprocess.run(
            ["semgrep", "--version"], capture_output=True, text=True, timeout=30, check=False
        )
    except Exception as exc:
        if required:
            raise ValidationError(f"Semgrep is unavailable: {exc}") from exc
        return "NOT_AVAILABLE_IN_PREFLIGHT_ENVIRONMENT"
    version = (result.stdout or result.stderr).strip()
    if result.returncode != 0 or not version:
        if required:
            raise ValidationError(f"Semgrep version check failed: {version}")
        return "NOT_AVAILABLE_IN_PREFLIGHT_ENVIRONMENT"
    return version


def direct_records(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValidationError(f"{label} must be a JSON list of records")
    return value


def rci_final_records(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or not isinstance(value.get("final_outputs"), list):
        raise ValidationError(f"{label} must contain final_outputs")
    records = value["final_outputs"]
    if not all(isinstance(item, dict) for item in records):
        raise ValidationError(f"{label} final_outputs contains a non-record")
    return records


def make_rci_adapter(
    finals: list[dict[str, Any]], subset_records: list[dict[str, Any]], variant: str
) -> list[dict[str, Any]]:
    metadata = {record["prompt_id"]: record for record in subset_records}
    adapted = []
    for final in finals:
        prompt_id = final["prompt_id"]
        if prompt_id not in metadata:
            raise ValidationError(f"RCI {variant} has unknown prompt ID {prompt_id}")
        source = metadata[prompt_id]
        code = final.get("generated_code")
        if not isinstance(code, str):
            raise ValidationError(f"RCI {variant} prompt {prompt_id} generated_code is not text")
        adapted.append(
            {
                "prompt_id": prompt_id,
                "cwe_id": source["cwe_id"],
                "language": source["language"],
                "generated_code": code,
                "rci_variant": variant,
            }
        )
    return adapted


def materialize_exact(path: Path, payload: bytes, label: str) -> str:
    expected_hash = hashlib.sha256(payload).hexdigest()
    if path.exists():
        require_hash(path, expected_hash, label)
    else:
        atomic_write_bytes(path, payload)
        require_hash(path, expected_hash, label)
    return expected_hash


def validate_direct_record_metadata(
    records: list[dict[str, Any]],
    condition: dict[str, Any],
    expected_ids: list[Any],
    subset_by_id: dict[Any, dict[str, Any]] | None,
) -> None:
    label = condition["key"]
    if len(records) != condition["records"]:
        raise ValidationError(f"{label} record count mismatch")
    if unique_ids(records, label) != expected_ids:
        raise ValidationError(f"{label} prompt IDs/order mismatch")
    for record in records:
        prompt_id = record["prompt_id"]
        if record.get("seed") != 42:
            raise ValidationError(f"{label} prompt {prompt_id} seed mismatch")
        if record.get("information_tier") != condition["tier"]:
            raise ValidationError(f"{label} prompt {prompt_id} information tier mismatch")
        if not isinstance(record.get("generated_code"), str):
            raise ValidationError(f"{label} prompt {prompt_id} generated_code is not text")
        if not isinstance(record.get("language"), str):
            raise ValidationError(f"{label} prompt {prompt_id} language is not text")
        if subset_by_id is not None:
            source = subset_by_id[prompt_id]
            if record["language"] != source["language"]:
                raise ValidationError(f"{label} prompt {prompt_id} language mismatch")
            if record.get("target_cwe") != source["cwe_id"]:
                raise ValidationError(f"{label} prompt {prompt_id} target CWE mismatch")
        if condition["method"] == "B1-CWE" and record.get("method") != "B1-CWE":
            raise ValidationError(f"{label} prompt {prompt_id} method mismatch")
        if condition["method"] == "CAA-CWE-StageA":
            if record.get("method") != "CAA-CWE":
                raise ValidationError(f"{label} prompt {prompt_id} method mismatch")
            frozen_condition = record.get("condition")
            if not isinstance(frozen_condition, dict):
                raise ValidationError(f"{label} prompt {prompt_id} condition is missing")
            if frozen_condition.get("stage") != "A":
                raise ValidationError(f"{label} prompt {prompt_id} stage mismatch")
            if frozen_condition.get("layer") != condition["layer"]:
                raise ValidationError(f"{label} prompt {prompt_id} layer mismatch")
            if frozen_condition.get("multiplier") != condition["multiplier"]:
                raise ValidationError(f"{label} prompt {prompt_id} multiplier mismatch")
        if condition["method"] == "B*-AlwaysOn-L19":
            if record.get("method") != "B*-AlwaysOn-L19" or record.get("layer") != 19:
                raise ValidationError(f"{label} prompt {prompt_id} method/layer mismatch")


def validate_run_manifest(
    run: dict[str, Any], entry: dict[str, Any], condition: dict[str, Any]
) -> None:
    if run.get("status") != "COMPLETE":
        raise ValidationError(f"{condition['key']} run manifest is not COMPLETE")
    if run.get("output_sha256") != entry["output_sha256"]:
        raise ValidationError(f"{condition['key']} run/output linkage mismatch")
    detail = run.get("condition", {})
    if detail.get("seed") != 42:
        raise ValidationError(f"{condition['key']} run-manifest seed mismatch")
    if detail.get("input_sha256", entry["prompt_manifest_sha256"]) != entry[
        "prompt_manifest_sha256"
    ]:
        raise ValidationError(f"{condition['key']} run/prompt linkage mismatch")
    if condition["method"] == "RCI-1" and detail.get("variant") != "rci1":
        raise ValidationError("RCI-1 run-manifest variant mismatch")
    if condition["method"] == "RCI-2" and detail.get("variant") != "rci2":
        raise ValidationError("RCI-2 run-manifest variant mismatch")
    if condition["method"] == "CAA-CWE-StageA":
        if detail.get("stage") != "A" or detail.get("layer") != condition["layer"]:
            raise ValidationError(f"{condition['key']} run-manifest stage/layer mismatch")
        if detail.get("multiplier") != 1.0:
            raise ValidationError(f"{condition['key']} run-manifest multiplier mismatch")


def validate_icd(path: Path, expected_ids: list[Any]) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValidationError(f"Malformed ICD output: {path}")
    if len(value) != len(expected_ids) or unique_ids(value, str(path)) != expected_ids:
        raise ValidationError(f"ICD population/order mismatch: {path}")
    required = {
        "prompt_id",
        "cwe_id",
        "language",
        "findings",
        "vulnerable_cwes",
        "is_vulnerable",
        "skipped",
    }
    for record in value:
        if not required.issubset(record):
            raise ValidationError(f"ICD record missing fields in {path}")
        if not isinstance(record["findings"], list) or not isinstance(record["skipped"], bool):
            raise ValidationError(f"Malformed ICD fields in {path}")
    skipped = sum(bool(record["skipped"]) for record in value)
    return {"record_count": len(value), "eligible_count": len(value) - skipped, "skipped_count": skipped}


def validate_transfer_shape(transfer: dict[str, Any]) -> None:
    entries = transfer.get("entries")
    if not isinstance(entries, list) or len(entries) != 7:
        raise ValidationError("Transfer manifest must have exactly seven entries")
    for index, (entry, expected) in enumerate(zip(entries, CONDITIONS), 1):
        checks = {
            "method": expected["method"],
            "layer": expected["layer"],
            "multiplier": expected["multiplier"],
            "information_tier": expected["tier"],
            "record_count": expected["records"],
            "seed": 42,
            "scan_status": "NOT_STARTED",
            "frozen_scanner_source_sha256": EXPECTED_SCANNER_SHA256,
        }
        for field, wanted in checks.items():
            if entry.get(field) != wanted:
                raise ValidationError(
                    f"Transfer entry {index} {field} mismatch: {entry.get(field)!r} != {wanted!r}"
                )


def preflight(workdir: Path, transfer_path: Path, require_semgrep: bool) -> dict[str, Any]:
    require_hash(transfer_path, EXPECTED_TRANSFER_SHA256, "transfer manifest")
    transfer = read_json(transfer_path)
    validate_transfer_shape(transfer)

    summary_path = resolve_path(workdir, "phase14_development_generation_summary.json")
    require_hash(
        summary_path, EXPECTED_GENERATION_SUMMARY_SHA256, "development generation summary"
    )
    summary = read_json(summary_path)
    if summary.get("status") != "DEVELOPMENT_GENERATION_COMPLETE_AWAITING_SCANS_AND_SELECTION":
        raise ValidationError("Development generation summary has the wrong status")
    summary_conditions = summary.get("conditions")
    if not isinstance(summary_conditions, list) or len(summary_conditions) != 7:
        raise ValidationError("Development generation summary must have seven conditions")
    if [item.get("record_count") for item in summary_conditions] != [120, 120, 120, 120, 120, 120, 20]:
        raise ValidationError("Development generation summary record counts mismatch")
    if summary_conditions[1].get("checkpoint_count") != 360:
        raise ValidationError("RCI-1 summary must establish 360 stage rows")
    if summary_conditions[2].get("checkpoint_count") != 600:
        raise ValidationError("RCI-2 summary must establish 600 stage rows")

    scanner_path = resolve_path(workdir, transfer["entries"][0]["frozen_scanner_source_path"])
    require_hash(scanner_path, EXPECTED_SCANNER_SHA256, "frozen scanner")
    load_scanner(scanner_path)

    subset_path = resolve_path(workdir, transfer["entries"][0]["prompt_manifest_path"])
    require_hash(subset_path, EXPECTED_SUBSET_SHA256, "baseline selection subset")
    subset = read_json(subset_path)
    if subset.get("split") != "DEVELOPMENT" or subset.get("validation_status") != "PASS":
        raise ValidationError("Frozen baseline subset is not validated DEVELOPMENT data")
    if subset.get("validation_checks", {}).get("no_heldout_prompt_ids") is not True:
        raise ValidationError("Frozen baseline subset lacks the no-heldout validation")
    subset_records = subset.get("records")
    if not isinstance(subset_records, list) or len(subset_records) != 120:
        raise ValidationError("Frozen baseline subset must contain 120 records")
    subset_ids = unique_ids(subset_records, "baseline subset")
    if subset_ids != subset.get("prompt_ids_source_order"):
        raise ValidationError("Frozen subset record order differs from prompt_ids_source_order")
    subset_by_id = {record["prompt_id"]: record for record in subset_records}

    artifacts = []
    for index, (entry, condition) in enumerate(zip(transfer["entries"], CONDITIONS), 1):
        output_path = resolve_path(workdir, entry["output_path"])
        run_path = resolve_path(workdir, entry["run_manifest_path"])
        prompt_path = resolve_path(workdir, entry["prompt_manifest_path"])
        require_hash(output_path, entry["output_sha256"], f"{condition['key']} generation")
        require_hash(run_path, entry["run_manifest_sha256"], f"{condition['key']} run manifest")
        require_hash(prompt_path, entry["prompt_manifest_sha256"], f"{condition['key']} prompt manifest")
        run = read_json(run_path)
        validate_run_manifest(run, entry, condition)
        raw = read_json(output_path)

        adapter_path = None
        adapter_hash = None
        adapter_ids = None
        scanner_input_path = output_path
        if condition["method"].startswith("RCI-"):
            finals = rci_final_records(raw, condition["key"])
            if len(finals) != 120 or unique_ids(finals, condition["key"]) != subset_ids:
                raise ValidationError(f"{condition['key']} final-output population/order mismatch")
            for record in finals:
                if record.get("variant") != condition["variant"]:
                    raise ValidationError(f"{condition['key']} variant mismatch")
                if record.get("seed") != 42 or record.get("information_tier") != "METADATA_FREE":
                    raise ValidationError(f"{condition['key']} frozen metadata mismatch")
            adapted_once = make_rci_adapter(finals, subset_records, condition["variant"])
            adapted_twice = make_rci_adapter(finals, subset_records, condition["variant"])
            bytes_once = json_bytes(adapted_once)
            bytes_twice = json_bytes(adapted_twice)
            if bytes_once != bytes_twice:
                raise ValidationError(f"{condition['key']} adapter is not byte deterministic")
            for source, adapted in zip(finals, adapted_once):
                if source["generated_code"] != adapted["generated_code"]:
                    raise ValidationError(f"{condition['key']} generated text changed in adapter")
            adapter_path = (workdir / condition["adapter_name"]).resolve()
            adapter_hash = materialize_exact(
                adapter_path, bytes_once, f"{condition['key']} scanner adapter"
            )
            adapter_ids = subset_ids
            scanner_input_path = adapter_path
            records = adapted_once
        else:
            records = direct_records(raw, condition["key"])
            if condition["method"] == "B*-AlwaysOn-L19":
                always_manifest = read_json(prompt_path)
                if sha256_path(prompt_path) != EXPECTED_ALWAYSON_MANIFEST_SHA256:
                    raise ValidationError("AlwaysOn prompt-manifest hash mismatch")
                verification = always_manifest.get("verification", {})
                if verification.get("held_out_overlap_count") != 0:
                    raise ValidationError("AlwaysOn prompt manifest has held-out overlap")
                if always_manifest.get("held_out_prompt_or_result_file_opened") is not False:
                    raise ValidationError("AlwaysOn prompt manifest reports held-out access")
                expected_ids = always_manifest.get("prompt_ids_execution_order")
                validate_direct_record_metadata(records, condition, expected_ids, None)
            else:
                validate_direct_record_metadata(records, condition, subset_ids, subset_by_id)

        expected_ids = unique_ids(records, f"{condition['key']} scanner population")
        if len(expected_ids) != condition["records"]:
            raise ValidationError(f"{condition['key']} scanner population count mismatch")
        artifacts.append(
            {
                "order": index,
                "key": condition["key"],
                "method": condition["method"],
                "layer": condition["layer"],
                "multiplier": condition["multiplier"],
                "seed": 42,
                "information_tier": condition["tier"],
                "generation_path": str(output_path),
                "generation_sha256": entry["output_sha256"],
                "run_manifest_path": str(run_path),
                "run_manifest_sha256": entry["run_manifest_sha256"],
                "prompt_manifest_path": str(prompt_path),
                "prompt_manifest_sha256": entry["prompt_manifest_sha256"],
                "scanner_input_path": str(scanner_input_path),
                "scanner_adapter_path": str(adapter_path) if adapter_path else None,
                "scanner_adapter_sha256": adapter_hash,
                "scanner_adapter_prompt_ids": adapter_ids,
                "scanner_adapter_generated_text_unchanged": True if adapter_path else None,
                "scanner_adapter_metadata_source": (
                    str(subset_path) if adapter_path else None
                ),
                "expected_prompt_ids": expected_ids,
                "record_count": len(expected_ids),
                "icd_path": str((workdir / condition["icd_name"]).resolve()),
            }
        )

    return {
        "schema_version": "phase14_scan_preflight_v1",
        "phase": 14,
        "status": "PASS",
        "preflight": "PASS",
        "scanner_executed": False,
        "security_metrics_computed": False,
        "held_out_prompt_or_result_file_accessed": False,
        "wrapper_path": str(Path(__file__).resolve()),
        "wrapper_sha256": sha256_path(Path(__file__).resolve()),
        "transfer_manifest_path": str(transfer_path),
        "transfer_manifest_sha256": EXPECTED_TRANSFER_SHA256,
        "generation_summary_path": str(summary_path),
        "generation_summary_sha256": EXPECTED_GENERATION_SUMMARY_SHA256,
        "scanner_path": str(scanner_path),
        "scanner_sha256": EXPECTED_SCANNER_SHA256,
        "baseline_subset_path": str(subset_path),
        "baseline_subset_sha256": EXPECTED_SUBSET_SHA256,
        "condition_order": [condition["key"] for condition in CONDITIONS],
        "condition_count": len(artifacts),
        "total_record_count": sum(item["record_count"] for item in artifacts),
        "artifacts": artifacts,
        "environment": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "semgrep_version": semgrep_version(require_semgrep),
            "historical_phase9_semgrep_registry_reproducibility": "NOT_REPRODUCIBLE",
        },
    }


def new_return_manifest(preflight_result: dict[str, Any]) -> dict[str, Any]:
    entries = []
    for artifact in preflight_result["artifacts"]:
        entries.append(
            {
                **artifact,
                "icd_sha256": None,
                "scanner_record_count": None,
                "eligible_count": None,
                "skipped_count": None,
                "scanner_sha256": EXPECTED_SCANNER_SHA256,
                "wrapper_sha256": preflight_result["wrapper_sha256"],
                "semgrep_version": preflight_result["environment"]["semgrep_version"],
                "completion_order": None,
                "scan_status": "NOT_STARTED",
                "resume_status": "NOT_APPLICABLE",
            }
        )
    return {
        "schema_version": "phase14_development_scan_return_manifest_v1",
        "phase": 14,
        "status": "SCANNING_IN_PROGRESS",
        "transfer_manifest_sha256": EXPECTED_TRANSFER_SHA256,
        "scanner_sha256": EXPECTED_SCANNER_SHA256,
        "wrapper_sha256": preflight_result["wrapper_sha256"],
        "environment": preflight_result["environment"],
        "historical_phase9_semgrep_registry_reproducibility": "NOT_REPRODUCIBLE",
        "held_out_prompt_or_result_file_accessed": False,
        "security_metrics_computed": False,
        "entries": entries,
        "complete_count": 0,
        "total_scanner_records": 0,
    }


def validate_complete_entry(
    entry: dict[str, Any], current: dict[str, Any], preflight_result: dict[str, Any]
) -> None:
    for field in (
        "key",
        "generation_sha256",
        "run_manifest_sha256",
        "prompt_manifest_sha256",
        "scanner_adapter_sha256",
        "expected_prompt_ids",
    ):
        if entry.get(field) != current.get(field):
            raise ValidationError(f"Completed {entry.get('key')} linkage mismatch for {field}")
    if entry.get("scanner_sha256") != EXPECTED_SCANNER_SHA256:
        raise ValidationError(f"Completed {entry.get('key')} scanner hash mismatch")
    if entry.get("wrapper_sha256") != preflight_result["wrapper_sha256"]:
        raise ValidationError(f"Completed {entry.get('key')} wrapper hash mismatch")
    icd_path = Path(current["icd_path"])
    if not icd_path.exists():
        raise ValidationError(f"Completed {entry.get('key')} ICD is missing")
    require_hash(icd_path, entry.get("icd_sha256"), f"completed {entry.get('key')} ICD")
    counts = validate_icd(icd_path, current["expected_prompt_ids"])
    if counts != {
        "record_count": entry.get("scanner_record_count"),
        "eligible_count": entry.get("eligible_count"),
        "skipped_count": entry.get("skipped_count"),
    }:
        raise ValidationError(f"Completed {entry.get('key')} ICD counts changed")


def run_scans(
    workdir: Path, preflight_result: dict[str, Any], resume: bool
) -> tuple[Path, dict[str, Any]]:
    return_path = (workdir / "phase14_development_scan_return_manifest.json").resolve()
    resumed_existing_run = return_path.exists()
    if resumed_existing_run:
        manifest = read_json(return_path)
        if not resume:
            raise ValidationError("Return manifest already exists; use --resume")
        for field, expected in (
            ("transfer_manifest_sha256", EXPECTED_TRANSFER_SHA256),
            ("scanner_sha256", EXPECTED_SCANNER_SHA256),
            ("wrapper_sha256", preflight_result["wrapper_sha256"]),
        ):
            if manifest.get(field) != expected:
                raise ValidationError(f"Existing return manifest {field} mismatch")
        if len(manifest.get("entries", [])) != 7:
            raise ValidationError("Existing return manifest does not have seven entries")
    else:
        manifest = new_return_manifest(preflight_result)
        atomic_write_json(return_path, manifest)

    scanner_path = Path(preflight_result["scanner_path"])
    scan_file = load_scanner(scanner_path)

    for index, current in enumerate(preflight_result["artifacts"]):
        entry = manifest["entries"][index]
        if entry.get("scan_status") == "COMPLETE":
            validate_complete_entry(entry, current, preflight_result)
            entry["resume_status"] = "REUSED_VALIDATED_COMPLETE"
            atomic_write_json(return_path, manifest)
            print(f"REUSED VALIDATED COMPLETE: {current['key']}")
            continue
        if entry.get("scan_status") != "NOT_STARTED":
            raise ValidationError(f"Unsafe nonterminal status for {current['key']}")

        # Recheck immutable inputs and the scanner immediately before each condition.
        require_hash(Path(current["generation_path"]), current["generation_sha256"], "generation")
        require_hash(Path(current["run_manifest_path"]), current["run_manifest_sha256"], "run manifest")
        require_hash(scanner_path, EXPECTED_SCANNER_SHA256, "frozen scanner")
        if current["scanner_adapter_path"]:
            require_hash(
                Path(current["scanner_adapter_path"]),
                current["scanner_adapter_sha256"],
                "scanner adapter",
            )

        final_icd = Path(current["icd_path"])
        temporary_icd = final_icd.with_name(f".{final_icd.name}.scan-tmp")
        if final_icd.exists() or temporary_icd.exists():
            raise ValidationError(
                f"Unlinked existing ICD/temporary output for {current['key']}; refusing overwrite"
            )
        print(f"SCANNING {index + 1}/7: {current['key']}")
        scan_file(Path(current["scanner_input_path"]), temporary_icd)
        if not temporary_icd.exists():
            raise ValidationError(f"Frozen scanner did not create output for {current['key']}")
        counts = validate_icd(temporary_icd, current["expected_prompt_ids"])
        os.replace(temporary_icd, final_icd)
        require_hash(Path(current["generation_path"]), current["generation_sha256"], "generation")
        require_hash(scanner_path, EXPECTED_SCANNER_SHA256, "frozen scanner")

        entry.update(
            {
                "icd_sha256": sha256_path(final_icd),
                "scanner_record_count": counts["record_count"],
                "eligible_count": counts["eligible_count"],
                "skipped_count": counts["skipped_count"],
                "completion_order": 1
                + sum(item.get("scan_status") == "COMPLETE" for item in manifest["entries"]),
                "scan_status": "COMPLETE",
                "resume_status": (
                    "NEWLY_SCANNED_AFTER_RESUME" if resumed_existing_run else "NEWLY_SCANNED"
                ),
            }
        )
        manifest["complete_count"] = sum(
            item.get("scan_status") == "COMPLETE" for item in manifest["entries"]
        )
        manifest["total_scanner_records"] = sum(
            item.get("scanner_record_count") or 0 for item in manifest["entries"]
        )
        manifest["status"] = (
            "DEVELOPMENT_SCANS_COMPLETE"
            if manifest["complete_count"] == 7
            else "SCANNING_IN_PROGRESS"
        )
        atomic_write_json(return_path, manifest)
        print(
            f"COMPLETE {current['key']}: records={counts['record_count']} "
            f"eligible={counts['eligible_count']} skipped={counts['skipped_count']}"
        )

    if manifest["complete_count"] != 7 or manifest["total_scanner_records"] != 740:
        raise ValidationError("Final scan manifest is not exactly 7 COMPLETE / 740 records")
    print(f"ALL SEVEN COMPLETE: {return_path}")
    return return_path, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if args.preflight_only and (args.all or args.resume):
        parser.error("--preflight-only cannot be combined with --all or --resume")
    if not args.preflight_only and not args.all:
        parser.error("scanning requires --all; add --resume for safe restart behavior")
    return args


def main() -> int:
    args = parse_args()
    workdir = args.workdir.expanduser().resolve()
    if not workdir.is_dir():
        raise ValidationError(f"Working directory does not exist: {workdir}")
    transfer_path = (
        args.manifest.expanduser().resolve()
        if args.manifest
        else (workdir / "phase14_development_scan_transfer_manifest.json").resolve()
    )
    result = preflight(workdir, transfer_path, require_semgrep=not args.preflight_only)
    preflight_path = (workdir / "phase14_scan_preflight.json").resolve()
    atomic_write_json(preflight_path, result)
    print(f"PREFLIGHT = PASS ({result['total_record_count']} records)")
    print(f"Preflight record: {preflight_path}")
    if args.preflight_only:
        return 0
    return_path, _ = run_scans(workdir, result, resume=args.resume)
    print(f"Return manifest SHA-256: {sha256_path(return_path)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        raise SystemExit(2)
