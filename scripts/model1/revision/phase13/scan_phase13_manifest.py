#!/usr/bin/env python3
"""Manifest-driven Phase 13 scan orchestration for a persistent Colab Drive.

This module deliberately contains no scanner rules.  It verifies every frozen
input, imports the frozen Phase 9 scanner, and calls that scanner's unchanged
``scan_file`` function once per alpha in the protocol-defined order.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ALPHAS = (10, 20, 30, 40, 50, 60, 80)
SEED = 42
EXPECTED_RECORD_COUNT = 180
EXPECTED_STEERED_COUNT = 150
EXPECTED_UNSTEERED_COUNT = 30

TRANSFER_SHA256 = "69b1f0069842a4afba0fb4696810aec238a2cb0456429f34f37fda1e88132df5"
SUBSET_SHA256 = "f8781a4532a0d58840a89652342fb88eeda7480cf42924ae1ee451f0b7b898c7"
SCANNER_SHA256 = "60cedd3d6ec2d19e5e7e5ba4d709230efdd5d9478047492fefda4dd88d9afbb7"

RETURN_MANIFEST_NAME = "phase13_scan_return_manifest.json"


class ValidationError(RuntimeError):
    """Raised when a frozen provenance or output-structure check fails."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValidationError(f"Cannot parse JSON file {path}: {exc}") from exc


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def repository_root() -> Path | None:
    resolved = Path(__file__).resolve()
    if len(resolved.parents) >= 5:
        candidate = resolved.parents[4]
        if (candidate / "revision").is_dir():
            return candidate
    return None


def resolve_input(
    recorded_path: str,
    workdir: Path,
    explicit_path: Path | None = None,
) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(explicit_path.expanduser())

    recorded = Path(recorded_path)
    candidates.extend((workdir / recorded.name, workdir / recorded, recorded))
    root = repository_root()
    if root is not None:
        candidates.append(root / recorded)

    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        key = str(resolved)
        if key not in seen and resolved.is_file():
            return resolved
        seen.add(key)
    raise ValidationError(
        f"Required input not found: {recorded_path}; checked "
        + ", ".join(str(path) for path in candidates)
    )


def check_hash(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    require(actual == expected, f"{label} hash mismatch: expected {expected}, got {actual} ({path})")
    return actual


def semgrep_version() -> str:
    try:
        result = subprocess.run(
            ["semgrep", "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except Exception as exc:
        raise ValidationError(f"Semgrep executable/version check failed: {exc}") from exc
    require(result.returncode == 0, f"Semgrep version check failed: {result.stderr.strip()}")
    version = (result.stdout or result.stderr).strip()
    require(bool(version), "Semgrep version check returned an empty version string")
    return version


def environment_record(scanner: Path, wrapper: Path, transfer: Path, subset: Path) -> dict[str, Any]:
    return {
        "captured_at_utc": utc_now(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "semgrep_version": semgrep_version(),
        "scanner_source_path": str(scanner),
        "scanner_source_sha256": sha256_file(scanner),
        "wrapper_path": str(wrapper),
        "wrapper_sha256": sha256_file(wrapper),
        "transfer_manifest_path": str(transfer),
        "transfer_manifest_sha256": sha256_file(transfer),
        "strength_subset_path": str(subset),
        "strength_subset_sha256": sha256_file(subset),
    }


def validate_subset(subset_path: Path) -> tuple[dict[str, Any], list[Any]]:
    check_hash(subset_path, SUBSET_SHA256, "Frozen strength-subset manifest")
    subset = load_json(subset_path)
    require(isinstance(subset, dict), "Strength-subset manifest must be a JSON object")
    records = subset.get("records")
    require(isinstance(records, list), "Strength-subset manifest records must be a list")
    require(len(records) == EXPECTED_RECORD_COUNT, "Strength-subset manifest must contain 180 records")
    ids = [record.get("prompt_id") for record in records]
    require(all(prompt_id is not None for prompt_id in ids), "Strength subset contains a missing prompt_id")
    require(len(set(ids)) == EXPECTED_RECORD_COUNT, "Strength subset prompt IDs are not unique")
    require(subset.get("split") == "DEVELOPMENT", "Strength subset is not marked DEVELOPMENT")
    require(subset.get("validation_status") == "PASS", "Strength-subset validation_status is not PASS")
    checks = subset.get("validation_checks", {})
    require(checks.get("no_heldout_prompt_ids") is True, "Frozen subset does not prove absence of held-out IDs")
    require(
        checks.get("source_metadata_and_b0_prompt_rendering_preserved") is True,
        "Frozen subset does not prove source metadata/prompt preservation",
    )
    counts = subset.get("counts", {})
    require(counts.get("strength_subset") == EXPECTED_RECORD_COUNT, "Subset count metadata is not 180")
    return subset, ids


def validate_generation(
    generation_path: Path,
    entry: dict[str, Any],
    expected_ids: list[Any],
) -> dict[str, Any]:
    alpha = entry["alpha"]
    expected_hash = entry["final_generation_sha256"]
    actual_hash = check_hash(generation_path, expected_hash, f"Alpha-{alpha} generation")
    records = load_json(generation_path)
    require(isinstance(records, list), f"Alpha {alpha} generation must be a JSON list")
    require(len(records) == EXPECTED_RECORD_COUNT, f"Alpha {alpha} generation count is not 180")
    ids = [record.get("prompt_id") for record in records]
    require(ids == expected_ids, f"Alpha {alpha} prompt IDs/order do not match the frozen subset")
    require(len(set(ids)) == EXPECTED_RECORD_COUNT, f"Alpha {alpha} prompt IDs are not unique")
    require(all(record.get("run_seed") == SEED for record in records), f"Alpha {alpha} run_seed is not uniformly 42")
    require(
        all(record.get("condition_alpha") == alpha for record in records),
        f"Alpha {alpha} condition_alpha metadata mismatch",
    )
    steered = sum(record.get("steered") is True for record in records)
    unsteered = sum(record.get("steered") is False for record in records)
    require(steered == EXPECTED_STEERED_COUNT, f"Alpha {alpha} steered count is not 150")
    require(unsteered == EXPECTED_UNSTEERED_COUNT, f"Alpha {alpha} unsteered count is not 30")
    generation_status_counts = {
        status: sum(record.get("generation_status") == status for record in records)
        for status in ("SUCCESS", "WHITESPACE_ONLY")
    }
    require(
        sum(generation_status_counts.values()) == EXPECTED_RECORD_COUNT,
        f"Alpha {alpha} contains an unexpected generation_status",
    )
    require(
        all(record.get("empty_status") in {"NONEMPTY", "WHITESPACE_ONLY"} for record in records),
        f"Alpha {alpha} contains an unexpected empty_status",
    )
    require(
        all(record.get("fallback_status") == "NONE" for record in records),
        f"Alpha {alpha} contains fallback generation",
    )
    require(all(record.get("error") is None for record in records), f"Alpha {alpha} contains a generation error")
    return {
        "path": str(generation_path),
        "sha256": actual_hash,
        "record_count": len(records),
        "steered_count": steered,
        "unsteered_count": unsteered,
        "generation_status_counts": generation_status_counts,
    }


def validate_transfer(
    transfer_path: Path,
    workdir: Path,
    subset_override: Path | None,
    scanner_override: Path | None,
) -> tuple[dict[str, Any], Path, Path, list[Any], dict[int, dict[str, Any]]]:
    check_hash(transfer_path, TRANSFER_SHA256, "Frozen transfer manifest")
    transfer = load_json(transfer_path)
    require(isinstance(transfer, dict), "Transfer manifest must be a JSON object")
    entries = transfer.get("entries")
    require(isinstance(entries, list), "Transfer manifest entries must be a list")
    require(transfer.get("entry_count") == len(ALPHAS), "Transfer manifest entry_count is not seven")
    require([entry.get("alpha") for entry in entries] == list(ALPHAS), "Transfer alpha sequence is not frozen order")

    first = entries[0]
    subset_path = resolve_input(first["active_strength_subset_path"], workdir, subset_override)
    scanner_path = resolve_input(first["scanner_source_path"], workdir, scanner_override)
    check_hash(scanner_path, SCANNER_SHA256, "Frozen scanner source")
    _, expected_ids = validate_subset(subset_path)

    resolved: dict[int, dict[str, Any]] = {}
    for entry in entries:
        alpha = entry.get("alpha")
        require(entry.get("seed") == SEED, f"Transfer entry alpha {alpha} seed is not 42")
        require(entry.get("record_count") == EXPECTED_RECORD_COUNT, f"Transfer entry alpha {alpha} count is not 180")
        require(entry.get("steered_count") == EXPECTED_STEERED_COUNT, f"Transfer alpha {alpha} steered count mismatch")
        require(entry.get("unsteered_count") == EXPECTED_UNSTEERED_COUNT, f"Transfer alpha {alpha} unsteered count mismatch")
        require(entry.get("scan_status") == "NOT_STARTED", f"Frozen transfer alpha {alpha} scan_status changed")
        require(entry.get("active_strength_subset_sha256") == SUBSET_SHA256, f"Transfer alpha {alpha} subset hash mismatch")
        require(entry.get("scanner_source_sha256") == SCANNER_SHA256, f"Transfer alpha {alpha} scanner hash mismatch")
        generation_path = resolve_input(entry["final_generation_path"], workdir)
        validation = validate_generation(generation_path, entry, expected_ids)
        resolved[alpha] = {
            "transfer_entry": entry,
            "generation_path": generation_path,
            "generation_validation": validation,
        }
    return transfer, subset_path, scanner_path, expected_ids, resolved


def validate_icd(path: Path, expected_ids: list[Any], alpha: int) -> dict[str, Any]:
    require(path.is_file(), f"Alpha {alpha} ICD output was not created: {path}")
    records = load_json(path)
    require(isinstance(records, list), f"Alpha {alpha} ICD must be a JSON list")
    require(len(records) == EXPECTED_RECORD_COUNT, f"Alpha {alpha} ICD count is not 180")
    ids = [record.get("prompt_id") for record in records]
    require(ids == expected_ids, f"Alpha {alpha} ICD prompt IDs/order mismatch")
    require(len(set(ids)) == EXPECTED_RECORD_COUNT, f"Alpha {alpha} ICD prompt IDs are not unique")
    required = {"prompt_id", "cwe_id", "language", "findings", "vulnerable_cwes", "is_vulnerable", "skipped"}
    for index, record in enumerate(records):
        require(isinstance(record, dict), f"Alpha {alpha} ICD record {index} is not an object")
        require(required.issubset(record), f"Alpha {alpha} ICD record {index} lacks frozen scanner fields")
        require(isinstance(record["findings"], list), f"Alpha {alpha} ICD record {index} findings is not a list")
        require(isinstance(record["vulnerable_cwes"], list), f"Alpha {alpha} ICD record {index} vulnerable_cwes is not a list")
        require(isinstance(record["is_vulnerable"], bool), f"Alpha {alpha} ICD record {index} is_vulnerable is not bool")
        require(isinstance(record["skipped"], bool), f"Alpha {alpha} ICD record {index} skipped is not bool")
    skipped = sum(record["skipped"] is True for record in records)
    eligible = len(records) - skipped
    return {
        "scanner_record_count": len(records),
        "scanner_eligible_count": eligible,
        "scanner_skipped_count": skipped,
        "scanner_output_sha256": sha256_file(path),
        "scanner_output_path": str(path),
    }


def import_frozen_scanner(scanner_path: Path):
    spec = importlib.util.spec_from_file_location("phase13_frozen_phase9_scanner", scanner_path)
    require(spec is not None and spec.loader is not None, f"Cannot construct import spec for {scanner_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(callable(getattr(module, "scan_file", None)), "Frozen scanner does not expose callable scan_file")
    return module


def new_return_manifest(
    transfer_path: Path,
    subset_path: Path,
    scanner_path: Path,
    wrapper_path: Path,
    environment: dict[str, Any],
    resolved: dict[int, dict[str, Any]],
    workdir: Path,
) -> dict[str, Any]:
    entries = []
    for alpha in ALPHAS:
        source = resolved[alpha]["generation_validation"]
        entries.append(
            {
                "alpha": alpha,
                "seed": SEED,
                "scan_status": "NOT_STARTED",
                "completion_order": None,
                "resume_status": "NOT_APPLICABLE_FRESH",
                "generation_path": source["path"],
                "generation_sha256": source["sha256"],
                "input_record_count": source["record_count"],
                "pre_scan_generation_sha256": None,
                "post_scan_generation_sha256": None,
                "generation_hash_unchanged": None,
                "scanner_output_path": str(workdir / f"bstar_strength_alpha{alpha}_seed42_icd.json"),
                "scanner_output_sha256": None,
                "scanner_record_count": None,
                "scanner_eligible_count": None,
                "scanner_skipped_count": None,
                "frozen_subset_sha256": SUBSET_SHA256,
                "frozen_scanner_sha256": SCANNER_SHA256,
                "wrapper_sha256": sha256_file(wrapper_path),
                "platform": environment["platform"],
                "python_version": environment["python_version"],
                "semgrep_version": environment["semgrep_version"],
                "started_at_utc": None,
                "completed_at_utc": None,
                "error": None,
            }
        )
    return {
        "schema_version": "1.0",
        "phase": 13,
        "status": "IN_PROGRESS",
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "scan_order": list(ALPHAS),
        "seed": SEED,
        "transfer_manifest_path": str(transfer_path),
        "transfer_manifest_sha256": TRANSFER_SHA256,
        "strength_subset_path": str(subset_path),
        "strength_subset_sha256": SUBSET_SHA256,
        "scanner_source_path": str(scanner_path),
        "scanner_source_sha256": SCANNER_SHA256,
        "wrapper_path": str(wrapper_path),
        "wrapper_sha256": sha256_file(wrapper_path),
        "historical_semgrep_version": "NOT_REPRODUCIBLE",
        "historical_semgrep_registry_state": "NOT_REPRODUCIBLE",
        "execution_environment": environment,
        "last_resume_environment": None,
        "entries": entries,
    }


def validate_existing_return_manifest(
    manifest: dict[str, Any],
    wrapper_path: Path,
    environment: dict[str, Any],
    expected_ids: list[Any],
    resolved: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    require(manifest.get("phase") == 13, "Existing return manifest phase is not 13")
    require(manifest.get("scan_order") == list(ALPHAS), "Existing return manifest scan order mismatch")
    require(manifest.get("seed") == SEED, "Existing return manifest seed mismatch")
    require(manifest.get("transfer_manifest_sha256") == TRANSFER_SHA256, "Existing return transfer hash mismatch")
    require(manifest.get("strength_subset_sha256") == SUBSET_SHA256, "Existing return subset hash mismatch")
    require(manifest.get("scanner_source_sha256") == SCANNER_SHA256, "Existing return scanner hash mismatch")
    require(manifest.get("wrapper_sha256") == sha256_file(wrapper_path), "Existing return wrapper hash mismatch")
    entries = manifest.get("entries")
    require(isinstance(entries, list) and len(entries) == len(ALPHAS), "Existing return entries malformed")
    require([entry.get("alpha") for entry in entries] == list(ALPHAS), "Existing return entry order mismatch")

    completion_orders: list[int] = []
    complete_flags: list[bool] = []
    for entry in entries:
        alpha = entry["alpha"]
        source = resolved[alpha]["generation_validation"]
        require(entry.get("seed") == SEED, f"Existing return alpha {alpha} seed mismatch")
        require(entry.get("generation_sha256") == source["sha256"], f"Existing return alpha {alpha} source hash mismatch")
        require(entry.get("input_record_count") == EXPECTED_RECORD_COUNT, f"Existing return alpha {alpha} input count mismatch")
        require(entry.get("frozen_subset_sha256") == SUBSET_SHA256, f"Existing return alpha {alpha} subset hash mismatch")
        require(entry.get("frozen_scanner_sha256") == SCANNER_SHA256, f"Existing return alpha {alpha} scanner hash mismatch")
        require(entry.get("wrapper_sha256") == sha256_file(wrapper_path), f"Existing return alpha {alpha} wrapper hash mismatch")
        output_path = Path(entry.get("scanner_output_path", ""))
        if entry.get("scan_status") == "COMPLETE":
            complete_flags.append(True)
            validated = validate_icd(output_path, expected_ids, alpha)
            require(entry.get("scanner_output_sha256") == validated["scanner_output_sha256"], f"Existing COMPLETE alpha {alpha} output hash mismatch")
            require(entry.get("scanner_record_count") == EXPECTED_RECORD_COUNT, f"Existing COMPLETE alpha {alpha} output count mismatch")
            require(entry.get("scanner_eligible_count") == validated["scanner_eligible_count"], f"Existing COMPLETE alpha {alpha} eligible count mismatch")
            require(entry.get("scanner_skipped_count") == validated["scanner_skipped_count"], f"Existing COMPLETE alpha {alpha} skipped count mismatch")
            require(entry.get("generation_hash_unchanged") is True, f"Existing COMPLETE alpha {alpha} lacks immutable generation proof")
            require(entry.get("post_scan_generation_sha256") == source["sha256"], f"Existing COMPLETE alpha {alpha} post-scan hash mismatch")
            order = entry.get("completion_order")
            require(isinstance(order, int), f"Existing COMPLETE alpha {alpha} completion order missing")
            completion_orders.append(order)
            entry["resume_status"] = "REUSED_VALIDATED_COMPLETE"
        else:
            complete_flags.append(False)
            require(
                entry.get("scan_status") in {"NOT_STARTED", "IN_PROGRESS", "FAILED"},
                f"Existing alpha {alpha} has unsupported incomplete status {entry.get('scan_status')}",
            )
    complete_count = sum(complete_flags)
    require(
        complete_flags == ([True] * complete_count + [False] * (len(ALPHAS) - complete_count)),
        "Existing COMPLETE scans are not a prefix of the frozen alpha order",
    )
    require(
        completion_orders == list(range(1, complete_count + 1)),
        "Existing completion_order values do not match the frozen alpha sequence",
    )
    manifest["last_resume_environment"] = environment
    manifest["updated_at_utc"] = utc_now()
    return manifest


def run(args: argparse.Namespace) -> int:
    require(args.all, "Phase 13 requires --all; selective alpha scanning is prohibited")
    wrapper_path = Path(__file__).resolve()
    workdir = args.workdir.expanduser().resolve()
    require(workdir.is_dir(), f"Persistent workdir does not exist: {workdir}")
    if not args.preflight_only:
        drive_root = Path("/content/drive").resolve()
        require(
            workdir == drive_root or drive_root in workdir.parents,
            f"Actual scans require a persistent mounted Colab Drive workdir under {drive_root}",
        )

    transfer_path = resolve_input(str(args.manifest), workdir, args.manifest)
    transfer, subset_path, scanner_path, expected_ids, resolved = validate_transfer(
        transfer_path, workdir, args.subset, args.scanner_source
    )
    del transfer

    preflight = {
        "status": "PASS",
        "alpha_order": list(ALPHAS),
        "seed": SEED,
        "record_count_per_alpha": EXPECTED_RECORD_COUNT,
        "steered_count_per_alpha": EXPECTED_STEERED_COUNT,
        "unsteered_count_per_alpha": EXPECTED_UNSTEERED_COUNT,
        "no_heldout_prompt_ids": True,
        "transfer_manifest_sha256": sha256_file(transfer_path),
        "strength_subset_sha256": sha256_file(subset_path),
        "scanner_source_sha256": sha256_file(scanner_path),
        "wrapper_sha256": sha256_file(wrapper_path),
        "generation_sha256": {str(alpha): resolved[alpha]["generation_validation"]["sha256"] for alpha in ALPHAS},
    }
    if args.preflight_only:
        print(json.dumps({"phase13_preflight": preflight}, indent=2))
        return 0

    environment = environment_record(scanner_path, wrapper_path, transfer_path, subset_path)
    scanner = import_frozen_scanner(scanner_path)
    return_path = (args.return_manifest or (workdir / RETURN_MANIFEST_NAME)).expanduser().resolve()
    require(
        return_path.parent == workdir,
        "Return manifest must be written directly in the persistent Phase 13 workdir",
    )

    if return_path.exists():
        require(args.resume, f"Return manifest already exists; use --resume after inspecting it: {return_path}")
        manifest = load_json(return_path)
        require(isinstance(manifest, dict), "Existing return manifest must be a JSON object")
        manifest = validate_existing_return_manifest(manifest, wrapper_path, environment, expected_ids, resolved)
    else:
        manifest = new_return_manifest(
            transfer_path, subset_path, scanner_path, wrapper_path, environment, resolved, workdir
        )
    atomic_write_json(return_path, manifest)

    entry_by_alpha = {entry["alpha"]: entry for entry in manifest["entries"]}
    next_completion_order = 1 + max(
        [entry.get("completion_order", 0) or 0 for entry in manifest["entries"]], default=0
    )

    for alpha in ALPHAS:
        entry = entry_by_alpha[alpha]
        generation_path = resolved[alpha]["generation_path"]
        expected_generation_hash = resolved[alpha]["generation_validation"]["sha256"]
        output_path = Path(entry["scanner_output_path"])

        # Re-verify every immutable protocol file immediately before each alpha.
        check_hash(transfer_path, TRANSFER_SHA256, "Frozen transfer manifest")
        check_hash(subset_path, SUBSET_SHA256, "Frozen strength-subset manifest")
        check_hash(scanner_path, SCANNER_SHA256, "Frozen scanner source")
        check_hash(wrapper_path, manifest["wrapper_sha256"], "Phase 13 wrapper")
        for frozen_alpha in ALPHAS:
            frozen_generation = resolved[frozen_alpha]["generation_path"]
            frozen_hash = resolved[frozen_alpha]["generation_validation"]["sha256"]
            check_hash(frozen_generation, frozen_hash, f"Alpha-{frozen_alpha} frozen generation")

        if entry["scan_status"] == "COMPLETE":
            continue

        if output_path.exists():
            require(
                entry["scan_status"] in {"IN_PROGRESS", "FAILED"},
                f"Unexpected pre-existing ICD for NOT_STARTED alpha {alpha}: {output_path}",
            )
            validated = validate_icd(output_path, expected_ids, alpha)
            post_hash = check_hash(generation_path, expected_generation_hash, f"Alpha-{alpha} generation after recovery")
            entry.update(validated)
            entry.update(
                {
                    "scan_status": "COMPLETE",
                    "completion_order": next_completion_order,
                    "resume_status": "RECOVERED_VALIDATED_EXISTING_OUTPUT",
                    "post_scan_generation_sha256": post_hash,
                    "generation_hash_unchanged": post_hash == expected_generation_hash,
                    "completed_at_utc": utc_now(),
                    "error": None,
                }
            )
            next_completion_order += 1
            manifest["updated_at_utc"] = utc_now()
            atomic_write_json(return_path, manifest)
            continue

        prior_status = entry["scan_status"]
        entry.update(
            {
                "scan_status": "IN_PROGRESS",
                "resume_status": (
                    "RETRY_AFTER_INTERRUPTION_NO_OUTPUT"
                    if prior_status == "IN_PROGRESS"
                    else "RETRY_AFTER_RECORDED_FAILURE_NO_OUTPUT"
                    if prior_status == "FAILED"
                    else "EXECUTED_FRESH"
                ),
                "pre_scan_generation_sha256": sha256_file(generation_path),
                "platform": environment["platform"],
                "python_version": environment["python_version"],
                "semgrep_version": environment["semgrep_version"],
                "started_at_utc": utc_now(),
                "completed_at_utc": None,
                "error": None,
            }
        )
        manifest["updated_at_utc"] = utc_now()
        atomic_write_json(return_path, manifest)

        try:
            scanner.scan_file(generation_path, output_path)
            validated = validate_icd(output_path, expected_ids, alpha)
            post_hash = check_hash(generation_path, expected_generation_hash, f"Alpha-{alpha} post-scan generation")
            entry.update(validated)
            entry.update(
                {
                    "scan_status": "COMPLETE",
                    "completion_order": next_completion_order,
                    "post_scan_generation_sha256": post_hash,
                    "generation_hash_unchanged": post_hash == entry["pre_scan_generation_sha256"],
                    "completed_at_utc": utc_now(),
                    "error": None,
                }
            )
            require(entry["generation_hash_unchanged"] is True, f"Alpha {alpha} generation changed during scanning")
            next_completion_order += 1
        except Exception as exc:
            entry["scan_status"] = "FAILED"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            manifest["status"] = "FAILED"
            manifest["updated_at_utc"] = utc_now()
            atomic_write_json(return_path, manifest)
            raise

        manifest["updated_at_utc"] = utc_now()
        atomic_write_json(return_path, manifest)

    require(all(entry["scan_status"] == "COMPLETE" for entry in manifest["entries"]), "Not all seven scans completed")
    require(
        [entry["completion_order"] for entry in manifest["entries"]] == list(range(1, len(ALPHAS) + 1)),
        "Completion order is not exactly the frozen alpha sequence",
    )
    manifest["status"] = "COMPLETE"
    manifest["completed_at_utc"] = utc_now()
    manifest["updated_at_utc"] = utc_now()
    atomic_write_json(return_path, manifest)

    summary = {
        "phase13_scan": "COMPLETE",
        "return_manifest_path": str(return_path),
        "return_manifest_sha256": sha256_file(return_path),
        "wrapper_sha256": sha256_file(wrapper_path),
        "environment": environment,
        "alphas": [
            {
                "alpha": entry["alpha"],
                "output_sha256": entry["scanner_output_sha256"],
                "record_count": entry["scanner_record_count"],
                "eligible": entry["scanner_eligible_count"],
                "skipped": entry["scanner_skipped_count"],
                "resume_status": entry["resume_status"],
            }
            for entry in manifest["entries"]
        ],
    }
    print(json.dumps(summary, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Frozen Phase 13 scan transfer manifest")
    parser.add_argument("--workdir", type=Path, required=True, help="Persistent mounted Colab Drive directory")
    parser.add_argument("--all", action="store_true", help="Required: scan all seven frozen alphas")
    parser.add_argument("--resume", action="store_true", help="Validate and reuse completed entries safely")
    parser.add_argument("--scanner-source", type=Path, help="Optional explicit location of frozen scanner source")
    parser.add_argument("--subset", type=Path, help="Optional explicit location of frozen strength subset")
    parser.add_argument("--return-manifest", type=Path, help="Optional return-manifest path (default: workdir)")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate every frozen input without importing the scanner or executing Semgrep",
    )
    return parser


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except ValidationError as exc:
        print(f"PHASE13_VALIDATION_ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
