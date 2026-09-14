#!/usr/bin/env python3
"""Locally verify returned CAA Stage B ICDs against frozen generation inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
OUTPUTS = ROOT / "revision" / "model1" / "phase14" / "outputs"
SUBSET = ROOT / "revision" / "model1" / "phase13" / "outputs" / "baseline_selection_subset.json"
TRANSFER = OUTPUTS / "caa_stage_b_scan_transfer_manifest.json"
RETURN = OUTPUTS / "caa_stage_b_scan_return_manifest.json"
SCANNER = ROOT / "phases" / "phase9" / "colab_scan_phase9.py"

EXPECTED = [
    ("caa_l16_m0p5", 0.5, "caa_stage_b_layer16_mult0p5_seed42"),
    ("caa_l16_m2", 2.0, "caa_stage_b_layer16_mult2p0_seed42"),
    ("caa_l16_m4", 4.0, "caa_stage_b_layer16_mult4p0_seed42"),
]


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    subset = read_json(SUBSET)
    expected_ids = [int(value) for value in subset["prompt_ids_source_order"]]
    subset_by_id = {int(row["prompt_id"]): row for row in subset["records"]}
    transfer = read_json(TRANSFER)
    returned = read_json(RETURN)
    errors: list[str] = []
    transfer_hash = sha256_path(TRANSFER)
    scanner_hash = sha256_path(SCANNER)
    if returned.get("status") != "COMPLETE" or returned.get("complete_count") != 3:
        errors.append("return manifest is not COMPLETE with three entries")
    if returned.get("transfer_manifest_sha256") != transfer_hash:
        errors.append("return/transfer hash linkage mismatch")
    if returned.get("scanner_sha256") != scanner_hash:
        errors.append("return scanner hash mismatch")
    if returned.get("held_out_accessed") is not False:
        errors.append("return manifest does not affirm zero held-out access")
    if len(transfer.get("entries", [])) != 3 or len(returned.get("entries", [])) != 3:
        errors.append("transfer/return entry count mismatch")

    verified = []
    for order, (expected, transfer_entry, return_entry) in enumerate(
        zip(EXPECTED, transfer.get("entries", []), returned.get("entries", [])), 1
    ):
        key, multiplier, stem = expected
        local_errors = []
        if (transfer_entry.get("key"), return_entry.get("key"), return_entry.get("completion_order")) != (key, key, order):
            local_errors.append("condition key/completion order mismatch")
        for field in ("generation_sha256", "run_manifest_sha256", "vector_manifest_sha256"):
            if return_entry.get(field) != transfer_entry.get(field):
                local_errors.append(f"returned {field} differs from transfer")
        if return_entry.get("scanner_sha256") != scanner_hash or return_entry.get("scan_status") != "COMPLETE":
            local_errors.append("scanner hash/status mismatch")
        if return_entry.get("record_count") != 120 or return_entry.get("eligible_count") != 120 or return_entry.get("skipped_count") != 0:
            local_errors.append("returned record/eligible/skipped counts mismatch")

        generation_path = OUTPUTS / f"{stem}.json"
        run_path = OUTPUTS / f"{stem}_run_manifest.json"
        icd_path = OUTPUTS / f"{stem}_icd.json"
        if sha256_path(generation_path) != transfer_entry.get("generation_sha256"):
            local_errors.append("local immutable generation hash mismatch")
        if sha256_path(run_path) != transfer_entry.get("run_manifest_sha256"):
            local_errors.append("local immutable run-manifest hash mismatch")
        if sha256_path(icd_path) != return_entry.get("icd_sha256"):
            local_errors.append("local ICD hash mismatch")

        generation = read_json(generation_path)
        icd = read_json(icd_path)
        generation_ids = [int(row["prompt_id"]) for row in generation]
        icd_ids = [int(row["prompt_id"]) for row in icd]
        if generation_ids != expected_ids or icd_ids != expected_ids or len(set(icd_ids)) != 120:
            local_errors.append("generation/ICD prompt correspondence mismatch")
        required = {"prompt_id", "cwe_id", "language", "findings", "vulnerable_cwes", "is_vulnerable", "skipped"}
        for generated, scanned in zip(generation, icd):
            prompt_id = int(generated["prompt_id"])
            if not required.issubset(scanned):
                local_errors.append(f"ICD schema missing fields at prompt {prompt_id}")
                break
            if scanned["language"] != generated["language"] or scanned["language"] != subset_by_id[prompt_id]["language"]:
                local_errors.append(f"language mismatch at prompt {prompt_id}")
                break
            if scanned["skipped"] is not False:
                local_errors.append(f"unexpected scanner skip at prompt {prompt_id}")
                break
            if not isinstance(scanned["findings"], list) or not isinstance(scanned["vulnerable_cwes"], list) or not isinstance(scanned["is_vulnerable"], bool):
                local_errors.append(f"malformed ICD fields at prompt {prompt_id}")
                break
        errors.extend(f"{key}: {error}" for error in local_errors)
        verified.append({
            "order": order,
            "key": key,
            "layer": 16,
            "multiplier": multiplier,
            "generation_path": generation_path.relative_to(ROOT).as_posix(),
            "generation_sha256": sha256_path(generation_path),
            "run_manifest_path": run_path.relative_to(ROOT).as_posix(),
            "run_manifest_sha256": sha256_path(run_path),
            "icd_path": icd_path.relative_to(ROOT).as_posix(),
            "icd_sha256": sha256_path(icd_path),
            "record_count": len(icd),
            "eligible_count": sum(not row["skipped"] for row in icd),
            "skipped_count": sum(bool(row["skipped"]) for row in icd),
            "prompt_order_exact": icd_ids == expected_ids,
            "status": "PASS" if not local_errors else "FAIL",
            "errors": local_errors,
        })

    result = {
        "schema_version": "phase14_caa_stage_b_scan_local_verification_v1",
        "status": "PASS" if not errors else "FAIL",
        "transfer_manifest_path": TRANSFER.relative_to(ROOT).as_posix(),
        "transfer_manifest_sha256": transfer_hash,
        "return_manifest_path": RETURN.relative_to(ROOT).as_posix(),
        "return_manifest_sha256": sha256_path(RETURN),
        "scanner_path": SCANNER.relative_to(ROOT).as_posix(),
        "scanner_sha256": scanner_hash,
        "python_version": returned.get("python_version"),
        "semgrep_version": returned.get("semgrep_version"),
        "entries": verified,
        "held_out_accessed": False,
        "errors": errors,
    }
    output = OUTPUTS / "caa_stage_b_scan_local_verification.json"
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    if errors:
        raise SystemExit("; ".join(errors))


if __name__ == "__main__":
    main()
