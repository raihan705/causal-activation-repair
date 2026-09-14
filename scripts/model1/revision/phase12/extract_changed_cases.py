#!/usr/bin/env python3
"""Extract scanner-status changes and separate checkpoint-repair provenance."""

from __future__ import annotations

import csv
import difflib
import hashlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = ROOT / "revision/model1/phase12/outputs"

PATHS = {
    "manifest": "revision/model1/phase12/outputs/canonical_bstar_manifest.json",
    "metrics": "revision/model1/phase12/outputs/canonical_seed42_metrics.json",
    "heldout_source": "data/cyberseceval/test_prompts.json",
    "bstar_config": "configs/bstar_config.json",
    "b0_output": "outputs/phase9/baseline_test_outputs.json",
    "b0_icd": "outputs/phase9/baseline_test_icd.json",
    "b2_checkpoint": "outputs/phase9/thea_static_test_outputs_ckpt.json",
    "b2_final": "outputs/phase9/thea_static_test_outputs.json",
    "b2_icd": "outputs/phase9/thea_static_test_icd.json",
    "bstar_checkpoint": "outputs/phase9/bstar_test_outputs_ckpt.json",
    "bstar_final": "outputs/phase9/bstar_test_outputs.json",
    "bstar_icd": "outputs/phase9/bstar_test_icd.json",
}

EXPECTED_HASHES = {
    "heldout_source": "7f015c7e398445953e05f9eddb17a9791bd4efb2103e236d20b7ac7ce28695ab",
    "bstar_config": "f4f5d4d6a697aa14f654763f31e52cd852ce87b9f284276edd4f784604ea36e6",
    "b0_output": "d532f612af2a5d25ece5f6292e46f9d3bb3722518aba4b78050c1f4c9e234488",
    "b0_icd": "250265bf336852de78ff467e456e9f11378357df9462002037a2df0627669560",
    "b2_checkpoint": "4b45e33266fca0e899c463d4f8aa1f4712582d62940f9875529bd3a70893546e",
    "b2_final": "60a2d9d0a5db53037c889f13c191075c2a5cef7cef8821a9fb43d5f2c06605e8",
    "b2_icd": "e0942d49babf1af2ac065272c4a1bc94fa7b5eee499db75d43febf8ec2e4aad2",
    "bstar_checkpoint": "c35851a16f5f58279a4ae16f40dbe0632026c18297c1dc4fe2814805ee1af091",
    "bstar_final": "ddea5dd54197fdedf6db1e0140e68fe2987a552fe87a6b735218a71f4e8c9d39",
    "bstar_icd": "81b05ad6b3c3ea5d87bee0e83534f1c1e84d36318df117c91a44c1bf6a88b158",
}

EXPECTED_PROVENANCE_IDS = {
    78, 1572, 1582, 566, 275, 647, 1598, 907, 764, 1568, 469, 1589, 1573
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="")
    os.replace(temporary, path)


def load_ordered(path: Path, source_ids: list[int], name: str) -> tuple[list[dict], dict[int, dict]]:
    records = read_json(path)
    if not isinstance(records, list):
        fail(f"{name} is not a JSON array")
    ids = [row.get("prompt_id") for row in records]
    if len(ids) != len(set(ids)):
        fail(f"{name} has duplicate prompt IDs")
    if ids != source_ids:
        fail(f"{name} does not match held-out source order")
    return records, {row["prompt_id"]: row for row in records}


def line_diff(b0_text: str, bstar_text: str) -> str:
    lines = difflib.unified_diff(
        b0_text.splitlines(),
        bstar_text.splitlines(),
        fromfile="B0",
        tofile="BSTAR",
        lineterm="",
    )
    return "\n".join(lines)


def fallback_status(record: dict, route_expected: bool) -> str:
    if record.get("note") == "raw_fallback_after_hook_failure":
        return "EXPLICIT_RAW_FALLBACK"
    if record.get("steered") is True:
        return "NO_EXPLICIT_FALLBACK_RECORDED"
    if not route_expected:
        return "UNMAPPED_RAW_BRANCH"
    return "UNRESOLVED_ROUTED_UNSTEERED"


def replacement_identity(record: dict) -> str:
    if record.get("note") == "raw_fallback_after_hook_failure" and record.get("steered") is False:
        return "EXPLICIT_RAW_FALLBACK"
    if record.get("steered") is True and record.get("note") is None:
        return "UNRESOLVED_RETRY_OR_RAW_FALLBACK_IDENTITY"
    return "UNEXPECTED_REPLACEMENT_METADATA"


def main() -> None:
    manifest_path = ROOT / PATHS["manifest"]
    metrics_path = ROOT / PATHS["metrics"]
    if not manifest_path.is_file() or not metrics_path.is_file():
        fail("canonical manifest/metrics missing; run the required predecessor scripts first")
    manifest = read_json(manifest_path)
    metrics = read_json(metrics_path)
    if manifest.get("decision_status") != "CANONICAL_PROVENANCE_RESOLVED":
        fail("canonical provenance is not resolved")
    if manifest.get("canonical_candidate") != "BSTAR":
        fail("canonical candidate is not BSTAR")
    if metrics.get("validation_status") != "PASS" or metrics.get("canonical_candidate") != "BSTAR":
        fail("canonical metric artifact is not a PASS BSTAR record")

    inventory: dict[str, dict[str, Any]] = {
        "manifest": {
            "path": PATHS["manifest"],
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_path(manifest_path),
        },
        "metrics": {
            "path": PATHS["metrics"],
            "bytes": metrics_path.stat().st_size,
            "sha256": sha256_path(metrics_path),
        },
    }
    for name, expected in EXPECTED_HASHES.items():
        path = ROOT / PATHS[name]
        if not path.is_file():
            fail(f"missing required input: {PATHS[name]}")
        actual = sha256_path(path)
        if actual != expected:
            fail(f"hash mismatch for {PATHS[name]}: expected {expected}, found {actual}")
        inventory[name] = {"path": PATHS[name], "bytes": path.stat().st_size, "sha256": actual}

    source = read_json(ROOT / PATHS["heldout_source"])
    source_ids = [row.get("prompt_id") for row in source]
    if len(source_ids) != 575 or len(source_ids) != len(set(source_ids)):
        fail("held-out source is not 575 unique records")
    source_index = {prompt_id: index for index, prompt_id in enumerate(source_ids)}
    source_map = {row["prompt_id"]: row for row in source}

    config = read_json(ROOT / PATHS["bstar_config"])
    feature_map = config.get("feature_map", {})
    alpha = config.get("alpha")
    if alpha != 40.0 or len(feature_map) != 9:
        fail("frozen B* config identity is unexpected")

    _, b0_output_map = load_ordered(ROOT / PATHS["b0_output"], source_ids, "B0 output")
    _, b0_icd_map = load_ordered(ROOT / PATHS["b0_icd"], source_ids, "B0 ICD")
    b2_checkpoint, b2_checkpoint_map = load_ordered(
        ROOT / PATHS["b2_checkpoint"], source_ids, "B2 checkpoint"
    )
    _, b2_final_map = load_ordered(ROOT / PATHS["b2_final"], source_ids, "B2 final")
    _, b2_icd_map = load_ordered(ROOT / PATHS["b2_icd"], source_ids, "B2 ICD")
    bstar_checkpoint, bstar_checkpoint_map = load_ordered(
        ROOT / PATHS["bstar_checkpoint"], source_ids, "BSTAR checkpoint"
    )
    _, bstar_final_map = load_ordered(ROOT / PATHS["bstar_final"], source_ids, "BSTAR final")
    _, bstar_icd_map = load_ordered(ROOT / PATHS["bstar_icd"], source_ids, "BSTAR ICD")

    changed_cases: list[dict[str, Any]] = []
    for prompt in source:
        prompt_id = prompt["prompt_id"]
        b0_scan = b0_icd_map[prompt_id]
        canonical_scan = bstar_icd_map[prompt_id]
        if bool(b0_scan.get("skipped")):
            continue
        if bool(b0_scan.get("is_vulnerable")) == bool(canonical_scan.get("is_vulnerable")):
            continue

        if bool(b0_scan.get("is_vulnerable")):
            direction = "B0_VULNERABLE_TO_CANONICAL_NONVULNERABLE"
        else:
            direction = "B0_SAFE_ELIGIBLE_TO_CANONICAL_VULNERABLE"

        b0_record = b0_output_map[prompt_id]
        bstar_record = bstar_final_map[prompt_id]
        source_cwe = prompt.get("cwe_identifier", "")
        expected_route = feature_map.get(source_cwe)
        route_expected = expected_route is not None
        intervention_active = bool(bstar_record.get("steered"))
        if intervention_active != bool(bstar_record.get("steered")):
            fail(f"intervention_active copy failed for prompt {prompt_id}")

        changed_cases.append({
            "source_index": source_index[prompt_id],
            "prompt_id": prompt_id,
            "language": prompt.get("language", ""),
            "source_cwe": source_cwe,
            "direction": direction,
            "b0_raw_output": b0_record.get("generated_code", ""),
            "canonical_raw_output": bstar_record.get("generated_code", ""),
            "b0_findings": b0_scan.get("findings", []),
            "canonical_findings": canonical_scan.get("findings", []),
            "b0_vulnerable_cwes": sorted(b0_scan.get("vulnerable_cwes", [])),
            "canonical_vulnerable_cwes": sorted(canonical_scan.get("vulnerable_cwes", [])),
            "b0_skipped": bool(b0_scan.get("skipped")),
            "canonical_skipped": bool(canonical_scan.get("skipped")),
            "bstar_steered": bool(bstar_record.get("steered")),
            "intervention_active": intervention_active,
            "route_expected": route_expected,
            "expected_route": expected_route,
            "feature": bstar_record.get("feature"),
            "layer": bstar_record.get("layer"),
            "alpha": bstar_record.get("alpha"),
            "fallback_status": fallback_status(bstar_record, route_expected),
            "b0_output_sha256": sha256_text(b0_record.get("generated_code", "")),
            "canonical_output_sha256": sha256_text(bstar_record.get("generated_code", "")),
            "line_level_diff": line_diff(
                b0_record.get("generated_code", ""), bstar_record.get("generated_code", "")
            ),
        })

    repair_cases = [
        case for case in changed_cases
        if case["direction"] == "B0_VULNERABLE_TO_CANONICAL_NONVULNERABLE"
    ]
    corruption_cases = [
        case for case in changed_cases
        if case["direction"] == "B0_SAFE_ELIGIBLE_TO_CANONICAL_VULNERABLE"
    ]
    active_cases = [case for case in changed_cases if case["intervention_active"]]
    inactive_cases = [case for case in changed_cases if not case["intervention_active"]]

    expected_repair_ids = set(metrics.get("repaired_prompt_ids", []))
    expected_corruption_ids = set(metrics.get("corruption_prompt_ids", []))
    observed_repair_ids = {case["prompt_id"] for case in repair_cases}
    observed_corruption_ids = {case["prompt_id"] for case in corruption_cases}
    checks = [
        (len(changed_cases) == 16, f"changed-case total is {len(changed_cases)}, expected 16"),
        (len(repair_cases) == 14, f"repair-candidate total is {len(repair_cases)}, expected 14"),
        (len(corruption_cases) == 2, f"corruption-candidate total is {len(corruption_cases)}, expected 2"),
        (observed_repair_ids == expected_repair_ids, "changed-case repairs differ from canonical metrics"),
        (observed_corruption_ids == expected_corruption_ids, "changed-case corruptions differ from canonical metrics"),
        (len(active_cases) == 13, f"intervention-active changes are {len(active_cases)}, expected 13"),
        (len(inactive_cases) == 3, f"intervention-inactive changes are {len(inactive_cases)}, expected 3"),
    ]
    failures = [message for condition, message in checks if not condition]
    if failures:
        fail("; ".join(failures))

    b2_empty_ids = {
        row["prompt_id"] for row in b2_checkpoint
        if not bool(row.get("generated_code", "").strip())
    }
    bstar_empty_ids = {
        row["prompt_id"] for row in bstar_checkpoint
        if not bool(row.get("generated_code", "").strip())
    }
    if b2_empty_ids != EXPECTED_PROVENANCE_IDS or bstar_empty_ids != EXPECTED_PROVENANCE_IDS:
        fail("direct-checkpoint empty populations differ from the frozen 13-record set")

    checkpoint_repair_provenance: list[dict[str, Any]] = []
    for prompt_id in source_ids:
        if prompt_id not in EXPECTED_PROVENANCE_IDS:
            continue
        prompt = source_map[prompt_id]
        b2c = b2_checkpoint_map[prompt_id]
        bsc = bstar_checkpoint_map[prompt_id]
        b2f = b2_final_map[prompt_id]
        bsf = bstar_final_map[prompt_id]
        b2scan = b2_icd_map[prompt_id]
        bsscan = bstar_icd_map[prompt_id]
        checkpoint_repair_provenance.append({
            "source_index": source_index[prompt_id],
            "prompt_id": prompt_id,
            "language": prompt.get("language", ""),
            "source_cwe": prompt.get("cwe_identifier", ""),
            "in_changed_security_population": prompt_id in (observed_repair_ids | observed_corruption_ids),
            "b0_scanner_skipped": bool(b0_icd_map[prompt_id].get("skipped")),
            "b0_is_vulnerable": bool(b0_icd_map[prompt_id].get("is_vulnerable")),
            "b2": {
                "direct_checkpoint_status": "DECODED_EMPTY_ROUTED",
                "direct_checkpoint_steered": bool(b2c.get("steered")),
                "direct_checkpoint_output_sha256": sha256_text(b2c.get("generated_code", "")),
                "final_replacement_status": replacement_identity(b2f),
                "final_nonempty": bool(b2f.get("generated_code", "").strip()),
                "final_steered": bool(b2f.get("steered")),
                "explicit_raw_fallback": b2f.get("note") == "raw_fallback_after_hook_failure",
                "final_output_sha256": sha256_text(b2f.get("generated_code", "")),
                "scanner_visible": not bool(b2scan.get("skipped")),
                "is_vulnerable": bool(b2scan.get("is_vulnerable")),
            },
            "bstar": {
                "direct_checkpoint_status": "DECODED_EMPTY_ROUTED",
                "direct_checkpoint_steered": bool(bsc.get("steered")),
                "direct_checkpoint_output_sha256": sha256_text(bsc.get("generated_code", "")),
                "final_replacement_status": replacement_identity(bsf),
                "final_nonempty": bool(bsf.get("generated_code", "").strip()),
                "final_steered": bool(bsf.get("steered")),
                "explicit_raw_fallback": bsf.get("note") == "raw_fallback_after_hook_failure",
                "final_output_sha256": sha256_text(bsf.get("generated_code", "")),
                "scanner_visible": not bool(bsscan.get("skipped")),
                "is_vulnerable": bool(bsscan.get("is_vulnerable")),
            },
            "candidate_final_text_identical": b2f.get("generated_code", "") == bsf.get("generated_code", ""),
            "candidate_vulnerability_status_diff": bool(b2scan.get("is_vulnerable")) != bool(bsscan.get("is_vulnerable")),
        })

    if len(checkpoint_repair_provenance) != 13:
        fail("checkpoint-repair provenance population is not 13")
    if any(row["in_changed_security_population"] for row in checkpoint_repair_provenance):
        fail("checkpoint-repair provenance was incorrectly merged with changed-security population")
    row_78 = next(row for row in checkpoint_repair_provenance if row["prompt_id"] == 78)
    if not row_78["candidate_vulnerability_status_diff"]:
        fail("prompt 78 is not the expected sole B2/BSTAR vulnerability-status discrepancy")
    if sum(row["candidate_vulnerability_status_diff"] for row in checkpoint_repair_provenance) != 1:
        fail("more than prompt 78 differs in B2/BSTAR vulnerability status")

    output = {
        "schema_version": "1.0",
        "phase": 12,
        "canonical_candidate": "BSTAR",
        "source_hashes": inventory,
        "population_definition": {
            "changed_security_population": "All scanner-eligible held-out IDs whose B0 and canonical is_vulnerable status differs.",
            "checkpoint_repair_provenance_population": "The 13 routed IDs decoded empty in both direct B2/BSTAR checkpoints; kept separate from changed-security cases.",
            "human_labels_present": False,
            "human_label_suggestions_present": False,
        },
        "mechanical_summary": {
            "changed_security_total": len(changed_cases),
            "b0_vulnerable_to_canonical_nonvulnerable": len(repair_cases),
            "b0_safe_eligible_to_canonical_vulnerable": len(corruption_cases),
            "intervention_active_true": len(active_cases),
            "intervention_active_false": len(inactive_cases),
            "intervention_active_true_prompt_ids": [case["prompt_id"] for case in active_cases],
            "intervention_active_false_prompt_ids": [case["prompt_id"] for case in inactive_cases],
            "checkpoint_repair_provenance_total": len(checkpoint_repair_provenance),
            "population_overlap_count": 0,
        },
        "changed_cases": changed_cases,
        "checkpoint_repair_provenance": checkpoint_repair_provenance,
        "validation_status": "PASS",
    }

    reverse_fields = [
        "source_index", "prompt_id", "language", "source_cwe", "direction",
        "bstar_steered", "intervention_active", "route_expected", "layer",
        "feature", "alpha", "fallback_status", "b0_output_sha256",
        "canonical_output_sha256", "b0_findings_json", "canonical_findings_json",
    ]
    reverse_rows = []
    for case in corruption_cases:
        reverse_rows.append({
            key: (
                json.dumps(case["b0_findings"], ensure_ascii=False, sort_keys=True)
                if key == "b0_findings_json"
                else json.dumps(case["canonical_findings"], ensure_ascii=False, sort_keys=True)
                if key == "canonical_findings_json"
                else case.get(key)
            )
            for key in reverse_fields
        })

    json_path = OUT_DIR / "changed_case_candidates.json"
    csv_path = OUT_DIR / "reverse_corruption_summary.csv"
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=reverse_fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(reverse_rows)
    atomic_write(json_path, json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    atomic_write(csv_path, csv_buffer.getvalue())

    print("validation_status=PASS")
    print(f"changed_security_total={len(changed_cases)}")
    print(f"repairs={len(repair_cases)}")
    print(f"corruptions={len(corruption_cases)}")
    print(f"intervention_active_true={len(active_cases)}")
    print(f"intervention_active_false={len(inactive_cases)}")
    print(f"checkpoint_repair_provenance_total={len(checkpoint_repair_provenance)}")
    print(f"changed_cases_sha256={sha256_path(json_path)}")
    print(f"reverse_corruption_sha256={sha256_path(csv_path)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1)
