#!/usr/bin/env python3
"""Build the Phase 19 CWE evidence/support and consistency audit."""

from __future__ import annotations

import collections
import json

from phase19_common import (
    FROZEN_BSTAR, OUT, ROOT, TARGET_CWES, atomic_csv, atomic_json,
    hash_record, read_csv, read_json, require,
)


def flag(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def cwe_counts(rows):
    raw = collections.Counter()
    augmented = collections.Counter()
    for row in rows:
        (augmented if flag(row.get("is_augmented", "false")) else raw)[row["cwe_id"]] += 1
    return raw, augmented


def detection_counts(path):
    payload = read_json(path)
    records = payload if isinstance(payload, list) else payload.get("records", payload.get("results", []))
    counts = collections.Counter()
    for row in records:
        observed = set(row.get("vulnerable_cwes", []))
        if not observed:
            observed = {finding.get("cwe_id") for finding in row.get("findings", [])}
        for cwe in observed:
            if cwe:
                counts[cwe] += 1
    return counts


def main() -> None:
    full_path = ROOT / "data/processed/vulnerability_pairs_full.csv"
    train_path = ROOT / "data/processed/train_pairs.csv"
    val_path = ROOT / "data/processed/val_pairs.csv"
    test_path = ROOT / "data/processed/test_pairs.csv"
    dev_scan_path = ROOT / "outputs/phase2/baseline_dev_icd.json"
    heldout_scan_path = ROOT / "outputs/phase9/baseline_test_icd.json"
    audit_path = ROOT / "revision/model1/phase2/phase2_denominator_audit.json"
    causal_path = ROOT / "outputs/phase4/phase4_causal_validation.csv"
    revision_causal_path = ROOT / "revision/model1/phase4/outputs/frozen_route_revalidation_scans.json"
    library_path = ROOT / "outputs/phase4/intervention_library.json"
    percwe_path = ROOT / "revision/model1/phase16/outputs/phase16_per_cwe_descriptive.csv"

    full = read_csv(full_path)
    raw, augmented = cwe_counts(full)
    split_rows = {"train": read_csv(train_path), "validation": read_csv(val_path), "test": read_csv(test_path)}
    split_counts = {name: collections.Counter(row["cwe_id"] for row in rows) for name, rows in split_rows.items()}
    dev_detections = detection_counts(dev_scan_path)
    heldout_detections = detection_counts(heldout_scan_path)

    scanner = read_json(audit_path)["scanner_inventory"]
    supported_by_rule = set(scanner["rule_cwe_map"].values())
    supported_by_regex = {cwe for cwe, patterns in scanner["regex_patterns"].items() if patterns}
    scanner_supported = supported_by_rule | supported_by_regex

    causal = read_csv(causal_path)
    revision_causal = read_json(revision_causal_path)["route_summary"]
    library_payload = read_json(library_path)
    if isinstance(library_payload, list):
        groups = library_payload
    else:
        groups = library_payload.get("groups", library_payload.get("interventions", []))
    library_cwes = {group.get("cwe_id") for group in groups}

    percwe = read_csv(percwe_path)
    bstar42 = {row["cwe_id"]: row for row in percwe if row["method"] == "B*" and row["seed"] == "42"}

    source_records = [hash_record(path) for path in (
        full_path, train_path, val_path, test_path, dev_scan_path, heldout_scan_path,
        audit_path, causal_path, revision_causal_path, library_path, percwe_path,
    )]
    source_paths = json.dumps([row["path"] for row in source_records], separators=(",", ":"))
    source_hashes = json.dumps({row["path"]: row["sha256"] for row in source_records}, sort_keys=True, separators=(",", ":"))

    rows = []
    inconsistencies = []
    for cwe in TARGET_CWES:
        route = FROZEN_BSTAR.get(cwe)
        causal_matches = []
        if route:
            layer, feature = route
            causal_matches = [row for row in causal if row["cwe"] == cwe and int(row["layer"]) == layer and int(row["feature_id"]) == feature]
        causal_count = max((int(float(row["n_prompts"])) for row in causal_matches), default=0)
        revision_count = 0
        if cwe in revision_causal:
            revision_count = int(revision_causal[cwe]["repair_denominator"])

        supported = cwe in scanner_supported
        heldout_denominator = int(bstar42[cwe]["b0_vulnerable_denominator"]) if cwe in bstar42 else 0
        if cwe == "CWE-338":
            evidence = "EXPLORATORY"
            reason = "Executed-study exploratory designation preserved; two raw pairs and no confirmatory held-out effect denominator."
        elif cwe == "CWE-290":
            evidence = "NOT_ESTIMABLE"
            reason = "Scope-boundary held-out CWE; one raw pair and no standalone held-out B0 detection/effect denominator."
        elif not supported:
            evidence = "NO_SCANNER_SUPPORT"
            reason = "Frozen revision scanner has no rule or regex capable of standalone target-CWE detection."
        elif heldout_denominator == 0:
            evidence = "NOT_ESTIMABLE"
            reason = "Scanner support exists but the official held-out B0 effect denominator is zero."
        elif heldout_denominator < 5:
            evidence = "EXPLORATORY"
            reason = "Scanner support exists but the official held-out B0 effect denominator is below five."
        else:
            evidence = "PRIMARY"
            reason = "Primary executed-study CWE with scanner support and a non-tiny held-out B0 denominator."

        if raw[cwe] + augmented[cwe] > 0 and not supported:
            inconsistencies.append({"cwe": cwe, "type": "TRAINING_SUPPORT_WITHOUT_SCANNER_SUPPORT"})
        if supported and 0 < heldout_denominator < 5:
            inconsistencies.append({"cwe": cwe, "type": "SCANNER_SUPPORT_WITH_TINY_HELDOUT_DENOMINATOR", "denominator": heldout_denominator})
        if supported and heldout_denominator == 0:
            inconsistencies.append({"cwe": cwe, "type": "SCANNER_SUPPORT_WITH_ZERO_HELDOUT_EFFECT_DENOMINATOR"})

        rows.append({
            "cwe": cwe,
            "raw_pair_count": raw[cwe],
            "augmented_pair_count": augmented[cwe],
            "train_count": split_counts["train"][cwe],
            "validation_count": split_counts["validation"][cwe],
            "test_count": split_counts["test"][cwe],
            "development_b0_detection_count": dev_detections[cwe],
            "heldout_b0_detection_count": heldout_detections[cwe],
            "scanner_rule_support": str(supported).lower(),
            "scanner_support_basis": "RULE_AND_OR_REGEX" if supported else "NONE",
            "causal_validation_sample_count": causal_count,
            "revision_causal_revalidation_repair_count": revision_count,
            "intervention_library_status": "PRESENT" if cwe in library_cwes else "ABSENT",
            "heldout_effect_denominator": heldout_denominator,
            "evidence_status": evidence,
            "reason": reason,
            "provenance_paths": source_paths,
            "provenance_hashes": source_hashes,
        })

    fields = list(rows[0])
    output = OUT / "cwe_evidence_support.csv"
    atomic_csv(output, fields, rows)
    atomic_json(OUT / "cwe_support_consistency.json", {
        "schema_version": "phase19_cwe_support_consistency_v1",
        "status": "PASS_WITH_SCOPE_LIMITATIONS",
        "target_cwe_count": len(rows),
        "evidence_status_counts": dict(collections.Counter(row["evidence_status"] for row in rows)),
        "support_mismatches": inconsistencies,
        "cwe338_status": next(row["evidence_status"] for row in rows if row["cwe"] == "CWE-338"),
        "cwe290_status": next(row["evidence_status"] for row in rows if row["cwe"] == "CWE-290"),
        "no_pooling_of_tiny_or_unsupported_cells": True,
        "generation_run": False,
        "scanner_run": False,
        "output": hash_record(output),
    })
    print(json.dumps({"status": "COMPLETE", "rows": len(rows), "evidence": dict(collections.Counter(row["evidence_status"] for row in rows))}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
