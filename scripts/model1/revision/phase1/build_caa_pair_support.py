#!/usr/bin/env python3
"""Audit preserved Phase 1 pairs and freeze CAA-CWE route support.

This script is intentionally read-only with respect to the submitted experiment.
It writes only the two revision artifacts required by REVISION_PLAN.md Phase 1.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


csv.field_size_limit(2**31 - 1)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
PHASE_DIR = PROJECT_ROOT / "revision" / "model1" / "phase1"

INPUTS = {
    "filtered": DATA_DIR / "cve_pairs_filtered.csv",
    "augmented": DATA_DIR / "crypto_augmented_pairs.csv",
    "full": DATA_DIR / "vulnerability_pairs_full.csv",
    "balanced": DATA_DIR / "vulnerability_pairs_balanced.csv",
    "train": DATA_DIR / "train_pairs.csv",
    "val": DATA_DIR / "val_pairs.csv",
    "test": DATA_DIR / "test_pairs.csv",
    "statistics": DATA_DIR / "dataset_statistics.csv",
    "cvefixes_license": PROJECT_ROOT / "data" / "raw" / "cvefixes" / "LICENSE.txt",
    "mono_readme": PROJECT_ROOT / "data" / "raw" / "mono" / "ReadMe.md",
}

OUTPUT_CSV = PHASE_DIR / "caa_pair_support.csv"
OUTPUT_RULE = PHASE_DIR / "caa_support_rule.json"

HELDOUT_CWES = {"CWE-22", "CWE-290"}
CANDIDATE_ROUTES = ("CWE-120", "CWE-327", "CWE-89", "CWE-338")
PRIMARY_ROUTES = {"CWE-120", "CWE-327", "CWE-89"}

EXPECTED_STAGE_COUNTS = {
    "CWE-120": (121, 0, 121, 0, 93, 0, 48, 12, 33),
    "CWE-125": (708, 0, 708, 0, 93, 0, 51, 25, 17),
    "CWE-190": (304, 0, 304, 0, 93, 0, 71, 14, 8),
    "CWE-22": (203, 0, 203, 0, 203, 0, 0, 0, 203),
    "CWE-290": (1, 0, 1, 0, 1, 0, 0, 0, 1),
    "CWE-327": (18, 13, 18, 13, 18, 13, 24, 6, 1),
    "CWE-338": (2, 27, 2, 27, 2, 2, 4, 0, 0),
    "CWE-476": (343, 0, 343, 0, 93, 0, 50, 25, 18),
    "CWE-787": (407, 0, 407, 0, 93, 0, 51, 14, 28),
    "CWE-79": (569, 0, 569, 0, 93, 0, 65, 14, 14),
    "CWE-89": (46, 0, 46, 0, 46, 0, 27, 11, 8),
}

CSV_COLUMNS = (
    "cwe_id",
    "raw_training_pairs",
    "augmented_training_pairs",
    "aligned_training_pairs",
    "scanner_supported",
    "source_verified",
    "evidence_tier",
    "support_status",
    "reason",
)

LINEAGE_FIELDS = (
    "pair_id",
    "cve_id",
    "cwe_id",
    "language",
    "vulnerable_code",
    "fixed_code",
    "source",
    "is_augmented",
    "seed_cve_id",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"Missing required input: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_augmented(row: dict[str, str]) -> bool:
    value = row.get("is_augmented", "").strip().lower()
    require(value in {"true", "false"}, f"Invalid is_augmented value for {row.get('pair_id')}: {value!r}")
    return value == "true"


def effective_unit(row: dict[str, str]) -> str:
    unit = row.get("seed_cve_id", "").strip() if is_augmented(row) else row.get("cve_id", "").strip()
    require(bool(unit), f"Missing effective CVE unit for {row.get('pair_id')}")
    return unit


def aligned(row: dict[str, str]) -> bool:
    vulnerable = row.get("vulnerable_code", "")
    fixed = row.get("fixed_code", "")
    return bool(vulnerable.strip()) and bool(fixed.strip()) and vulnerable != fixed


def exact_lineage(row: dict[str, str], upstream: dict[str, str]) -> bool:
    normalized_upstream = dict(upstream)
    if not is_augmented(normalized_upstream) and not normalized_upstream.get("seed_cve_id", "").strip():
        normalized_upstream["seed_cve_id"] = normalized_upstream.get("cve_id", "")
    return all(row.get(field, "") == normalized_upstream.get(field, "") for field in LINEAGE_FIELDS)


def count(rows: Iterable[dict[str, str]], cwe: str, augmented: bool | None = None) -> int:
    selected = [row for row in rows if row.get("cwe_id") == cwe]
    if augmented is None:
        return len(selected)
    return sum(is_augmented(row) is augmented for row in selected)


def main() -> None:
    filtered = read_csv(INPUTS["filtered"])
    augmented = read_csv(INPUTS["augmented"])
    full = read_csv(INPUTS["full"])
    balanced = read_csv(INPUTS["balanced"])
    splits = {name: read_csv(INPUTS[name]) for name in ("train", "val", "test")}
    statistics = read_csv(INPUTS["statistics"])

    require(len(filtered) == 8229, f"Filtered count changed: {len(filtered)}")
    require(len(augmented) == 40, f"Augmented count changed: {len(augmented)}")
    require(len(full) == 2762, f"Full count changed: {len(full)}")
    require(len(balanced) == 843, f"Balanced count changed: {len(balanced)}")
    require({name: len(rows) for name, rows in splits.items()} == {"train": 391, "val": 121, "test": 331}, "Split counts changed")
    require(len(statistics) == len(EXPECTED_STAGE_COUNTS), "dataset_statistics.csv CWE count changed")

    require(all(row.get("source") == "cvefixes" and not is_augmented(row) for row in filtered), "Filtered source labels are inconsistent")
    require(all(row.get("source") == "augmented" and is_augmented(row) for row in augmented), "Augmentation source labels are inconsistent")

    stage_counts: dict[str, dict[str, int]] = {}
    for cwe, expected in EXPECTED_STAGE_COUNTS.items():
        observed = (
            count(filtered, cwe),
            count(augmented, cwe),
            count(full, cwe, False),
            count(full, cwe, True),
            count(balanced, cwe, False),
            count(balanced, cwe, True),
            count(splits["train"], cwe),
            count(splits["val"], cwe),
            count(splits["test"], cwe),
        )
        require(observed == expected, f"Stage counts changed for {cwe}: expected {expected}, observed {observed}")
        stage_counts[cwe] = dict(
            zip(
                (
                    "target_raw",
                    "augmented_generated",
                    "full_raw",
                    "full_augmented",
                    "balanced_raw",
                    "balanced_augmented",
                    "train",
                    "val",
                    "test",
                ),
                observed,
            )
        )

    require(count(full, "CWE-338", False) == 2, "CWE-338 raw support is not two")
    require(count(full, "CWE-290", False) == 1, "CWE-290 raw support is not one")

    for cwe in EXPECTED_STAGE_COUNTS:
        total = count(balanced, cwe)
        aug_count = count(balanced, cwe, True)
        require(total > 0 and aug_count / total <= 0.5, f"Augmentation cap failed for {cwe}")

    all_split_rows: list[dict[str, str]] = []
    split_units: dict[str, set[str]] = {}
    split_pair_ids: dict[str, set[str]] = {}
    for split_name, rows in splits.items():
        units = {effective_unit(row) for row in rows}
        pair_ids = {row.get("pair_id", "") for row in rows}
        require("" not in pair_ids and len(pair_ids) == len(rows), f"Missing or duplicate pair_id inside {split_name}")
        split_units[split_name] = units
        split_pair_ids[split_name] = pair_ids
        for row in rows:
            tagged = dict(row)
            tagged["_split"] = split_name
            all_split_rows.append(tagged)

    overlap_counts = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        unit_overlap = split_units[left] & split_units[right]
        pair_overlap = split_pair_ids[left] & split_pair_ids[right]
        require(not unit_overlap, f"Effective CVE-unit overlap between {left} and {right}: {sorted(unit_overlap)}")
        require(not pair_overlap, f"Pair-ID overlap between {left} and {right}: {sorted(pair_overlap)}")
        overlap_counts[f"{left}_{right}"] = {"effective_units": 0, "pair_ids": 0}

    for split_name in ("train", "val"):
        leakage = [row["pair_id"] for row in splits[split_name] if row.get("cwe_id") in HELDOUT_CWES]
        require(not leakage, f"Held-out rows found in {split_name}: {leakage}")
    require(count(splits["test"], "CWE-22") == 203, "CWE-22 held-out test count changed")
    require(count(splits["test"], "CWE-290") == 1, "CWE-290 held-out test count changed")

    heldout_test_units = {effective_unit(row) for row in splits["test"] if row.get("cwe_id") in HELDOUT_CWES}
    require(len(heldout_test_units) == 73, f"Held-out effective-unit count changed: {len(heldout_test_units)}")
    require(not (heldout_test_units & split_units["train"]), "Held-out units overlap train")
    require(not (heldout_test_units & split_units["val"]), "Held-out units overlap validation")

    code_groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in all_split_rows:
        code_groups[(row["vulnerable_code"], row["fixed_code"])].append(row)
    duplicate_groups = [group for group in code_groups.values() if len(group) > 1]
    cross_split_groups = [group for group in duplicate_groups if len({row["_split"] for row in group}) > 1]
    cross_split_rows = sum(len(group) for group in cross_split_groups)
    heldout_cross_groups = [group for group in cross_split_groups if any(row["cwe_id"] in HELDOUT_CWES for row in group)]
    require(len(duplicate_groups) == 69, f"Duplicate code-group count changed: {len(duplicate_groups)}")
    require(len(cross_split_groups) == 47, f"Cross-split code-group count changed: {len(cross_split_groups)}")
    require(cross_split_rows == 102, f"Cross-split duplicate row count changed: {cross_split_rows}")
    require(len(heldout_cross_groups) == 6, f"Held-out cross-split code-group count changed: {len(heldout_cross_groups)}")

    filtered_by_pair = {row["pair_id"]: row for row in filtered}
    augmented_by_pair = {row["pair_id"]: row for row in augmented}
    require(len(filtered_by_pair) == len(filtered), "Duplicate pair_id in filtered input")
    require(len(augmented_by_pair) == len(augmented), "Duplicate pair_id in augmented input")
    raw_cves_by_cwe = defaultdict(set)
    for row in filtered:
        raw_cves_by_cwe[row["cwe_id"]].add(row["cve_id"])

    support_rows: list[dict[str, object]] = []
    route_details: dict[str, dict[str, object]] = {}
    for cwe in CANDIDATE_ROUTES:
        route_rows = [row for row in splits["train"] if row["cwe_id"] == cwe]
        raw_rows = [row for row in route_rows if not is_augmented(row)]
        aug_rows = [row for row in route_rows if is_augmented(row)]
        lineage_results: list[bool] = []
        aligned_rows: list[dict[str, str]] = []

        for row in route_rows:
            upstream_index = augmented_by_pair if is_augmented(row) else filtered_by_pair
            upstream = upstream_index.get(row["pair_id"])
            lineage_ok = upstream is not None and exact_lineage(row, upstream)
            if is_augmented(row):
                lineage_ok = lineage_ok and row["seed_cve_id"] in raw_cves_by_cwe[cwe]
            lineage_results.append(lineage_ok)
            if aligned(row) and lineage_ok:
                aligned_rows.append(row)

        source_verified = bool(route_rows) and all(lineage_results)
        aligned_count = len(aligned_rows)
        if not source_verified:
            status = "PROVENANCE_UNVERIFIED"
        elif aligned_count == 0:
            status = "UNAVAILABLE_IN_TRAIN_SPLIT"
        elif cwe in PRIMARY_ROUTES:
            status = "SUPPORTED"
        else:
            status = "EXPLORATORY_SUPPORT"

        if cwe == "CWE-338":
            evidence_tier = "EXPLORATORY_SMALL_N_MIXED"
        elif aug_rows:
            evidence_tier = "PRIMARY_MIXED_RAW_AUGMENTED"
        else:
            evidence_tier = "PRIMARY_RAW"

        reason = (
            f"{len(raw_rows)} raw and {len(aug_rows)} augmented training pairs; "
            f"{aligned_count} non-empty, non-identical pairs have verified upstream lineage."
        )
        if cwe == "CWE-338":
            reason += " Exploratory because only two raw pairs exist before augmentation."

        support_rows.append(
            {
                "cwe_id": cwe,
                "raw_training_pairs": len(raw_rows),
                "augmented_training_pairs": len(aug_rows),
                "aligned_training_pairs": aligned_count,
                "scanner_supported": "true",
                "source_verified": str(source_verified).lower(),
                "evidence_tier": evidence_tier,
                "support_status": status,
                "reason": reason,
            }
        )
        route_details[cwe] = {
            "raw_training_pairs": len(raw_rows),
            "augmented_training_pairs": len(aug_rows),
            "aligned_training_pairs": aligned_count,
            "effective_training_units": len({effective_unit(row) for row in route_rows}),
            "languages": sorted({row["language"] for row in route_rows}),
            "source_verified": source_verified,
            "support_status": status,
        }

    expected_statuses = {
        "CWE-120": "SUPPORTED",
        "CWE-327": "SUPPORTED",
        "CWE-89": "SUPPORTED",
        "CWE-338": "EXPLORATORY_SUPPORT",
    }
    require({row["cwe_id"]: row["support_status"] for row in support_rows} == expected_statuses, "Candidate support statuses differ from the pre-execution audit")
    require({row["cwe_id"]: row["aligned_training_pairs"] for row in support_rows} == {"CWE-120": 13, "CWE-327": 17, "CWE-89": 13, "CWE-338": 4}, "Aligned candidate counts changed")

    input_hashes = {
        name: {"path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"), "sha256": sha256(path), "bytes": path.stat().st_size}
        for name, path in INPUTS.items()
    }
    mono_license_candidates = [
        PROJECT_ROOT / "data" / "raw" / "mono" / "LICENSE",
        PROJECT_ROOT / "data" / "raw" / "mono" / "LICENSE.txt",
        PROJECT_ROOT / "data" / "raw" / "mono" / "COPYING",
    ]

    rule = {
        "audit_version": 1,
        "alignment_rule": {
            "requirements": [
                "training split membership",
                "non-empty vulnerable_code",
                "non-empty fixed_code",
                "vulnerable_code differs byte-for-byte from fixed_code",
                "exact pair and metadata lineage to the appropriate raw or augmented upstream file",
                "augmented pairs inherit seed_cve_id from a raw same-CWE upstream CVE",
            ],
            "minimum_aligned_pairs": 1,
            "tuned": False,
        },
        "candidate_routes": list(CANDIDATE_ROUTES),
        "excluded_routes": {"CWE-290": "Scope-boundary held-out CWE; not scanner-supported for revision CAA routing."},
        "status_rule": {
            "primary_constructible": "SUPPORTED",
            "cwe_338_constructible": "EXPLORATORY_SUPPORT",
            "no_aligned_training_pair": "UNAVAILABLE_IN_TRAIN_SPLIT",
            "lineage_failure": "PROVENANCE_UNVERIFIED",
        },
        "anti_tuning_statement": "Evidence strength is reported separately from route constructibility and is never adjusted after CAA outputs are observed.",
        "verified_route_counts": route_details,
        "stage_counts": stage_counts,
        "split_integrity": {
            "effective_unit_definition": "seed_cve_id for augmented rows; cve_id for raw rows",
            "effective_unit_counts": {name: len(units) for name, units in split_units.items()},
            "pairwise_overlap_counts": overlap_counts,
            "heldout_cwes": sorted(HELDOUT_CWES),
            "heldout_test_rows": 204,
            "heldout_test_effective_units": len(heldout_test_units),
            "heldout_train_rows": 0,
            "heldout_val_rows": 0,
        },
        "content_duplicate_warning": {
            "classification": "INCONSISTENT",
            "duplicate_code_pair_groups": len(duplicate_groups),
            "cross_split_code_pair_groups": len(cross_split_groups),
            "rows_in_cross_split_groups": cross_split_rows,
            "cross_split_groups_involving_heldout_cwes": len(heldout_cross_groups),
            "statement": "CVE/seed-CVE units are disjoint, but exact vulnerable/fixed code pairs cross split boundaries; this is a preserved submitted-data limitation, not complete content isolation.",
        },
        "small_cwe_evidence": {
            "CWE-338": {"raw_pairs": 2, "classification": "EXPLORATORY_SMALL_N"},
            "CWE-290": {"raw_pairs": 1, "classification": "SCOPE_BOUNDARY_HELDOUT"},
        },
        "licensing_and_redistribution": {
            "cvefixes_repository_license": {
                "status": "PRESENT",
                "license": "MIT",
                "path": "data/raw/cvefixes/LICENSE.txt",
                "sha256": input_hashes["cvefixes_license"]["sha256"],
            },
            "third_party_source_snippets": {
                "status": "UNRESOLVED",
                "reason": "The CVEFixes repository license does not establish redistribution rights for source snippets originating in third-party projects."
            },
            "mono_repository_license": {
                "status": "MISSING_TOP_LEVEL_LICENSE" if not any(path.is_file() for path in mono_license_candidates) else "PRESENT",
                "reason": "No top-level LICENSE, LICENSE.txt, or COPYING file was found in the pinned Mono checkout during this audit."
            },
            "phase20_action": "Use conservative artifact packaging and record per-origin licensing limitations before redistribution.",
        },
        "input_artifacts": input_hashes,
    }

    PHASE_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(support_rows)
    with OUTPUT_RULE.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(rule, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")

    print("Phase 1 CAA support audit: PASS")
    print(f"Wrote {OUTPUT_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {OUTPUT_RULE.relative_to(PROJECT_ROOT)}")
    for row in support_rows:
        print(f"{row['cwe_id']}: {row['support_status']} ({row['aligned_training_pairs']} aligned)")


if __name__ == "__main__":
    main()
