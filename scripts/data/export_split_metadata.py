#!/usr/bin/env python3
"""Export metadata and code hashes without redistributing source/fixed code."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


REQUIRED_COLUMNS = {
    "pair_id",
    "cve_id",
    "cwe_id",
    "language",
    "source",
    "is_augmented",
    "seed_cve_id",
    "vulnerable_code",
    "fixed_code",
    "role",
}

OUTPUT_COLUMNS = [
    "split",
    "source_order",
    "pair_id",
    "cve_id",
    "cwe_id",
    "language",
    "source",
    "is_augmented",
    "seed_cve_id",
    "role",
    "vulnerable_code_sha256",
    "fixed_code_sha256",
    "code_pair_sha256",
]


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_split(path: Path, split: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for source_order, row in enumerate(reader):
            vulnerable = row["vulnerable_code"]
            fixed = row["fixed_code"]
            vulnerable_hash = sha256_text(vulnerable)
            fixed_hash = sha256_text(fixed)
            rows.append(
                {
                    "split": split,
                    "source_order": source_order,
                    "pair_id": row["pair_id"],
                    "cve_id": row["cve_id"],
                    "cwe_id": row["cwe_id"],
                    "language": row["language"],
                    "source": row["source"],
                    "is_augmented": row["is_augmented"],
                    "seed_cve_id": row["seed_cve_id"],
                    "role": row["role"],
                    "vulnerable_code_sha256": vulnerable_hash,
                    "fixed_code_sha256": fixed_hash,
                    "code_pair_sha256": sha256_text(vulnerable_hash + ":" + fixed_hash),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    rows.extend(read_split(args.train, "train"))
    rows.extend(read_split(args.validation, "validation"))
    rows.extend(read_split(args.test, "test"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} metadata-only rows to {args.output}")


if __name__ == "__main__":
    main()
