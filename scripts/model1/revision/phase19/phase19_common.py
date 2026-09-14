#!/usr/bin/env python3
"""Shared helpers for the Model 1 statistical sensitivity analyses."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

csv.field_size_limit(min(sys.maxsize, 2_147_483_647))

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "revision" / "model1" / "phase19" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

TARGET_CWES = [
    "CWE-120", "CWE-125", "CWE-787", "CWE-190", "CWE-476",
    "CWE-89", "CWE-79", "CWE-327", "CWE-338", "CWE-22", "CWE-290",
]

FROZEN_BSTAR = {
    "CWE-120": (19, 14193),
    "CWE-327": (23, 14449),
    "CWE-89": (23, 1652),
    "CWE-338": (23, 7533),
    "CWE-79": (16, 9816),
    "CWE-125": (23, 16655),
    "CWE-787": (19, 1515),
    "CWE-190": (19, 16897),
    "CWE-476": (23, 18397),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def hash_record(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing input: {rel(path)}")
    return {"path": rel(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
