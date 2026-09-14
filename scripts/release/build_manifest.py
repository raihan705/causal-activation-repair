#!/usr/bin/env python3
"""Build a deterministic SHA-256 inventory for the public release."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HASH_FILE = ROOT / "manifests" / "SHA256SUMS"
MANIFEST_FILE = ROOT / "manifests" / "release_manifest.json"
EXCLUDED = {HASH_FILE.resolve(), MANIFEST_FILE.resolve()}
IGNORED_DIRECTORY_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "logs",
    "scratch",
    "tmp",
    "venv",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    files = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.resolve() not in EXCLUDED
        and not IGNORED_DIRECTORY_NAMES.intersection(path.relative_to(ROOT).parts)
    )
    entries = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in files
    ]

    HASH_FILE.parent.mkdir(parents=True, exist_ok=True)
    HASH_FILE.write_text(
        "".join(f"{entry['sha256']}  {entry['path']}\n" for entry in entries),
        encoding="utf-8",
    )
    MANIFEST_FILE.write_text(
        json.dumps(
            {
                "schema_version": "causal_activation_repair_release_manifest_v1",
                "file_count_excluding_generated_manifests": len(entries),
                "total_bytes_excluding_generated_manifests": sum(e["size_bytes"] for e in entries),
                "files": entries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Recorded {len(entries)} files")


if __name__ == "__main__":
    main()
