#!/usr/bin/env python3
"""Package completed route-wise generation outputs for isolated HumanEval evaluation."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from routewise_utility_common import (
    EXPERIMENT, OUTPUTS, PROTOCOL, load_protocol, read_json, require,
    route_output, route_run_manifest, sha256_path,
)


EVALUATOR = Path(__file__).with_name("evaluate_routewise_humaneval.py")
BUNDLE = EXPERIMENT / "routewise_humaneval_colab_bundle.zip"
MANIFEST = OUTPUTS / "routewise_humaneval_bundle_manifest.json"


def deterministic_zip(path: Path, members: list[tuple[str, Path]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for archive_name, source in sorted(members):
            info = zipfile.ZipInfo(archive_name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, source.read_bytes())
    temporary.replace(path)


def main() -> int:
    protocol = load_protocol()
    members: list[tuple[str, Path]] = [
        (PROTOCOL.name, PROTOCOL),
        ("evaluate_routewise_humaneval.py", EVALUATOR),
        ("paired_b0_seed42_outputs.json", OUTPUTS / "paired_b0_seed42_outputs.json"),
        ("paired_b0_seed42_run_manifest.json", OUTPUTS / "paired_b0_seed42_run_manifest.json"),
    ]
    for cwe_id, route in protocol["routes"].items():
        output = route_output(cwe_id, route)
        run_manifest = route_run_manifest(cwe_id, route)
        require(output.is_file() and run_manifest.is_file(), f"Missing completed route: {cwe_id}")
        run = read_json(run_manifest)
        require(run.get("status") == "COMPLETE" and run.get("output_sha256") == sha256_path(output),
                f"Route output is not verified COMPLETE: {cwe_id}")
        members.extend([(output.name, output), (run_manifest.name, run_manifest)])
    for _, source in members:
        require(source.is_file(), f"Missing bundle member: {source}")
    deterministic_zip(BUNDLE, members)
    manifest = {
        "schema_version": "routewise_humaneval_bundle_manifest_v1",
        "status": "COMPLETE",
        "bundle": BUNDLE.name,
        "bundle_sha256": sha256_path(BUNDLE),
        "members": {name: {"sha256": sha256_path(source), "bytes": source.stat().st_size}
                    for name, source in members},
    }
    from routewise_utility_common import atomic_json
    atomic_json(MANIFEST, manifest)
    print(json.dumps({"status": "COMPLETE", "bundle": str(BUNDLE),
                      "bundle_sha256": manifest["bundle_sha256"],
                      "member_count": len(members)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

