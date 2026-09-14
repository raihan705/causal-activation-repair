#!/usr/bin/env python3
"""Build a compact Colab bundle for official HumanEval evaluation."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from routewise_utility_common import (
    EXPERIMENT, OUTPUTS, atomic_json, baseline_output, baseline_run_manifest,
    canonical_sha256, read_json, require, route_slug, sha256_path,
)
from run_reduced_routewise_utility import (
    PROTOCOL, load_reduced_protocol, output_path, run_manifest_path,
)


EVALUATOR = Path(__file__).with_name("evaluate_reduced_routewise_humaneval.py")
INPUT = OUTPUTS / "reduced_humaneval_evaluation_input.json"
MANIFEST = OUTPUTS / "reduced_humaneval_bundle_manifest.json"
BUNDLE = EXPERIMENT / "reduced_routewise_humaneval_colab_bundle.zip"


def validated_output(path: Path, manifest_path: Path, expected: int) -> list[dict[str, Any]]:
    require(path.is_file() and manifest_path.is_file(), f"Missing generation artifact: {path.name}")
    run = read_json(manifest_path)
    require(run.get("status") == "COMPLETE", f"{manifest_path.name} is not COMPLETE")
    require(run.get("output_sha256") == sha256_path(path), f"{path.name} hash mismatch")
    rows = read_json(path)
    require(isinstance(rows, list) and len(rows) == expected, f"{path.name} count mismatch")
    return rows


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
    protocol = load_reduced_protocol()
    baseline = validated_output(baseline_output(), baseline_run_manifest(), 1716)
    b0_human = [row for row in baseline if row["benchmark"] == "humaneval"]
    require(len(b0_human) == 164, "B0 HumanEval count mismatch")
    task_ids = [row["stable_id"] for row in b0_human]
    conditions: dict[str, Any] = {
        "B0": {
            "slug": "b0",
            "layer": None,
            "feature": None,
            "source_output_sha256": sha256_path(baseline_output()),
            "records": [{"task_id": row["stable_id"], "completion": row["generated_text"]}
                        for row in b0_human],
        }
    }
    for cwe_id, route in protocol["routes"].items():
        path = output_path(cwe_id, route)
        rows = validated_output(path, run_manifest_path(cwe_id, route), 726)
        human = [row for row in rows if row["benchmark"] == "humaneval"]
        require([row["stable_id"] for row in human] == task_ids, f"{cwe_id} HumanEval order mismatch")
        conditions[cwe_id] = {
            "slug": route_slug(cwe_id, route),
            "layer": int(route["layer"]),
            "feature": int(route["feature"]),
            "source_output_sha256": sha256_path(path),
            "records": [{"task_id": row["stable_id"], "completion": row["generated_text"]}
                        for row in human],
        }
    payload = {
        "schema_version": "reduced_routewise_humaneval_input_v1",
        "status": "FROZEN",
        "protocol_sha256": sha256_path(PROTOCOL),
        "task_count_per_condition": 164,
        "condition_count": len(conditions),
        "task_ids_sha256": canonical_sha256(task_ids),
        "conditions": conditions,
    }
    atomic_json(INPUT, payload)
    deterministic_zip(BUNDLE, [(INPUT.name, INPUT), (EVALUATOR.name, EVALUATOR)])
    manifest = {
        "schema_version": "reduced_routewise_humaneval_bundle_manifest_v1",
        "status": "COMPLETE",
        "bundle_path": str(BUNDLE.relative_to(EXPERIMENT.parent.parent.parent)).replace("\\", "/"),
        "bundle_sha256": sha256_path(BUNDLE),
        "members": {
            INPUT.name: {"sha256": sha256_path(INPUT), "bytes": INPUT.stat().st_size},
            EVALUATOR.name: {"sha256": sha256_path(EVALUATOR), "bytes": EVALUATOR.stat().st_size},
        },
        "condition_count": len(conditions),
        "task_count_per_condition": 164,
    }
    atomic_json(MANIFEST, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

