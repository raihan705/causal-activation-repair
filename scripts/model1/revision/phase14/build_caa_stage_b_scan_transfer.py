#!/usr/bin/env python3
"""Build the deterministic three-condition CAA Stage B scan manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
OUTPUTS = ROOT / "revision" / "model1" / "phase14" / "outputs"
SCANNER = ROOT / "phases" / "phase9" / "colab_scan_phase9.py"
SUBSET = ROOT / "revision" / "model1" / "phase13" / "outputs" / "baseline_selection_subset.json"
VECTOR_MANIFEST = OUTPUTS / "caa_vector_manifest.json"
VALIDATION = OUTPUTS / "caa_stage_b_generation_structural_validation.json"

CONDITIONS = [
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


def main() -> None:
    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))
    if validation.get("status") != "PASS":
        raise RuntimeError("CAA Stage B generation structural validation did not pass")
    entries = []
    for order, (key, multiplier, stem) in enumerate(CONDITIONS, 1):
        generation = OUTPUTS / f"{stem}.json"
        run = OUTPUTS / f"{stem}_run_manifest.json"
        entries.append({
            "order": order,
            "key": key,
            "method": "CAA-CWE-StageB",
            "information_tier": "ORACLE_CWE",
            "layer": 16,
            "multiplier": multiplier,
            "seed": 42,
            "generation_path": generation.name,
            "generation_sha256": sha256_path(generation),
            "run_manifest_path": run.name,
            "run_manifest_sha256": sha256_path(run),
            "record_count": 120,
            "prompt_manifest_path": SUBSET.name,
            "prompt_manifest_sha256": sha256_path(SUBSET),
            "vector_manifest_path": VECTOR_MANIFEST.name,
            "vector_manifest_sha256": sha256_path(VECTOR_MANIFEST),
            "frozen_scanner_source_path": SCANNER.name,
            "frozen_scanner_source_sha256": sha256_path(SCANNER),
            "icd_path": f"{stem}_icd.json",
            "scan_status": "NOT_STARTED",
        })
    result = {
        "schema_version": "phase14_caa_stage_b_scan_transfer_manifest_v1",
        "phase": 14,
        "status": "FROZEN_AWAITING_COLAB_SCANS",
        "entry_count": 3,
        "completion_order": ["caa_l16_m0p5", "caa_l16_m2", "caa_l16_m4"],
        "existing_l16_m1_reused_not_rescanned": True,
        "generation_files_immutable_after_inclusion": True,
        "scanner_executed": False,
        "held_out_accessed": False,
        "structural_validation_path": VALIDATION.name,
        "structural_validation_sha256": sha256_path(VALIDATION),
        "entries": entries,
    }
    output = OUTPUTS / "caa_stage_b_scan_transfer_manifest.json"
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)


if __name__ == "__main__":
    main()
