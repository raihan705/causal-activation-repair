#!/usr/bin/env python3
"""Verify reviewed Phase 14 inputs before CAA Stage B execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
OUTPUTS = ROOT / "revision" / "model1" / "phase14" / "outputs"

EXPECTED = {
    "revision/model1/phase14/outputs/phase14_development_metrics.json": "d2b611214adcf517f602dbf25029847aefabafab30b54088bcad2ba8229d6e88",
    "revision/model1/phase14/outputs/rci_selection.json": "c81e75be759f64784afd2a3c98e730a4f16ed7f6465be5ad8a207a557f9b1753",
    "revision/model1/phase14/outputs/caa_stage_a_selection.json": "bdd2b491bbef5e102392b0b5a8177b2f47f3bb230ad20210e7bc6feaafa7125a",
    "revision/model1/phase14/outputs/phase14_method_statuses.json": "590425acc190cfa060be557c674eb6082569d74c4ba0930785ac94f634a979fe",
    "revision/model1/phase14/outputs/rci_human_audit_sheet_labeled.json": "7fcdf074e717cf5d1ebba6b060ed027306fb748aa696c7669f0cd1f995e95c5b",
    "revision/model1/phase14/outputs/rci_human_audit_sheet.json": "fbfc190316406567fc480a26853c37903b7c985f31e2281100f4f8d116ccb2fa",
    "revision/model1/phase14/outputs/phase14_development_scan_return_manifest.json": "e6f876677a163d6d3ada8c1183c2c4bf7e57803d8fcb3919214de70d0ba03731",
    "revision/model1/phase14/outputs/phase14_development_scan_local_verification.json": "7e03a0c417b9f752b48a82f05c9b286254b11ed65f250dcf0583c9986788f7cb",
    "revision/model1/phase14/outputs/caa_vector_manifest.json": "172e252e8e7955491bfed1118258cc6f39ec2a3e59b099dbd4df31a49270f36e",
    "revision/model1/phase14/outputs/caa_stage_a_layer16_mult1p0_seed42.json": "75557e7841cc5d252551df36ffd32db76525418a9f2082ebca4b1990d55745e0",
    "revision/model1/phase14/outputs/caa_stage_a_layer16_mult1p0_seed42_icd.json": "f1a26974fd5cb1cc50cbbf7ca8f5b06e23a1c646452dec38fe64c763c21a9fc3",
    "revision/model1/phase14/scripts/run_caa_cwe_dev.py": "90a8fb659d1aa83e2c4c0279caeecedd7c7518825f0cbe8e2660a490c179b632",
    "revision/model1/phase14/scripts/select_caa_configuration.py": "5a567ee0dd530dff25d90f538bc279011ac33ef578dde66c0404dd7461518634",
    "phases/phase9/colab_scan_phase9.py": "60cedd3d6ec2d19e5e7e5ba4d709230efdd5d9478047492fefda4dd88d9afbb7",
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    files: list[dict[str, Any]] = []
    errors: list[str] = []
    for relative, expected in EXPECTED.items():
        path = ROOT / relative
        actual = sha256_path(path) if path.is_file() else None
        matches = actual == expected
        files.append({"path": relative, "expected_sha256": expected, "actual_sha256": actual,
                      "matches_reviewed_hash": matches})
        if not matches:
            errors.append(f"reviewed hash mismatch: {relative}")

    vector_manifest = json.loads((OUTPUTS / "caa_vector_manifest.json").read_text(encoding="utf-8"))
    vectors = []
    for entry in vector_manifest["entries"]:
        relative = entry["artifact_path"]
        expected = entry["artifact_sha256"]
        actual = sha256_path(ROOT / relative) if (ROOT / relative).is_file() else None
        matches = actual == expected
        vectors.append({"path": relative, "expected_sha256": expected, "actual_sha256": actual,
                        "matches_manifest": matches, "layer": entry["layer"], "cwe_id": entry["cwe_id"]})
        if not matches:
            errors.append(f"CAA vector hash mismatch: {relative}")

    forbidden = [
        "caa_stage_b_layer16_mult0p5_seed42.json",
        "caa_stage_b_layer16_mult2p0_seed42.json",
        "caa_stage_b_layer16_mult4p0_seed42.json",
    ]
    present = [name for name in forbidden if (OUTPUTS / name).exists()]
    if present:
        errors.append(f"Stage B outcome files existed before the freeze: {present}")

    result = {
        "schema_version": "phase14_stage_b_input_verification_v1",
        "status": "PASS" if not errors else "FAIL",
        "reviewed_files": files,
        "caa_vectors": vectors,
        "stage_b_outcome_files_present_before_freeze": present,
        "errors": errors,
        "held_out_accessed": False,
    }
    output = OUTPUTS / "phase14_stage_b_input_verification.json"
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    if errors:
        raise SystemExit("; ".join(errors))


if __name__ == "__main__":
    main()
