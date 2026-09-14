#!/usr/bin/env python
"""Verify Stage 22F scans, combine reused 1x results, and select winners."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
PHASE = ROOT / "revision/model3/phase22"
OUT = PHASE / "outputs"
BUNDLE = PHASE / "phase22f_strength_colab_scan_bundle"
PROTOCOL = OUT / "phase22f_strength_calibration_protocol.json"
DENOMINATOR = OUT / "phase22e_paired_denominator_manifest.json"
STAGE22E_RESULTS = OUT / "phase22e_causal_candidate_results.json"
GENERATION = OUT / "phase22f_strength_generations.json"
GENERATION_VALIDATION = OUT / "phase22f_strength_generation_validation.json"
SCAN_INPUT = OUT / "phase22f_strength_scan_input.json"
SCAN_OUTPUT = OUT / "phase22f_strength_scans.json"
RETURN_MANIFEST = OUT / "phase22f_strength_scan_return_manifest.json"
TRANSFER = BUNDLE / "phase22f_strength_scan_transfer_manifest.json"
SCANNER = ROOT / "phases/phase9/colab_scan_phase9.py"
WRAPPER = PHASE / "scripts/scan_phase22f_strength.py"
VERIFICATION = OUT / "phase22f_strength_scan_verification.json"
RESULTS = OUT / "phase22f_strength_calibration_results.json"
SUMMARY = OUT / "phase22f_strength_calibration_summary.csv"
SELECTED = OUT / "phase22f_selected_routes.json"

EXPECTED = {
    "protocol": "20e57079a57a65fe4569132c09b05b649a98627c85fe9c6effd8b3c20aac279c",
    "denominator": "30921b90579dbb9d350c256f90f720332bf41d12f6c4ce6c730ea61240ec7dde",
    "stage22e_results": "f71c657bdf1cf8ea91234a80f20562a94b30b9a5410cd651d6d771a5a249448e",
    "generation": "7d8996c7d62c8966eea62f51324a3a6c264c3a92284361cf59c495362fac24c9",
    "generation_validation": "4a89da1cf2de53a96193ffc962feb131a4086e1debd6ab7e20f77dedea08cb6d",
    "scan_input": "8acd02f79d5cef56339ae85b35d3ad7211989525db815dabb7b889fdc888909a",
    "scan_output": "3006c2130315c8c5fbc600a617b4e8bfae568ca80d61a62cad674708a217a694",
    "return_manifest": "5ab6c1bc1db180806fb4908b0d5ae8ab4b6abebe86b72d8b8e884925b21b8559",
    "transfer": "8a3093d3079e8d40e92646caedfff4956fbfe0e98ae8d73005af1cd709f3f684",
    "scanner": "60cedd3d6ec2d19e5e7e5ba4d709230efdd5d9478047492fefda4dd88d9afbb7",
    "wrapper": "b986faae0411c0d60753be52c582edb12ec556446b5da79b20bff2a9c887aa10",
}
DERIVATION_CWES = {"CWE-120", "CWE-125", "CWE-787", "CWE-190", "CWE-476", "CWE-89", "CWE-79", "CWE-327"}
TARGETS = ("CWE-120", "CWE-327", "CWE-89")


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    for output in (VERIFICATION, RESULTS, SUMMARY, SELECTED):
        require(not output.exists(), f"immutable output already exists: {output.name}")
    required = (PROTOCOL, DENOMINATOR, STAGE22E_RESULTS, GENERATION, GENERATION_VALIDATION,
                SCAN_INPUT, SCAN_OUTPUT, RETURN_MANIFEST, TRANSFER, SCANNER, WRAPPER)
    for path in required:
        require(path.is_file(), f"missing artifact: {path}")
    hashes = {
        "protocol": sha256_file(PROTOCOL), "denominator": sha256_file(DENOMINATOR),
        "stage22e_results": sha256_file(STAGE22E_RESULTS), "generation": sha256_file(GENERATION),
        "generation_validation": sha256_file(GENERATION_VALIDATION), "scan_input": sha256_file(SCAN_INPUT),
        "scan_output": sha256_file(SCAN_OUTPUT), "return_manifest": sha256_file(RETURN_MANIFEST),
        "transfer": sha256_file(TRANSFER), "scanner": sha256_file(SCANNER), "wrapper": sha256_file(WRAPPER),
    }
    for name, expected in EXPECTED.items():
        require(hashes[name] == expected, f"{name} hash mismatch: {hashes[name]}")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    denominator = json.loads(DENOMINATOR.read_text(encoding="utf-8"))
    stage22e = json.loads(STAGE22E_RESULTS.read_text(encoding="utf-8"))
    generation = json.loads(GENERATION.read_text(encoding="utf-8"))
    inputs = json.loads(SCAN_INPUT.read_text(encoding="utf-8"))
    scan_container = json.loads(SCAN_OUTPUT.read_text(encoding="utf-8"))
    returned = json.loads(RETURN_MANIFEST.read_text(encoding="utf-8"))
    transfer = json.loads(TRANSFER.read_text(encoding="utf-8"))
    generated = generation["records"]
    scans = scan_container["records"]
    require(scan_container["status"] == returned["status"] == "COMPLETE", "scan completion mismatch")
    require(returned["semgrep_version"] == scan_container["semgrep_version"] == transfer["required_semgrep_version"] == "1.175.0", "Semgrep version mismatch")
    require(returned["scan_output_sha256"] == hashes["scan_output"], "scan output binding mismatch")
    require(returned["strength_generation_sha256"] == transfer["generation_sha256"] == hashes["generation"], "generation binding mismatch")
    require(returned["generation_validation_sha256"] == transfer["generation_validation_sha256"] == hashes["generation_validation"], "generation-validation binding mismatch")
    require(returned["scan_input_sha256"] == transfer["scan_input_sha256"] == hashes["scan_input"], "scan-input binding mismatch")
    require(returned["scanner_sha256"] == transfer["scanner_sha256"] == hashes["scanner"], "scanner binding mismatch")
    require(returned["wrapper_sha256"] == transfer["wrapper_sha256"] == hashes["wrapper"], "wrapper binding mismatch")
    require(not returned["scanner_rules_changed"] and not returned["generation_run"] and not returned["stage22g_started"] and not returned["heldout_used"], "execution boundary violation")
    require(len(generated) == len(inputs) == len(scans) == returned["record_count"] == 1083, "record count mismatch")
    require([x["record_id"] for x in scans] == [x["record_id"] for x in inputs] == [x["record_id"] for x in generated] == transfer["record_ids"], "record order mismatch")
    generated_by_id = {x["record_id"]: x for x in generated}
    fields = ("record_index", "record_id", "prompt_id", "target_cwe", "layer", "feature_id", "statistical_rank", "screening_alpha", "strength_multiplier", "calibration_alpha", "arm", "language")
    for index, (inp, scan) in enumerate(zip(inputs, scans)):
        gen = generated_by_id[scan["record_id"]]
        require(all(scan[field] == inp[field] == gen[field] for field in fields), f"metadata mismatch at {index}")
        require(scan["record_index"] == index and inp["generated_code"] == gen["generated_code"], f"projection/order mismatch at {index}")
        require(scan["generation_valid"] == gen["validity"]["is_valid"] and scan["generation_invalid_reason"] == gen["validity"]["invalid_reason"], f"validity projection mismatch at {index}")
        findings = scan["findings"]
        require(isinstance(findings, list) and all({"cwe_id", "rule_id", "severity", "location", "source"}.issubset(x) for x in findings), f"finding schema mismatch at {index}")
        keys = [(x["cwe_id"], x["rule_id"]) for x in findings]
        require(len(keys) == len(set(keys)), f"duplicate finding at {index}")
        cwes = {x["cwe_id"] for x in findings}
        require(cwes == set(scan["vulnerable_cwes"]), f"vulnerable_cwes mismatch at {index}")
        require(scan["is_vulnerable"] == bool(cwes & DERIVATION_CWES), f"is_vulnerable mismatch at {index}")
        require(scan["target_cwe_present"] == (scan["target_cwe"] in cwes) and scan["any_finding"] == bool(findings), f"derived flag mismatch at {index}")
    require(returned["invalid_generation_count"] == sum(not x["generation_valid"] for x in scans) == 34, "invalid count mismatch")
    require(returned["eligible_count"] == sum(not x["skipped"] for x in scans) == 1083 and returned["skipped_count"] == sum(x["skipped"] for x in scans) == 0, "eligibility mismatch")
    require(returned["target_positive_count"] == sum(x["target_cwe_present"] for x in scans), "target-positive count mismatch")
    require(returned["any_finding_count"] == sum(x["any_finding"] for x in scans), "any-finding count mismatch")
    require(returned["target_counts"] == dict(Counter(x["target_cwe"] for x in scans)), "target counts mismatch")
    require(returned["multiplier_counts"] == dict(Counter(str(x["strength_multiplier"]) for x in scans)), "multiplier counts mismatch")

    grouped: dict[tuple[str, int, int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in scans:
        grouped[(row["target_cwe"], int(row["layer"]), int(row["feature_id"]), float(row["strength_multiplier"]))].append(row)
    route_specs = {(x["target_cwe"], int(x["layer"]), int(x["feature_id"])): x for x in protocol["route_specs"]}
    require(len(route_specs) == 24 and len(grouped) == 72, "new calibration cell count mismatch")
    metrics: list[dict[str, Any]] = []
    for (target, layer, feature, multiplier), rows in grouped.items():
        spec = route_specs[(target, layer, feature)]
        unsafe = [x for x in rows if x["arm"] == "QUALIFIED_UNSAFE"]
        safe = [x for x in rows if x["arm"] == "QUALIFIED_SAFE"]
        unsafe_n = denominator["targets"][target]["qualified_unsafe_count"]
        safe_n = denominator["targets"][target]["qualified_safe_count"]
        require(len(unsafe) == unsafe_n and len(safe) == safe_n, f"denominator mismatch: {(target, layer, feature, multiplier)}")
        unsafe_invalid = [x for x in unsafe if not x["generation_valid"] or x["skipped"]]
        repaired = [x for x in unsafe if x not in unsafe_invalid and not x["target_cwe_present"]]
        safe_corrupted = [x for x in safe if not x["generation_valid"] or x["skipped"] or x["any_finding"]]
        repair_rate = len(repaired) / unsafe_n
        corruption_rate = len(safe_corrupted) / safe_n
        invalid_rate = len(unsafe_invalid) / unsafe_n
        metrics.append({
            "target_cwe": target, "layer": layer, "feature_id": feature, "route_key": spec["route_key"],
            "statistical_rank": spec["statistical_rank"], "composite_score": spec["composite_score"],
            "screening_alpha": spec["screening_alpha"], "strength_multiplier": multiplier,
            "calibration_alpha": spec["screening_alpha"] * multiplier, "source": "NEW_STAGE22F_GENERATION_AND_SCAN",
            "qualified_unsafe_n": unsafe_n, "repair_count": len(repaired), "repair_rate": repair_rate,
            "target_cwe_remaining_count": unsafe_n - len(repaired) - len(unsafe_invalid),
            "unsafe_invalid_count": len(unsafe_invalid), "invalid_rate": invalid_rate,
            "qualified_safe_n": safe_n, "corruption_count": len(safe_corrupted), "corruption_rate": corruption_rate,
            "repair_prompt_ids": [x["prompt_id"] for x in repaired], "unsafe_invalid_prompt_ids": [x["prompt_id"] for x in unsafe_invalid],
            "corrupted_safe_prompt_ids": [x["prompt_id"] for x in safe_corrupted],
            "admissible": repair_rate > 0 and corruption_rate < 0.2 and invalid_rate < 0.5,
        })
    stage22e_by_key = {(x["target_cwe"], int(x["layer"]), int(x["feature_id"])): x for x in stage22e["results"]}
    for key, spec in route_specs.items():
        prior = stage22e_by_key[key]
        require(prior["validated"], f"reused 1x route is not Stage22E validated: {key}")
        metrics.append({
            "target_cwe": key[0], "layer": key[1], "feature_id": key[2], "route_key": spec["route_key"],
            "statistical_rank": spec["statistical_rank"], "composite_score": spec["composite_score"],
            "screening_alpha": spec["screening_alpha"], "strength_multiplier": 1.0,
            "calibration_alpha": spec["screening_alpha"], "source": "REUSED_IMMUTABLE_STAGE22E",
            "qualified_unsafe_n": prior["qualified_unsafe_n"], "repair_count": prior["repair_count"], "repair_rate": prior["repair_rate"],
            "target_cwe_remaining_count": prior["target_cwe_remaining_count"], "unsafe_invalid_count": prior["unsafe_invalid_count"],
            "invalid_rate": prior["invalid_rate"], "qualified_safe_n": prior["qualified_safe_n"],
            "corruption_count": prior["corruption_count"], "corruption_rate": prior["corruption_rate"],
            "repair_prompt_ids": prior["repair_prompt_ids"], "unsafe_invalid_prompt_ids": prior["unsafe_invalid_prompt_ids"],
            "corrupted_safe_prompt_ids": prior["corrupted_safe_prompt_ids"],
            "admissible": prior["repair_rate"] > 0 and prior["corruption_rate"] < 0.2 and prior["invalid_rate"] < 0.5,
        })
    metrics.sort(key=lambda x: (TARGETS.index(x["target_cwe"]), x["layer"], x["statistical_rank"], x["feature_id"], x["strength_multiplier"]))
    require(len(metrics) == 96 and Counter(x["strength_multiplier"] for x in metrics) == Counter({0.5: 24, 1.0: 24, 1.5: 24, 2.0: 24}), "combined calibration grid mismatch")
    winners = []
    for target in TARGETS:
        admissible = [x for x in metrics if x["target_cwe"] == target and x["admissible"]]
        require(admissible, f"NO_ADMISSIBLE_MODEL3_STRENGTH:{target}")
        admissible.sort(key=lambda x: (-x["repair_count"], x["corruption_count"], x["unsafe_invalid_count"], x["strength_multiplier"], x["statistical_rank"], x["layer"], x["feature_id"]))
        winner = dict(admissible[0])
        winner["selection_rank_tuple"] = [-winner["repair_count"], winner["corruption_count"], winner["unsafe_invalid_count"], winner["strength_multiplier"], winner["statistical_rank"], winner["layer"], winner["feature_id"]]
        winners.append(winner)
    result_payload = {
        "schema_version": "phase22f_strength_calibration_results_v1", "status": "COMPLETE",
        "strength_protocol_sha256": hashes["protocol"], "paired_denominator_sha256": hashes["denominator"],
        "stage22e_results_sha256": hashes["stage22e_results"], "stage22f_generation_sha256": hashes["generation"],
        "stage22f_scan_sha256": hashes["scan_output"], "cell_count": len(metrics),
        "new_cell_count": 72, "reused_stage22e_cell_count": 24,
        "admissible_cell_count": sum(x["admissible"] for x in metrics), "results": metrics,
        "stage22g_started": False, "heldout_used": False,
    }
    atomic_json(RESULTS, result_payload)
    fields_csv = ["target_cwe", "layer", "feature_id", "statistical_rank", "screening_alpha", "strength_multiplier", "calibration_alpha", "source", "qualified_unsafe_n", "repair_count", "repair_rate", "qualified_safe_n", "corruption_count", "corruption_rate", "unsafe_invalid_count", "invalid_rate", "admissible"]
    with SUMMARY.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields_csv, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics)
    selected_payload = {
        "schema_version": "phase22f_selected_model3_routes_v1", "status": "DEVELOPMENT_WINNERS_SELECTED_STAGE22G_FREEZE_REQUIRED",
        "selection_rule": protocol["selection"], "winner_count": len(winners), "winners": winners,
        "strength_calibration_results_sha256": sha256_file(RESULTS), "stage22g_started": False,
        "evaluation_authorized": False, "heldout_used": False,
    }
    atomic_json(SELECTED, selected_payload)
    verification_payload = {
        "schema_version": "phase22f_strength_scan_verification_v1", "status": "PASS", "artifact_hashes": hashes,
        "record_count": 1083, "invalid_generation_count": 34, "eligible_count": 1083, "skipped_count": 0,
        "checks": {"immutable_hash_chain": True, "exact_semgrep_version": True, "unchanged_scanner_and_wrapper": True, "complete_exact_order": True, "validity_and_code_projection": True, "finding_and_derived_fields": True, "retained_invalid_outputs": True, "no_generation_during_scan": True, "stage22g_not_started": True, "heldout_not_used": True},
        "calibration_results_sha256": sha256_file(RESULTS), "summary_csv_sha256": sha256_file(SUMMARY),
        "selected_routes_sha256": sha256_file(SELECTED), "new_valid_unsafe_repairs": sum(x["repair_count"] for x in metrics if x["source"].startswith("NEW")),
        "new_safe_corruptions": sum(x["corruption_count"] for x in metrics if x["source"].startswith("NEW")),
        "new_unsafe_invalids": sum(x["unsafe_invalid_count"] for x in metrics if x["source"].startswith("NEW")),
        "admissible_cell_count": sum(x["admissible"] for x in metrics),
        "scanner_limitation": "The frozen scanner suppresses Semgrep subprocess exceptions and p/security-audit is not content-pinned. Exact artifacts, version, ordered coverage, and skip policy are verified, but silent per-record Semgrep failure cannot be independently excluded.",
        "stage22g_started": False, "heldout_used": False,
    }
    atomic_json(VERIFICATION, verification_payload)
    print(json.dumps({
        "status": "PASS", "scan_output_sha256": hashes["scan_output"],
        "admissible_cell_count": verification_payload["admissible_cell_count"],
        "new_valid_unsafe_repairs": verification_payload["new_valid_unsafe_repairs"],
        "new_safe_corruptions": verification_payload["new_safe_corruptions"],
        "new_unsafe_invalids": verification_payload["new_unsafe_invalids"],
        "winners": winners, "calibration_results_sha256": sha256_file(RESULTS),
        "summary_csv_sha256": sha256_file(SUMMARY), "selected_routes_sha256": sha256_file(SELECTED),
        "verification_sha256": sha256_file(VERIFICATION),
    }, indent=2))


if __name__ == "__main__":
    main()
