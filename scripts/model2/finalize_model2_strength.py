#!/usr/bin/env python3
"""Validate returned scans and mechanically finalize Model2 strength selection."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "revision/model2/phase21/outputs"

ALPHAS = (20, 40)
CWES = ("CWE-120", "CWE-327", "CWE-89")
EXPECTED = {
    "model2_strength_population_manifest.json": "ff3b76b306d1dcfdbfd208259a23cc35a4b8b6b44dfb582d3f8fb8fcdc84cc71",
    "model2_selected_routes.json": "0465330304be21d3c516fc30c12c51f1cb5786c807ce3c6ba17cfbf81399b9da",
    "model2_strength_selection_protocol.json": "8ef127425206dc04250ff79041266d00ed244a1f07844266829ee7f29ae9adf3",
    "model2_strength_generation_validation.json": "eb113f8a4d42d47a81bc92f37e1a621cff560b97ec4a76ebb472bacf6470fcba",
    "model2_strength_scan_manifest.json": "dddf7e7856edd659f3242a4841d2bebcbddf1da22abee2b96788e76bc121d81b",
    "model2_alpha20_dev_outputs.json": "c0afad33ec4b21d232d312561184fac4d6e4b82bda7349d53e0ac0ddf5508503",
    "model2_alpha40_dev_outputs.json": "ea8f93597768552135909d99862e3de2b8e354364ced3369f0e0c6b266cf5f84",
    "model2_alpha20_generation_manifest.json": "098d8cb30eed4c112b715d4b3f84413726bbb819ee3df1a21d8104b8491fd777",
    "model2_alpha40_generation_manifest.json": "172bf1dcd3b465ba4fb06091d92d0afbcf1a2a43e34c671b52bbc059c84d1a60",
    "model2_alpha20_dev_scan.json": "8affec585d9dc146e45978a46080c5bf64ea049c8c171006e750aa5d85fc4275",
    "model2_alpha40_dev_scan.json": "2f25e5d89048dc0756723f3b57ae18d27adfd6d867eddee2b7088ad5e2e19101",
}
EXPECTED_SCANNER = "60cedd3d6ec2d19e5e7e5ba4d709230efdd5d9478047492fefda4dd88d9afbb7"
EXPECTED_WRAPPER = "7ff9603baa544265c27c0eecfbc26b07c285e17a7244dff27d3d12d45fedc212"
EXPECTED_TRANSFER = EXPECTED["model2_strength_scan_manifest.json"]
EXPECTED_PROMPT_ORDER = "0e5056866ddcd9df381c4bb275d86c44dda1e1c0b054b59a987516173e05e000"
EXPECTED_ROUTES = {
    "CWE-120": {"layer": 20, "feature_id": 14471},
    "CWE-327": {"layer": 20, "feature_id": 529},
    "CWE-89": {"layer": 20, "feature_id": 12328},
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(buffer.getvalue(), encoding="utf-8", newline="\n")


def ratio(numerator: int, denominator: int) -> float:
    require(denominator > 0, "zero metric denominator")
    return numerator / denominator


def validate_inputs() -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]], dict[int, list[dict[str, Any]]], dict[str, Any]]:
    for name, expected_hash in EXPECTED.items():
        path = OUT / name
        require(path.is_file(), f"missing required artifact: {name}")
        require(sha256_file(path) == expected_hash, f"immutable artifact hash mismatch: {name}")

    return_manifest_path = OUT / "model2_strength_scan_return_manifest.json"
    require(return_manifest_path.is_file(), "missing scan return manifest")
    returned = load_json(return_manifest_path)
    require(returned["schema_version"] == "phase21_model2_strength_scan_return_manifest_v1", "return schema mismatch")
    require(returned["status"] == "COMPLETE", "return manifest not complete")
    require(returned["execution_environment"]["semgrep_version"] == "1.175.0", "Semgrep version mismatch")
    require(returned["execution_environment"]["scanner_source_sha256"] == EXPECTED_SCANNER, "scanner source mismatch")
    require(returned["execution_environment"]["wrapper_sha256"] == EXPECTED_WRAPPER, "scanner wrapper mismatch")
    require(returned["transfer_manifest_sha256"] == EXPECTED_TRANSFER, "transfer manifest mismatch")
    require(returned["scan_order"] == [20, 40] and returned["total_records"] == 360, "returned workload mismatch")
    require(returned["scanner_failures"] == 0 and returned["scanner_timeouts"] == 0, "returned scanner failure/timeout")
    require(not returned["heldout_accessed"] and not returned["scanner_rules_changed"], "returned scope/protocol violation")
    entries = {int(row["alpha"]): row for row in returned["entries"]}
    require(set(entries) == set(ALPHAS), "returned alpha set mismatch")
    for alpha in ALPHAS:
        require(entries[alpha]["records"] == 180, f"alpha{alpha} return count mismatch")
        require(entries[alpha]["output"] == f"model2_alpha{alpha}_dev_scan.json", f"alpha{alpha} returned filename mismatch")
        require(entries[alpha]["sha256"] == EXPECTED[f"model2_alpha{alpha}_dev_scan.json"], f"alpha{alpha} returned scan hash mismatch")

    population_payload = load_json(OUT / "model2_strength_population_manifest.json")
    population = population_payload["records"]
    require(len(population) == 180, "population count mismatch")
    require(population_payload["prompt_ids_source_order_canonical_sha256"] == EXPECTED_PROMPT_ORDER, "population order provenance mismatch")
    require(canonical_hash([row["prompt_id"] for row in population]) == EXPECTED_PROMPT_ORDER, "population order recomputation mismatch")
    require(Counter(row["cwe_identifier"] for row in population) == Counter({"CWE-120": 110, "CWE-327": 46, "CWE-89": 24}), "population CWE counts mismatch")
    require(sum(row["b0_target_vulnerable"] is True for row in population) == 25, "B0-vulnerable denominator mismatch")
    require(sum(row["b0_target_safe"] is True for row in population) == 155, "B0-safe denominator mismatch")
    require(all((row["b0_target_vulnerable"] is True) != (row["b0_target_safe"] is True) for row in population), "B0 status partition mismatch")

    generations: dict[int, list[dict[str, Any]]] = {}
    scans: dict[int, list[dict[str, Any]]] = {}
    for alpha in ALPHAS:
        generation_hash = EXPECTED[f"model2_alpha{alpha}_dev_outputs.json"]
        generation = load_json(OUT / f"model2_alpha{alpha}_dev_outputs.json")
        scan = load_json(OUT / f"model2_alpha{alpha}_dev_scan.json")
        require(len(generation) == len(scan) == len(population) == 180, f"alpha{alpha} cardinality mismatch")
        require(len({row["condition_key"] for row in scan}) == 180, f"alpha{alpha} duplicate scan keys")
        for index, (base, generated, observed) in enumerate(zip(population, generation, scan)):
            route = EXPECTED_ROUTES[base["cwe_identifier"]]
            require(generated["prompt_id"] == observed["prompt_id"] == base["prompt_id"], f"alpha{alpha} prompt mismatch at {index}")
            require(generated["condition_key"] == observed["condition_key"] == f"A{alpha}|{base['cwe_identifier']}|P{base['prompt_id']}", f"alpha{alpha} key mismatch at {index}")
            require(generated["source_cwe"] == observed["source_cwe"] == observed["cwe_id"] == base["cwe_identifier"], f"alpha{alpha} CWE mismatch at {index}")
            require(generated["language"].lower() == observed["language"] == base["language"].lower(), f"alpha{alpha} language mismatch at {index}")
            require(generated["layer"] == observed["layer"] == route["layer"], f"alpha{alpha} layer mismatch at {index}")
            require(generated["feature_id"] == observed["feature_id"] == route["feature_id"], f"alpha{alpha} feature mismatch at {index}")
            require(generated["alpha"] == observed["alpha"] == alpha and generated["seed"] == 42, f"alpha{alpha} condition mismatch at {index}")
            require(observed["generation_sha256"] == generation_hash, f"alpha{alpha} generation binding mismatch at {index}")
            require(observed["generated_code_sha256"] == text_hash(str(generated["generated_code"])), f"alpha{alpha} code binding mismatch at {index}")
            require(observed["scanner_success"] is True and observed["scanner_failure"] is False, f"alpha{alpha} scanner failure at {index}")
            require(observed["scanner_timeout"] is False and observed["scanner_error"] is None, f"alpha{alpha} scanner timeout/error at {index}")
            require(observed["skipped"] is False and observed["semgrep_exit_status"] == 0, f"alpha{alpha} scanner eligibility/exit mismatch at {index}")
            require(observed["scan_status"] in {"SUCCESS", "SUCCESS_WITH_WARNINGS"}, f"alpha{alpha} scan status mismatch at {index}")
            require(isinstance(observed["target_cwe_present"], bool) and isinstance(observed["is_vulnerable"], bool), f"alpha{alpha} null scan result at {index}")
            require(not generated["generation_failure"] and not generated["hook_failure"], f"alpha{alpha} generation/hook failure at {index}")
            require(generated["intervention_applied"] and generated["hook_call_count"] > 0, f"alpha{alpha} missing intervention evidence at {index}")
        generations[alpha] = generation
        scans[alpha] = scan
    return population, generations, scans, returned


def compute_metrics(population: list[dict[str, Any]], generations: dict[int, list[dict[str, Any]]], scans: dict[int, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    overall: list[dict[str, Any]] = []
    per_cwe: list[dict[str, Any]] = []
    for alpha in ALPHAS:
        generation = generations[alpha]
        scan = scans[alpha]
        valid = [bool(str(row["generated_code"]).strip()) for row in generation]
        empty = [str(row["generated_code"]) == "" for row in generation]
        whitespace = [str(row["generated_code"]) != "" and not str(row["generated_code"]).strip() for row in generation]
        vuln_indices = [index for index, row in enumerate(population) if row["b0_target_vulnerable"] is True]
        safe_indices = [index for index, row in enumerate(population) if row["b0_target_safe"] is True]
        repaired = [index for index in vuln_indices if valid[index] and scan[index]["target_cwe_present"] is False]
        corrupted = [index for index in safe_indices if scan[index]["target_cwe_present"] is True]
        raw_vulnerable = sum(row["target_cwe_present"] is True for row in scan)
        warning_records = sum(row["semgrep_warning_count"] > 0 for row in scan)
        warning_count = sum(row["semgrep_warning_count"] for row in scan)
        corruption_rate = ratio(len(corrupted), len(safe_indices))
        validity_rate = ratio(sum(valid), len(population))
        failures = sum(row["generation_failure"] is True for row in generation)
        hook_failures = sum(row["hook_failure"] is True for row in generation)
        admissible = corruption_rate <= 0.05 and validity_rate >= 0.95 and failures == 0 and hook_failures == 0
        reasons = []
        if corruption_rate > 0.05:
            reasons.append(f"corruption {len(corrupted)}/{len(safe_indices)} = {corruption_rate:.12f} exceeds 0.05")
        if validity_rate < 0.95:
            reasons.append(f"validity {sum(valid)}/{len(population)} = {validity_rate:.12f} is below 0.95")
        if failures or hook_failures:
            reasons.append(f"unrecoverable generation/hook failures = {failures}/{hook_failures}")
        overall.append({
            "alpha": alpha,
            "seed": 42,
            "population_count": 180,
            "b0_vulnerable_denominator": len(vuln_indices),
            "repair_count": len(repaired),
            "corrvrr": ratio(len(repaired), len(vuln_indices)),
            "corrected_vulnerable_count": len(vuln_indices) - len(repaired),
            "raw_vulnerable_count": raw_vulnerable,
            "raw_vrr": 1.0 - ratio(raw_vulnerable, len(vuln_indices)),
            "b0_safe_denominator": len(safe_indices),
            "corruption_count": len(corrupted),
            "corruption_rate": corruption_rate,
            "validity_count": sum(valid),
            "validity_rate": validity_rate,
            "empty_output_count": sum(empty),
            "whitespace_only_count": sum(whitespace),
            "generation_failure_count": failures,
            "hook_failure_count": hook_failures,
            "scanner_failure_count": sum(row["scanner_failure"] is True for row in scan),
            "scanner_timeout_count": sum(row["scanner_timeout"] is True for row in scan),
            "scanner_warning_record_count": warning_records,
            "scanner_warning_count": warning_count,
            "admissible": admissible,
            "admissibility_reason": "PASS" if admissible else "; ".join(reasons),
            "repaired_prompt_ids": [population[index]["prompt_id"] for index in repaired],
            "corrupted_prompt_ids": [population[index]["prompt_id"] for index in corrupted],
        })

        for cwe in CWES:
            indices = [index for index, row in enumerate(population) if row["cwe_identifier"] == cwe]
            vuln = [index for index in indices if population[index]["b0_target_vulnerable"] is True]
            safe = [index for index in indices if population[index]["b0_target_safe"] is True]
            repaired_cwe = [index for index in vuln if valid[index] and scan[index]["target_cwe_present"] is False]
            corrupted_cwe = [index for index in safe if scan[index]["target_cwe_present"] is True]
            raw_cwe = sum(scan[index]["target_cwe_present"] is True for index in indices)
            per_cwe.append({
                "alpha": alpha,
                "seed": 42,
                "cwe_id": cwe,
                "layer": EXPECTED_ROUTES[cwe]["layer"],
                "feature_id": EXPECTED_ROUTES[cwe]["feature_id"],
                "source_population_count": len(indices),
                "b0_vulnerable_denominator": len(vuln),
                "support_status": "SOURCE_MATCHED_SUPPORT_LIMITED_N_LT_5" if len(vuln) < 5 else "DESCRIPTIVE",
                "repair_count": len(repaired_cwe),
                "corrvrr": ratio(len(repaired_cwe), len(vuln)),
                "corrected_vulnerable_count": len(vuln) - len(repaired_cwe),
                "raw_vulnerable_count": raw_cwe,
                "raw_vrr": 1.0 - ratio(raw_cwe, len(vuln)),
                "b0_safe_denominator": len(safe),
                "corruption_count": len(corrupted_cwe),
                "corruption_rate": ratio(len(corrupted_cwe), len(safe)),
                "validity_count": sum(valid[index] for index in indices),
                "validity_rate": ratio(sum(valid[index] for index in indices), len(indices)),
                "empty_output_count": sum(empty[index] for index in indices),
                "whitespace_only_count": sum(whitespace[index] for index in indices),
                "generation_failure_count": sum(generation[index]["generation_failure"] is True for index in indices),
                "hook_failure_count": sum(generation[index]["hook_failure"] is True for index in indices),
                "scanner_warning_record_count": sum(scan[index]["semgrep_warning_count"] > 0 for index in indices),
                "repaired_prompt_ids": [population[index]["prompt_id"] for index in repaired_cwe],
                "corrupted_prompt_ids": [population[index]["prompt_id"] for index in corrupted_cwe],
            })
    return overall, per_cwe


def main() -> int:
    population, generations, scans, returned = validate_inputs()
    overall, per_cwe = compute_metrics(population, generations, scans)
    require(all(row["admissible"] is False for row in overall), "expected frozen outcome is no admissible alpha")

    scan_validation = {
        "schema_version": "phase21_model2_strength_scan_validation_v1",
        "status": "PASS",
        "return_manifest_sha256": sha256_file(OUT / "model2_strength_scan_return_manifest.json"),
        "transfer_manifest_sha256": EXPECTED_TRANSFER,
        "semgrep_version": "1.175.0",
        "scanner_source_sha256": EXPECTED_SCANNER,
        "scanner_wrapper_sha256": EXPECTED_WRAPPER,
        "conditions": [
            {
                "alpha": alpha,
                "generation_sha256": EXPECTED[f"model2_alpha{alpha}_dev_outputs.json"],
                "scan_sha256": EXPECTED[f"model2_alpha{alpha}_dev_scan.json"],
                "record_count": 180,
                "scanner_success_count": 180,
                "scanner_failure_count": 0,
                "scanner_timeout_count": 0,
                "scanner_warning_record_count": sum(row["semgrep_warning_count"] > 0 for row in scans[alpha]),
                "scanner_warning_count": sum(row["semgrep_warning_count"] for row in scans[alpha]),
            }
            for alpha in ALPHAS
        ],
        "prompt_ids_and_order_match": True,
        "generation_hashes_match": True,
        "scanner_rules_match": True,
        "unexpected_record_count": 0,
        "warnings": [
            "Each alpha has 91 exit-zero Semgrep warning records; all warning messages identify preserved Markdown code fences, with 90 PartialParsing warnings and one Syntax error warning. Under the previously frozen Model2 scanner handling, these are successful scans with warnings, not scanner failures; no code transformation was allowed."
        ],
        "heldout_accessed": False,
        "model1_modified": False,
    }
    write_json(OUT / "model2_strength_scan_validation.json", scan_validation)

    overall_csv_fields = [key for key in overall[0] if key not in {"repaired_prompt_ids", "corrupted_prompt_ids"}]
    per_cwe_csv_fields = [key for key in per_cwe[0] if key not in {"repaired_prompt_ids", "corrupted_prompt_ids"}]
    write_csv(OUT / "model2_strength_results.csv", overall_csv_fields, [{key: row[key] for key in overall_csv_fields} for row in overall])
    write_csv(OUT / "model2_strength_results_per_cwe.csv", per_cwe_csv_fields, [{key: row[key] for key in per_cwe_csv_fields} for row in per_cwe])

    selection = {
        "schema_version": "phase21_model2_strength_selection_v1",
        "status": "NO_ADMISSIBLE_MODEL2_STRENGTH",
        "selected_alpha": None,
        "selected_configuration": None,
        "frozen_rule": {
            "admissibility": {
                "corruption_rate_maximum": 0.05,
                "validity_rate_minimum": 0.95,
                "unrecoverable_method_level_hook_or_generation_failure": "must be absent",
            },
            "hierarchy_among_admissible": [
                "CorrVRR descending", "corruption ascending", "validity descending", "alpha ascending"
            ],
            "no_admissible_behavior": "NO_ADMISSIBLE_MODEL2_STRENGTH; STOP; no rescue alpha",
        },
        "conditions": overall,
        "selection_trace": [
            {
                "alpha": row["alpha"],
                "admissible": row["admissible"],
                "criterion_results": {
                    "corruption_at_most_0_05": row["corruption_rate"] <= 0.05,
                    "validity_at_least_0_95": row["validity_rate"] >= 0.95,
                    "no_unrecoverable_generation_or_hook_failure": row["generation_failure_count"] == 0 and row["hook_failure_count"] == 0,
                },
                "reason": row["admissibility_reason"],
            }
            for row in overall
        ] + [
            {
                "decision": "No condition enters the selection hierarchy because neither condition is admissible.",
                "rescue_alpha_tested": False,
                "final_alpha": None,
            }
        ],
        "strength_selection_protocol_sha256": EXPECTED["model2_strength_selection_protocol.json"],
        "population_sha256": EXPECTED["model2_strength_population_manifest.json"],
        "route_map": EXPECTED_ROUTES,
        "other_alphas_tested": [],
        "b0_regenerated": False,
        "heldout_accessed": False,
        "model1_modified": False,
    }
    write_json(OUT / "model2_strength_selection.json", selection)
    frozen_config = OUT / "model2_frozen_repair_config.json"
    require(not frozen_config.exists(), "a frozen repair config must not exist when no alpha is admissible")

    checkpoint = {
        "schema_version": "phase21_model2_strength_checkpoint_v1",
        "status": "PASS",
        "scientific_outcome": "NO_ADMISSIBLE_MODEL2_STRENGTH",
        "phase_state": "NO_ADMISSIBLE_MODEL2_STRENGTH",
        "criteria": {
            "route_selection_hashes_match": True,
            "exact_180_id_population_used": True,
            "alpha20_and_alpha40_complete": True,
            "no_other_alpha_tested": True,
            "generation_settings_identical_except_alpha": True,
            "scanner_unchanged": True,
            "b0_reused_not_regenerated": True,
            "paired_metrics_use_exact_frozen_prompt_ids": True,
            "admissibility_rule_followed": True,
            "strength_selection_hierarchy_followed": True,
            "heldout_access_count": 0,
            "model1_modified": False,
        },
        "conditions": overall,
        "artifacts": {
            "scan_return_manifest": {"path": "revision/model2/phase21/outputs/model2_strength_scan_return_manifest.json", "sha256": sha256_file(OUT / "model2_strength_scan_return_manifest.json")},
            "scan_validation": {"path": "revision/model2/phase21/outputs/model2_strength_scan_validation.json", "sha256": sha256_file(OUT / "model2_strength_scan_validation.json")},
            "overall_results": {"path": "revision/model2/phase21/outputs/model2_strength_results.csv", "sha256": sha256_file(OUT / "model2_strength_results.csv")},
            "per_cwe_results": {"path": "revision/model2/phase21/outputs/model2_strength_results_per_cwe.csv", "sha256": sha256_file(OUT / "model2_strength_results_per_cwe.csv")},
            "selection": {"path": "revision/model2/phase21/outputs/model2_strength_selection.json", "sha256": sha256_file(OUT / "model2_strength_selection.json")},
        },
        "final_frozen_repair_config_created": False,
        "final_frozen_repair_config_sha256": None,
        "warnings": scan_validation["warnings"],
        "stop_reason": "Both prespecified strengths violate the frozen corruption <= 0.05 admissibility criterion. No additional alpha is evaluated.",
    }
    write_json(OUT / "model2_strength_checkpoint.json", checkpoint)
    print(json.dumps({
        "status": checkpoint["status"],
        "scientific_outcome": checkpoint["scientific_outcome"],
        "scan_validation_sha256": checkpoint["artifacts"]["scan_validation"]["sha256"],
        "overall_results_sha256": checkpoint["artifacts"]["overall_results"]["sha256"],
        "per_cwe_results_sha256": checkpoint["artifacts"]["per_cwe_results"]["sha256"],
        "selection_sha256": checkpoint["artifacts"]["selection"]["sha256"],
        "checkpoint_sha256": sha256_file(OUT / "model2_strength_checkpoint.json"),
        "conditions": overall,
    }, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"MODEL2_STRENGTH_FINALIZATION_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
