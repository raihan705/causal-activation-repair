#!/usr/bin/env python3
"""Validate Phase21 strength generations and build the immutable scan handoff."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
PHASE = ROOT / "revision/model2/phase21"
OUT = PHASE / "outputs"
SCRIPTS = PHASE / "scripts"
BUNDLE = PHASE / "model2_strength_scan_bundle"
ZIP_PATH = OUT / "model2_strength_scan_bundle.zip"

ALPHAS = (20, 40)
EXPECTED_OUTPUT_HASHES = {
    20: "c0afad33ec4b21d232d312561184fac4d6e4b82bda7349d53e0ac0ddf5508503",
    40: "ea8f93597768552135909d99862e3de2b8e354364ced3369f0e0c6b266cf5f84",
}
EXPECTED_MANIFEST_HASHES = {
    20: "098d8cb30eed4c112b715d4b3f84413726bbb819ee3df1a21d8104b8491fd777",
    40: "172bf1dcd3b465ba4fb06091d92d0afbcf1a2a43e34c671b52bbc059c84d1a60",
}
EXPECTED_AUX_HASHES = {
    "model2_strength_population_manifest.json": "ff3b76b306d1dcfdbfd208259a23cc35a4b8b6b44dfb582d3f8fb8fcdc84cc71",
    "model2_selected_routes.json": "0465330304be21d3c516fc30c12c51f1cb5786c807ce3c6ba17cfbf81399b9da",
    "model2_strength_selection_protocol.json": "8ef127425206dc04250ff79041266d00ed244a1f07844266829ee7f29ae9adf3",
    "model2_scanner_provenance.json": "e249982d929f13c7a0cda0211f73e4cd721331b98cf483e9b205f3c4936d64a9",
}
EXPECTED_SCANNER_HASH = "60cedd3d6ec2d19e5e7e5ba4d709230efdd5d9478047492fefda4dd88d9afbb7"
EXPECTED_RUNNER_HASH = "dfb82ec6ce8fa40361afde75f410d7f5ca47fc3a307d6c4f789fc37cb0dae127"
EXPECTED_ROUTE_MAP_HASH = "3c4d060906e863b037d66b033234c9a8911753c5dc029f124232be5c229e87f0"
EXPECTED_PROMPT_ORDER_HASH = "0e5056866ddcd9df381c4bb275d86c44dda1e1c0b054b59a987516173e05e000"
EXPECTED_ROUTES = {
    "CWE-120": {"layer": 20, "feature_id": 14471},
    "CWE-327": {"layer": 20, "feature_id": 529},
    "CWE-89": {"layer": 20, "feature_id": 12328},
}

FROZEN_SCANNER = ROOT / "phases/phase9/colab_scan_phase9.py"
SCAN_RUNNER = SCRIPTS / "scan_model2_strength.py"
VALIDATION = OUT / "model2_strength_generation_validation.json"
TRANSFER = OUT / "model2_strength_scan_manifest.json"


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
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def load_scanner(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("phase21_strength_frozen_scanner", path)
    require(spec is not None and spec.loader is not None, "cannot import frozen scanner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_generations() -> dict[str, Any]:
    for name, expected in EXPECTED_AUX_HASHES.items():
        path = OUT / name
        require(path.is_file(), f"missing frozen input: {name}")
        require(sha256_file(path) == expected, f"frozen input hash mismatch: {name}")
    require(FROZEN_SCANNER.is_file(), "missing frozen scanner")
    require(sha256_file(FROZEN_SCANNER) == EXPECTED_SCANNER_HASH, "frozen scanner hash mismatch")
    require(SCAN_RUNNER.is_file(), "missing strength scan runner")

    population = load_json(OUT / "model2_strength_population_manifest.json")
    selected = load_json(OUT / "model2_selected_routes.json")
    protocol = load_json(OUT / "model2_strength_selection_protocol.json")
    require(population["prompt_ids_source_order_canonical_sha256"] == EXPECTED_PROMPT_ORDER_HASH, "population order hash mismatch")
    require(selected["route_map_canonical_sha256"] == EXPECTED_ROUTE_MAP_HASH, "route map hash mismatch")
    require(selected["routes"] == EXPECTED_ROUTES, "selected routes changed")
    require(protocol["strength_candidates"] == [20, 40], "strength candidates changed")
    require(protocol["forbidden_rescue_alphas"] == [10, 30, 50, 60, 80], "rescue-alpha rule changed")
    require(protocol["generation"] == {
        "decode_generated_tokens_only": True,
        "do_sample": True,
        "max_new_tokens": 512,
        "seed": 42,
        "skip_special_tokens": True,
        "temperature": 0.2,
        "top_p": 0.95,
    }, "generation protocol changed")

    population_records = population["records"]
    require(len(population_records) == 180, "population count mismatch")
    prompt_ids = population["prompt_ids_source_order"]
    require(prompt_ids == [row["prompt_id"] for row in population_records], "population ID order mismatch")
    require(canonical_hash(prompt_ids) == EXPECTED_PROMPT_ORDER_HASH, "recomputed prompt order hash mismatch")
    population_by_id = {row["prompt_id"]: row for row in population_records}
    require(len(population_by_id) == 180, "duplicate population prompt IDs")

    conditions = []
    all_records: dict[int, list[dict[str, Any]]] = {}
    for alpha in ALPHAS:
        output = OUT / f"model2_alpha{alpha}_dev_outputs.json"
        manifest_path = OUT / f"model2_alpha{alpha}_generation_manifest.json"
        require(output.is_file() and manifest_path.is_file(), f"missing alpha{alpha} generation artifact")
        require(sha256_file(output) == EXPECTED_OUTPUT_HASHES[alpha], f"alpha{alpha} output hash mismatch")
        require(sha256_file(manifest_path) == EXPECTED_MANIFEST_HASHES[alpha], f"alpha{alpha} manifest hash mismatch")
        rows = load_json(output)
        manifest = load_json(manifest_path)
        require(isinstance(rows, list) and len(rows) == 180, f"alpha{alpha} record count mismatch")
        require(manifest["status"] == "COMPLETE", f"alpha{alpha} manifest is not complete")
        require(manifest["generation_output_sha256"] == EXPECTED_OUTPUT_HASHES[alpha], f"alpha{alpha} manifest/output binding mismatch")
        require(manifest["record_count"] == 180 and manifest["unique_condition_key_count"] == 180, f"alpha{alpha} manifest count mismatch")
        require(manifest["generation_failure_count"] == 0 and manifest["hook_failure_count"] == 0, f"alpha{alpha} generation/hook failure")
        require(manifest["invalid_count"] == 0 and manifest["empty_count"] == 0 and manifest["whitespace_only_count"] == 0, f"alpha{alpha} invalid generation")
        require(not manifest["duplicate_keys"] and manifest["unexpected_prompt_ids"] == [], f"alpha{alpha} identity failure")
        require(manifest["other_alphas_tested"] == [], f"alpha{alpha} records an unauthorized alpha")
        require(not manifest["b0_regenerated"] and not manifest["scanner_run"] and not manifest["heldout_accessed"] and not manifest["model1_modified"], f"alpha{alpha} scope violation")
        require([row["prompt_id"] for row in rows] == prompt_ids, f"alpha{alpha} prompt order mismatch")
        require(len({row["condition_key"] for row in rows}) == 180, f"alpha{alpha} duplicate condition keys")
        for index, row in enumerate(rows):
            source = population_records[index]
            route = EXPECTED_ROUTES[source["cwe_identifier"]]
            require(row["record_index"] == index, f"alpha{alpha} record index mismatch at {index}")
            require(row["condition_key"] == f"A{alpha}|{source['cwe_identifier']}|P{source['prompt_id']}", f"alpha{alpha} condition key mismatch at {index}")
            require(row["prompt_id"] == source["prompt_id"] and row["source_index"] == source["source_index"], f"alpha{alpha} source identity mismatch at {index}")
            require(row["source_cwe"] == source["cwe_identifier"] and row["language"] == source["language"], f"alpha{alpha} source metadata mismatch at {index}")
            require(row["layer"] == route["layer"] and row["feature_id"] == route["feature_id"], f"alpha{alpha} route mismatch at {index}")
            require(row["alpha"] == alpha and row["seed"] == 42, f"alpha{alpha} condition mismatch at {index}")
            require(row["source_prompt_sha256"] == source["source_prompt_sha256"], f"alpha{alpha} source hash mismatch at {index}")
            require(row["rendered_prompt_sha256"] == source["b0_rendered_prompt_sha256"], f"alpha{alpha} rendered-prompt mismatch at {index}")
            require(row["runner_sha256"] == EXPECTED_RUNNER_HASH, f"alpha{alpha} runner mismatch at {index}")
            require(row["generation_settings"] == {"temperature": 0.2, "top_p": 0.95, "max_new_tokens": 512, "do_sample": True}, f"alpha{alpha} generation settings mismatch at {index}")
            require(row["intervention_applied"] and row["hook_call_count"] > 0 and row["hook_finite"], f"alpha{alpha} intervention evidence failure at {index}")
            require(not row["generation_failure"] and not row["hook_failure"] and row["generation_status"] == "SUCCESS", f"alpha{alpha} execution failure at {index}")
            require(row["fallback_status"] == "NONE" and row["generation_error"] is None, f"alpha{alpha} fallback/error at {index}")
            require(row["validity"]["is_valid"] and bool(str(row["generated_code"]).strip()), f"alpha{alpha} invalid output at {index}")
        all_records[alpha] = rows
        conditions.append({
            "alpha": alpha,
            "generation_filename": output.name,
            "generation_sha256": sha256_file(output),
            "generation_manifest_filename": manifest_path.name,
            "generation_manifest_sha256": sha256_file(manifest_path),
            "expected_scan_filename": f"model2_alpha{alpha}_dev_scan.json",
            "record_count": 180,
        })

    unauthorized = sorted(
        path.name for path in OUT.glob("model2_alpha*_dev_outputs.json")
        if path.name not in {f"model2_alpha{alpha}_dev_outputs.json" for alpha in ALPHAS}
    )
    require(unauthorized == [], f"unauthorized strength outputs found: {unauthorized}")
    for index in range(180):
        require(all_records[20][index]["prompt_id"] == all_records[40][index]["prompt_id"], f"cross-alpha prompt mismatch at {index}")
        require(all_records[20][index]["source_cwe"] == all_records[40][index]["source_cwe"], f"cross-alpha source-CWE mismatch at {index}")
        require(all_records[20][index]["feature_id"] == all_records[40][index]["feature_id"], f"cross-alpha route mismatch at {index}")

    return {
        "schema_version": "phase21_model2_strength_generation_validation_v1",
        "status": "PASS",
        "errors": [],
        "conditions": conditions,
        "total_records": 360,
        "records_per_alpha": 180,
        "source_cwe_counts_per_alpha": dict(sorted(Counter(row["cwe_identifier"] for row in population_records).items())),
        "prompt_ids_source_order_sha256": EXPECTED_PROMPT_ORDER_HASH,
        "route_map_canonical_sha256": EXPECTED_ROUTE_MAP_HASH,
        "selected_routes": EXPECTED_ROUTES,
        "generation_failure_count": 0,
        "hook_failure_count": 0,
        "invalid_count": 0,
        "scanner_run": False,
        "metrics_computed": False,
        "heldout_accessed": False,
        "model1_modified": False,
    }


def deterministic_zip(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(source.iterdir(), key=lambda item: item.name):
            info = zipfile.ZipInfo(path.name, date_time=(2026, 9, 3, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())


def main() -> int:
    validation = validate_generations()
    write_json(VALIDATION, validation)

    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    BUNDLE.mkdir(parents=True)

    copies = [
        OUT / "model2_alpha20_dev_outputs.json",
        OUT / "model2_alpha40_dev_outputs.json",
        OUT / "model2_alpha20_generation_manifest.json",
        OUT / "model2_alpha40_generation_manifest.json",
        OUT / "model2_strength_population_manifest.json",
        OUT / "model2_selected_routes.json",
        OUT / "model2_strength_selection_protocol.json",
        OUT / "model2_scanner_provenance.json",
        VALIDATION,
        SCAN_RUNNER,
    ]
    for source in copies:
        shutil.copy2(source, BUNDLE / source.name)
    shutil.copy2(FROZEN_SCANNER, BUNDLE / "colab_scan_phase9_frozen.py")

    frozen = load_scanner(FROZEN_SCANNER)
    rule_hashes = {
        "rule_cwe_map_sha256": canonical_hash(frozen.RULE_CWE_MAP),
        "regex_patterns_sha256": canonical_hash(frozen.REGEX_PATTERNS),
        "target_languages_sha256": canonical_hash(sorted(frozen.TARGET_LANGUAGES)),
        "all_target_cwes_sha256": canonical_hash(sorted(frozen.ALL_TARGET_CWES)),
    }
    copied_names = [source.name for source in copies] + ["colab_scan_phase9_frozen.py"]
    immutable = [
        {"name": name, "sha256": sha256_file(BUNDLE / name), "size_bytes": (BUNDLE / name).stat().st_size}
        for name in copied_names
    ]
    manifest = {
        "schema_version": "phase21_model2_strength_scan_manifest_v1",
        "status": "FROZEN_WAITING_MODEL2_STRENGTH_SCAN",
        "required_semgrep_version": "1.175.0",
        "registry_config": "p/security-audit",
        "registry_snapshot_hash": "NOT_CONTENT_PINNED",
        "scan_order": [20, 40],
        "expected_total_records": 360,
        "records_per_alpha": 180,
        "prompt_ids_source_order": load_json(OUT / "model2_strength_population_manifest.json")["prompt_ids_source_order"],
        "prompt_ids_source_order_sha256": EXPECTED_PROMPT_ORDER_HASH,
        "conditions": validation["conditions"],
        "scanner_source": {"name": "colab_scan_phase9_frozen.py", "sha256": EXPECTED_SCANNER_HASH},
        "scanner_runner": {"name": SCAN_RUNNER.name, "sha256": sha256_file(SCAN_RUNNER)},
        "scanner_rule_hashes": rule_hashes,
        "immutable_files": immutable,
        "code_transformation": "NONE",
        "fail_closed": True,
        "resume_checkpoint_interval": 25,
        "expected_return_files": [
            "model2_alpha20_dev_scan.json",
            "model2_alpha40_dev_scan.json",
            "model2_strength_scan_return_manifest.json",
        ],
        "generation_run": False,
        "metrics_computed": False,
        "heldout_accessed": False,
        "model1_modified": False,
    }
    manifest["self_sha256_excluding_field"] = canonical_hash(manifest)
    write_json(TRANSFER, manifest)
    shutil.copy2(TRANSFER, BUNDLE / TRANSFER.name)
    write_json(BUNDLE / "README.json", {
        "status": "READY",
        "purpose": "Phase21 Model2 alpha20/alpha40 development strength scanning only",
        "command": "python scan_model2_strength.py --manifest model2_strength_scan_manifest.json --workdir . --all --resume",
        "return_files": manifest["expected_return_files"],
    })
    deterministic_zip(BUNDLE, ZIP_PATH)
    print(json.dumps({
        "status": "COMPLETE",
        "generation_validation_path": str(VALIDATION.relative_to(ROOT)).replace("\\", "/"),
        "generation_validation_sha256": sha256_file(VALIDATION),
        "bundle_path": str(BUNDLE.relative_to(ROOT)).replace("\\", "/"),
        "zip_path": str(ZIP_PATH.relative_to(ROOT)).replace("\\", "/"),
        "zip_sha256": sha256_file(ZIP_PATH),
        "transfer_manifest_sha256": sha256_file(TRANSFER),
        "scanner_runner_sha256": sha256_file(SCAN_RUNNER),
        "frozen_scanner_sha256": sha256_file(FROZEN_SCANNER),
        "total_records": 360,
        "return_files": manifest["expected_return_files"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"MODEL2_STRENGTH_BUNDLE_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
