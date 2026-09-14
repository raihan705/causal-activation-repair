#!/usr/bin/env python3
"""Fail-closed structural, integrity, privacy, and readability audit."""

from __future__ import annotations

import csv
import ast
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
HASH_FILE = ROOT / "manifests" / "SHA256SUMS"
MANIFEST_FILE = ROOT / "manifests" / "release_manifest.json"
IGNORED_IN_HASH_SET = {
    HASH_FILE.relative_to(ROOT).as_posix(),
    MANIFEST_FILE.relative_to(ROOT).as_posix(),
}
FORBIDDEN_SUFFIXES = {
    ".db",
    ".sqlite",
    ".sql",
    ".pt",
    ".pth",
    ".safetensors",
    ".bin",
    ".npz",
    ".npy",
    ".zip",
    ".7z",
}
SECRET_PATTERNS = {
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "GitHub classic token": re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    "GitHub fine-grained token": re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    "private Windows user path": re.compile(r"[A-Za-z]:\\Users\\", re.IGNORECASE),
}
FORBIDDEN_PUBLIC_BASENAMES = {
    "pasted-text.txt",
    "source_to_release_map.csv",
    "status.csv",
    "work.md",
}
SOURCE_BEARING_JSON_FIELDS = {
    "code_text",
    "completion_text",
    "fixed_code",
    "generated_code",
    "prompt_text",
    "raw_prompt",
    "source_prompt",
    "vulnerable_code",
}
SOURCE_BEARING_CSV_FIELDS = SOURCE_BEARING_JSON_FIELDS
TEXT_SUFFIXES = {".py", ".md", ".json", ".csv", ".yaml", ".yml", ".txt", ".cff", ".sh", ".ps1"}
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
REQUIRED_PATHS = {
    ".gitignore",
    "CITATION.cff",
    "LICENSE",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "docs/data_and_code_availability_statement.md",
    "manifests/excluded_materials.csv",
    "results/paper_figures/fig_bstar_per_cwe_profile.pdf",
    "results/paper_figures/fig_bstar_per_cwe_profile.png",
    "results/paper_tables/fig_bstar_per_cwe_profile_data.csv",
}
LOCAL_IMPORT_PREFIXES = (
    "build_",
    "caa_",
    "compute_",
    "finalize_",
    "phase",
    "preflight_",
    "routewise_",
    "run_",
    "select_",
    "summarize_",
    "validate_",
)
MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def find_source_bearing_json_fields(value: object, prefix: str = "$") -> list[str]:
    findings: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}"
            if key in SOURCE_BEARING_JSON_FIELDS and child not in (None, "", []):
                findings.append(child_prefix)
            findings.extend(find_source_bearing_json_fields(child, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(find_source_bearing_json_fields(child, f"{prefix}[{index}]"))
    return findings


def release_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and not IGNORED_DIRECTORY_NAMES.intersection(path.relative_to(ROOT).parts)
    )


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []
    files = release_files()
    relative_files = {path.relative_to(ROOT).as_posix() for path in files}

    for relative in sorted(REQUIRED_PATHS - relative_files):
        fail(errors, f"missing required public file: {relative}")

    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        if path.name.lower() in FORBIDDEN_PUBLIC_BASENAMES:
            fail(errors, f"private workflow file is not distributable: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            fail(errors, f"forbidden large/restricted type: {relative}")
        if path.stat().st_size > 20 * 1024 * 1024:
            fail(errors, f"file exceeds 20 MiB release ceiling: {relative}")

        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {"LICENSE", ".gitignore"}:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                fail(errors, f"non-UTF-8 text file: {relative}")
                continue
            for label, pattern in SECRET_PATTERNS.items():
                if pattern.search(text):
                    fail(errors, f"{label} detected in {relative}")

        if path.suffix.lower() == ".json":
            try:
                parsed_json = json.loads(path.read_text(encoding="utf-8"))
                for field_path in find_source_bearing_json_fields(parsed_json):
                    fail(errors, f"source-bearing JSON field in {relative}: {field_path}")
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                fail(errors, f"invalid JSON {relative}: {exc}")
        elif path.suffix.lower() == ".csv":
            try:
                with path.open("r", encoding="utf-8-sig", newline="") as handle:
                    header = next(csv.reader(handle), None) or []
                    for field in sorted(SOURCE_BEARING_CSV_FIELDS.intersection(header)):
                        fail(errors, f"source-bearing CSV field in {relative}: {field}")
            except (UnicodeDecodeError, csv.Error) as exc:
                fail(errors, f"invalid CSV {relative}: {exc}")
        elif path.suffix.lower() == ".py":
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=relative)
            except (UnicodeDecodeError, SyntaxError) as exc:
                fail(errors, f"invalid Python syntax {relative}: {exc}")

        if path.suffix.lower() == ".md":
            text = path.read_text(encoding="utf-8")
            for target in MARKDOWN_LINK.findall(text):
                target = target.strip().strip("<>")
                if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                    continue
                target = unquote(target.split("#", 1)[0])
                if target and not (path.parent / target).resolve().exists():
                    fail(errors, f"broken local Markdown link in {relative}: {target}")

    local_modules = {path.stem for path in files if path.suffix.lower() == ".py"}
    for path in (path for path in files if path.suffix.lower() == ".py"):
        relative = path.relative_to(ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.append(node.module.split(".", 1)[0])
        for module in imported:
            if module.startswith(LOCAL_IMPORT_PREFIXES) and module not in local_modules:
                fail(errors, f"missing local Python module imported by {relative}: {module}.py")

    if not HASH_FILE.is_file():
        fail(errors, "missing manifests/SHA256SUMS")
    else:
        expected: dict[str, str] = {}
        for line_number, line in enumerate(HASH_FILE.read_text(encoding="utf-8").splitlines(), 1):
            try:
                digest, relative = line.split("  ", 1)
            except ValueError:
                fail(errors, f"malformed SHA256SUMS line {line_number}")
                continue
            if relative in expected:
                fail(errors, f"duplicate SHA256SUMS path on line {line_number}: {relative}")
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                fail(errors, f"invalid SHA-256 digest on line {line_number}: {digest}")
            expected[relative] = digest

        actual_paths = {
            path.relative_to(ROOT).as_posix()
            for path in files
            if path.relative_to(ROOT).as_posix() not in IGNORED_IN_HASH_SET
        }
        if actual_paths != set(expected):
            for relative in sorted(actual_paths - set(expected)):
                fail(errors, f"unmanifested file: {relative}")
            for relative in sorted(set(expected) - actual_paths):
                fail(errors, f"missing manifested file: {relative}")
        for relative in sorted(actual_paths.intersection(expected)):
            observed = sha256_file(ROOT / relative)
            if observed != expected[relative]:
                fail(errors, f"hash mismatch: {relative}")

    if not MANIFEST_FILE.is_file():
        fail(errors, "missing manifests/release_manifest.json")
    else:
        try:
            manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
            entries = manifest["files"]
            manifest_paths = [entry["path"] for entry in entries]
            manifest_by_path = {entry["path"]: entry for entry in entries}
            if len(manifest_paths) != len(manifest_by_path):
                fail(errors, "duplicate path in manifests/release_manifest.json")
            actual_paths = relative_files - IGNORED_IN_HASH_SET
            if set(manifest_paths) != actual_paths:
                for relative in sorted(actual_paths - set(manifest_paths)):
                    fail(errors, f"file absent from release_manifest.json: {relative}")
                for relative in sorted(set(manifest_paths) - actual_paths):
                    fail(errors, f"stale release_manifest.json path: {relative}")
            for relative in sorted(actual_paths.intersection(manifest_by_path)):
                entry = manifest_by_path[relative]
                observed_sha = sha256_file(ROOT / relative)
                observed_size = (ROOT / relative).stat().st_size
                if entry.get("sha256") != observed_sha:
                    fail(errors, f"release_manifest.json hash mismatch: {relative}")
                if entry.get("size_bytes") != observed_size:
                    fail(errors, f"release_manifest.json size mismatch: {relative}")
            if manifest.get("file_count_excluding_generated_manifests") != len(actual_paths):
                fail(errors, "release_manifest.json file count is incorrect")
            if manifest.get("total_bytes_excluding_generated_manifests") != sum(
                (ROOT / relative).stat().st_size for relative in actual_paths
            ):
                fail(errors, "release_manifest.json byte total is incorrect")
        except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            fail(errors, f"invalid release manifest structure: {exc}")

    # Check the canonical values used for manuscript reporting. These checks
    # complement the file-integrity audit by detecting denominator or summary
    # drift across the compact public artifacts.
    try:
        scanner_audit = json.loads(
            (ROOT / "results" / "model1" / "scanner_audit_summary.json").read_text(encoding="utf-8")
        )
        confusion = scanner_audit["confusion_matrix"]
        if confusion != {
            "true_positive": 20,
            "false_positive": 4,
            "false_negative": 5,
            "true_negative": 21,
        }:
            fail(errors, "canonical scanner confusion matrix has changed")
        if not math.isclose(scanner_audit["false_positive_rate"]["value"], 0.16):
            fail(errors, "canonical scanner false-positive rate is not 0.16")

        with (ROOT / "results" / "model1" / "development_method_selection.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            development = {row["condition"]: row for row in csv.DictReader(handle)}
        development_checks = {
            "B1": (25, 0.416667, 48, 0.2, 13, 1341),
            "B2-a20": (36, 0.6, 27, 0.55, 5, 1306),
            "B2-a40": (36, 0.6, 27, 0.55, 3, 1315),
        }
        for condition, expected in development_checks.items():
            row = development[condition]
            observed = (
                int(row["paired_repairs"]),
                float(row["corrvrr"]),
                int(row["aggregate_scanner_positive"]),
                float(row["raw_vrr"]),
                int(row["corruptions"]),
                int(row["valid_outputs"]),
            )
            if observed[:1] + observed[2:3] + observed[4:] != expected[:1] + expected[2:3] + expected[4:]:
                fail(errors, f"canonical development counts changed for {condition}")
            if not math.isclose(observed[1], expected[1], abs_tol=1e-6) or not math.isclose(
                observed[3], expected[3], abs_tol=1e-6
            ):
                fail(errors, f"canonical development rates changed for {condition}")
            if int(row["b0_positive_denominator"]) != 60 or int(
                row["b0_safe_scanner_eligible_denominator"]
            ) != 863:
                fail(errors, f"canonical development denominators changed for {condition}")

        with (ROOT / "results" / "model1" / "phase16_aggregate_results.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            heldout = {row["method"]: row for row in csv.DictReader(handle)}
        heldout_checks = {
            "B1": (0.514676, 0.245484, 0.022090, 1.0),
            "B*": (0.551057, 0.504259, 0.004594, 0.982609),
        }
        for condition, expected in heldout_checks.items():
            observed = heldout[condition]
            values = tuple(
                float(observed[key])
                for key in ("corrvrr_mean", "raw_vrr_mean", "corruption_rate_mean", "validity_rate_mean")
            )
            if any(not math.isclose(a, b, abs_tol=1e-6) for a, b in zip(values, expected)):
                fail(errors, f"canonical held-out rates changed for {condition}")

        with (ROOT / "results" / "model1" / "phase16_per_seed_results.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            per_seed = {
                (row["method"], int(row["seed"])): row
                for row in csv.DictReader(handle)
            }
        seed42_checks = {
            "B0": (28, 0, 0, 575),
            "B1": (25, 13, 10, 575),
            "B*": (15, 15, 2, 575),
        }
        for condition, expected in seed42_checks.items():
            row = per_seed[(condition, 42)]
            observed = tuple(
                int(row[key])
                for key in ("raw_vulnerable_count", "repair_count", "corruption_count", "validity_count")
            )
            if observed != expected or int(row["b0_vulnerable_denominator"]) != 28:
                fail(errors, f"canonical seed-42 counts changed for {condition}")

        with (ROOT / "results" / "model1" / "causal_revalidation_summary.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            causal_rows = list(csv.DictReader(handle))
        causal_total = next(row for row in causal_rows if row["cwe_id"] == "ALL")
        if (
            int(causal_total["repair_denominator"]) != 29
            or int(causal_total["repairs"]) != 29
            or int(causal_total["valid_safe_controls"]) != 16
            or int(causal_total["target_findings_among_valid_safe_controls"]) != 0
        ):
            fail(errors, "canonical Model 1 causal-revalidation totals have changed")

        with (ROOT / "results" / "model2" / "model2_strength_results.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            gemma = {int(row["alpha"]): row for row in csv.DictReader(handle)}
        for alpha, expected in {20: (8, 12, 0.32, 0.07741935483870968), 40: (7, 10, 0.28, 0.06451612903225806)}.items():
            row = gemma[alpha]
            if int(row["repair_count"]) != expected[0] or int(row["corruption_count"]) != expected[1]:
                fail(errors, f"canonical Gemma counts changed for alpha {alpha}")
            if not math.isclose(float(row["corrvrr"]), expected[2]) or not math.isclose(
                float(row["corruption_rate"]), expected[3]
            ):
                fail(errors, f"canonical Gemma rates changed for alpha {alpha}")

        with (ROOT / "results" / "model2" / "model2_causal_feature_results.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            model2_feature_results = list(csv.DictReader(handle))
        model2_feature_validated = [
            row for row in model2_feature_results if row["validation_status"] == "VALIDATED"
        ]
        model2_validated_by_cwe = {
            cwe: sum(row["target_cwe"] == cwe for row in model2_feature_validated)
            for cwe in ("CWE-120", "CWE-327", "CWE-89")
        }
        if (
            len(model2_feature_results) != 90
            or len(model2_feature_validated) != 80
            or model2_validated_by_cwe != {"CWE-120": 30, "CWE-327": 28, "CWE-89": 22}
        ):
            fail(errors, "canonical Gemma causal-screen counts have changed")

        with (ROOT / "results" / "model2" / "model2_causal_cross_cwe_summary.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            model2_causal = list(csv.DictReader(handle))
        model2_validated = [row for row in model2_causal if row["validation_status"] == "VALIDATED"]
        validated_routes: dict[tuple[str, str], int] = {}
        for row in model2_validated:
            route = (row["layer"], row["feature_id"])
            validated_routes[route] = validated_routes.get(route, 0) + 1
        if (
            len(model2_causal) != 48
            or len(model2_validated) != 40
            or len(validated_routes) != 21
            or sum(count >= 2 for count in validated_routes.values()) != 17
            or sum(count == 3 for count in validated_routes.values()) != 2
        ):
            fail(errors, "canonical Gemma cross-CWE recurrence counts have changed")

        with (ROOT / "results" / "model3" / "phase22e_causal_candidate_summary.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            qwen_causal = list(csv.DictReader(handle))
        qwen_validated = [row for row in qwen_causal if row["validated"].lower() == "true"]
        qwen_validated_by_cwe = {
            cwe: sum(row["target_cwe"] == cwe for row in qwen_validated)
            for cwe in ("CWE-120", "CWE-327", "CWE-89")
        }
        if (
            len(qwen_causal) != 90
            or len(qwen_validated) != 24
            or qwen_validated_by_cwe != {"CWE-120": 15, "CWE-327": 1, "CWE-89": 8}
        ):
            fail(errors, "canonical Qwen causal-replication counts have changed")

        with (ROOT / "results" / "model3" / "phase22h_metrics_summary.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            qwen_rows = list(csv.DictReader(handle))
        qwen = next(row for row in qwen_rows if row["seed"] == "ALL_SEEDS" and row["scope"] == "OVERALL_MICRO")
        qwen_raw_vrr = (int(qwen["b0_any_finding_count"]) - int(qwen["routed_any_finding_count"])) / int(
            qwen["b0_any_finding_count"]
        )
        if (
            int(qwen["qualified_unsafe_n"]) != 37
            or int(qwen["repair_count"]) != 10
            or int(qwen["qualified_safe_n"]) != 197
            or int(qwen["corruption_count"]) != 3
            or not math.isclose(float(qwen["corrvrr"]), 10 / 37)
            or not math.isclose(qwen_raw_vrr, 7 / 37)
        ):
            fail(errors, "canonical Qwen pooled result has changed")

        with (ROOT / "results" / "model1" / "phase18_method_latency.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            latency = {
                (row["method"], row["repetition"]): row
                for row in csv.DictReader(handle)
            }
        b0_latency = float(latency[("B0", "ALL")]["mean_latency_seconds"])
        bstar_latency = float(latency[("B*", "ALL")]["mean_latency_seconds"])
        if not math.isclose(b0_latency, 34.57330776865749) or not math.isclose(
            bstar_latency, 37.10695766733338
        ):
            fail(errors, "canonical Model 1 generation latency has changed")
        if not math.isclose((bstar_latency / b0_latency - 1) * 100, 7.328341030107555):
            fail(errors, "canonical B* latency overhead has changed")

        with (ROOT / "results" / "model1" / "phase18_scanner_time.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            scanner_times = [float(row["wall_seconds"]) for row in csv.DictReader(handle)]
        if len(scanner_times) != 3 or not math.isclose(sum(scanner_times) / 3, 298.50344999):
            fail(errors, "canonical scanner timing has changed")

        with (ROOT / "results" / "model1" / "phase18_bandit_overhead.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            bandit_rows = list(csv.DictReader(handle))
        bandit_all = next(row for row in bandit_rows if row["repetition"] == "ALL")
        if int(bandit_all["decision_count"]) != 150 or not math.isclose(
            float(bandit_all["mean_latency_seconds"]), 5.813133333333334e-05
        ):
            fail(errors, "canonical bandit-decision timing has changed")

        with (ROOT / "results" / "model1" / "sae_reconstruction_summary.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            reconstruction = list(csv.DictReader(handle))
        layer_viability = {
            int(layer): {row["viable"] for row in reconstruction if int(row["layer"]) == layer}
            for layer in (12, 16, 19, 23)
        }
        if layer_viability != {12: {"False"}, 16: {"True"}, 19: {"True"}, 23: {"True"}}:
            fail(errors, "canonical Model 1 SAE layer viability has changed")

        with (ROOT / "results" / "model1" / "layer_sensitivity_metrics.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            sensitivity = {int(row["layer"]): row for row in csv.DictReader(handle)}
        expected_sensitivity = {
            16: (0.858125, 2.276562),
            19: (0.910195, 1.903906),
            23: (1.012682, 1.730469),
        }
        for layer, expected in expected_sensitivity.items():
            observed = sensitivity[layer]
            if not math.isclose(float(observed["mean_logit_shift"]), expected[0]) or not math.isclose(
                float(observed["max_logit_shift"]), expected[1]
            ):
                fail(errors, f"canonical layer-sensitivity values changed for layer {layer}")

        intervention_boundary = json.loads(
            (ROOT / "results" / "model1" / "intervention_boundary_summary.json").read_text(
                encoding="utf-8"
            )
        )
        if (
            intervention_boundary["group_intervention"]["bstar"]["absent_or_whitespace_outputs"] != 26
            or intervention_boundary["group_intervention"]["b3_ungated"]["absent_or_whitespace_outputs"] != 68
            or intervention_boundary["plausibility_gate"]["generation_steps"] != 182965
            or intervention_boundary["plausibility_gate"]["nonzero_steering_steps"] != 0
            or not math.isclose(
                intervention_boundary["layer_placement"]["bstar_layer_sensitive"]["corrvrr"], 0.60
            )
            or not math.isclose(
                intervention_boundary["layer_placement"]["fixed_layer_19"]["corrvrr"], 0.533333
            )
        ):
            fail(errors, "canonical intervention-boundary evidence has changed")

        with (ROOT / "results" / "paper_tables" / "fig_bstar_per_cwe_profile_data.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            figure5 = {
                (row["split"], row["cwe"]): (row["B0"], row["BSTAR"])
                for row in csv.DictReader(handle)
            }
        expected_figure5 = {
            ("dev", "CWE-120"): ("22.0", "6.0"),
            ("dev", "CWE-327"): ("29.0", "18.0"),
            ("dev", "CWE-89"): ("9.0", "0.0"),
            ("dev", "CWE-338"): ("7.0", "1.0"),
            ("test", "CWE-120"): ("14.0", "3.0"),
            ("test", "CWE-327"): ("12.0", "10.0"),
            ("test", "CWE-89"): ("2.0", "0.0"),
            ("test", "CWE-338"): ("", ""),
        }
        if figure5 != expected_figure5:
            fail(errors, "canonical Figure 5 source data have changed")

        with (ROOT / "results" / "model1" / "reduced_routewise_utility_results.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            routewise_utility = {
                (row["benchmark"], row["condition"], row["route_cwe_id"]): int(row["success_count"])
                for row in csv.DictReader(handle)
            }
        routewise_checks = {
            ("HumanEval", "B0", ""): 19,
            ("BigCodeBench", "B0", ""): 146,
            ("MMLU", "B0", ""): 241,
            ("HumanEval", "B*_SINGLE_ROUTE_ACTIVE", "CWE-125"): 6,
            ("BigCodeBench", "B*_SINGLE_ROUTE_ACTIVE", "CWE-89"): 125,
            ("MMLU", "B*_SINGLE_ROUTE_ACTIVE", "CWE-327"): 237,
        }
        if any(routewise_utility.get(key) != expected for key, expected in routewise_checks.items()):
            fail(errors, "canonical routewise utility results have changed")

        with (ROOT / "results" / "model1" / "phase17_utility_results.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            alwayson_utility = {
                (row["benchmark"], row["condition"]): int(row["success_count"])
                for row in csv.DictReader(handle)
            }
        alwayson_checks = {
            ("HumanEval", "B0"): 14,
            ("BigCodeBench", "B0"): 1127,
            ("MMLU", "B0"): 234,
            ("HumanEval", "B*-AlwaysOn-L19"): 0,
            ("BigCodeBench", "B*-AlwaysOn-L19"): 0,
            ("MMLU", "B*-AlwaysOn-L19"): 0,
        }
        if any(alwayson_utility.get(key) != expected for key, expected in alwayson_checks.items()):
            fail(errors, "canonical AlwaysOn utility results have changed")

        construct_review = json.loads(
            (ROOT / "results" / "construct_review_summary.json").read_text(encoding="utf-8")
        )
        model1_review = construct_review["model1_seed42_bstar_changed_cases"]
        model2_review = construct_review["model2_static_strength_repair_transitions"]
        if (
            model1_review["transition_count"] != 17
            or model1_review["scanner_repair_transitions"]["count"] != 15
            or model1_review["scanner_repair_transitions"]["genuine_security_improvement"] != 1
            or model2_review["transition_count"] != 15
            or model2_review["true_repair"] != 4
            or model2_review["partial_repair"] != 6
        ):
            fail(errors, "canonical changed-case aggregate counts have changed")

        dataset_summary = json.loads(
            (ROOT / "data" / "dataset_construction_summary.json").read_text(encoding="utf-8")
        )
        construction = dataset_summary["construction_counts"]
        if (
            construction["filtered_language_length_deduplicated_pairs"] != 8229
            or construction["target_cwe_scope_raw_pairs"] != 2722
            or construction["balanced_working_set_total"] != 843
        ):
            fail(errors, "canonical dataset-construction counts have changed")
    except (KeyError, StopIteration, TypeError, ValueError, OSError, csv.Error, json.JSONDecodeError) as exc:
        fail(errors, f"canonical result-consistency audit could not complete: {exc}")

    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8") if (ROOT / "LICENSE").is_file() else ""
    availability_text = (
        (ROOT / "docs" / "data_and_code_availability_statement.md").read_text(encoding="utf-8")
        if (ROOT / "docs" / "data_and_code_availability_statement.md").is_file()
        else ""
    )
    if "LICENSE NOT YET SELECTED" in license_text:
        warnings.append("select the code license before public publication")
    if "[GITHUB_URL]" in availability_text or "[ARCHIVE_DOI]" in availability_text:
        warnings.append("replace repository/archive placeholders after publication")

    if errors:
        print("RELEASE VALIDATION: FAIL")
        for error in errors:
            print(f"- {error}")
        return 1

    total_bytes = sum(path.stat().st_size for path in files)
    print(f"RELEASE VALIDATION: PASS ({len(files)} files, {total_bytes / 1024 / 1024:.2f} MiB)")
    for warning in warnings:
        print(f"PUBLICATION ACTION: {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
