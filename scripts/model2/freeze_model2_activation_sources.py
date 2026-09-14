#!/usr/bin/env python
"""Freeze exact development-only sources for Model2 layer viability."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "revision/model2/phase21/outputs"
TRAIN = ROOT / "data/processed/train_pairs.csv"
VAL = ROOT / "data/processed/val_pairs.csv"
B0 = OUT / "model2_b0_dev_outputs.json"
SCAN = OUT / "model2_b0_dev_scan.json"
SCAN_RETURN = OUT / "model2_b0_dev_scan_return_manifest.json"
SCANNER = ROOT / "phases/phase9/colab_scan_phase9.py"
RUNNER = Path(__file__).with_name("extract_model2_source_activations.py")
SOURCE_MANIFEST = OUT / "model2_activation_source_manifest.json"
SCANNER_PROVENANCE = OUT / "model2_scanner_provenance.json"
PLANNED = ("CWE-120", "CWE-327", "CWE-89")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    def normalize(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: normalize(val) for key, val in item.items()}
        if isinstance(item, (set, frozenset)):
            return sorted(normalize(val) for val in item)
        if isinstance(item, (list, tuple)):
            return [normalize(val) for val in item]
        return item

    payload = json.dumps(normalize(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != value:
            raise RuntimeError(f"frozen output already exists with different content: {path}")
        return
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def scanner_literal(name: str) -> Any:
    tree = ast.parse(SCANNER.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError(f"scanner literal not found: {name}")


def main() -> None:
    if not RUNNER.is_file():
        raise RuntimeError("activation extraction runner missing")
    train = read_csv(TRAIN)
    val = read_csv(VAL)
    pairs = []
    for split, rows in (("train", train), ("validation", val)):
        for split_index, row in enumerate(rows):
            if row["cwe_id"] in PLANNED:
                if not row["vulnerable_code"].strip() or not row["fixed_code"].strip():
                    raise RuntimeError(f"empty paired code: {row['pair_id']}")
                pairs.append((split, split_index, row))
    pair_ids = [x[2]["pair_id"] for x in pairs]
    if len(pairs) != 128 or len(set(pair_ids)) != 128:
        raise RuntimeError("planned development CVE pair population mismatch")
    b0 = json.loads(B0.read_text(encoding="utf-8"))
    scan = json.loads(SCAN.read_text(encoding="utf-8"))
    scan_map = {int(x["prompt_id"]): x for x in scan}
    cyber = [x for x in b0 if x["cwe_id"] in PLANNED]
    if len(cyber) != 261 or len({int(x["prompt_id"]) for x in cyber}) != 261:
        raise RuntimeError("planned Model2 B0 source population mismatch")
    records = []
    source_order = 0
    for role, code_field in (("CVE_VULNERABLE", "vulnerable_code"), ("CVE_FIXED", "fixed_code")):
        for split, split_index, row in pairs:
            text = row[code_field]
            records.append({
                "source_order": source_order, "source": role, "source_split": split,
                "source_split_index": split_index, "record_id": row["pair_id"],
                "pair_id": row["pair_id"], "prompt_id": None, "cve_id": row["cve_id"],
                "cwe_id": row["cwe_id"], "language": row["language"],
                "text_field": code_field, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "scanner_eligible": None, "scanner_target_positive": None,
            })
            source_order += 1
    for row in cyber:
        pid = int(row["prompt_id"])
        scan_row = scan_map[pid]
        text = row["generated_code"]
        records.append({
            "source_order": source_order, "source": "MODEL2_B0", "source_split": "development",
            "source_split_index": int(row["source_index"]), "record_id": str(pid),
            "pair_id": None, "prompt_id": pid, "cve_id": None,
            "cwe_id": row["cwe_id"], "language": row["language"],
            "text_field": "generated_code", "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "scanner_eligible": not scan_row["skipped"],
            "scanner_target_positive": row["cwe_id"] in set(scan_row["vulnerable_cwes"]),
        })
        source_order += 1
    expected_counts = {
        "CVE_VULNERABLE": {"CWE-120": 60, "CWE-327": 30, "CWE-89": 38},
        "CVE_FIXED": {"CWE-120": 60, "CWE-327": 30, "CWE-89": 38},
        "MODEL2_B0": {"CWE-120": 110, "CWE-327": 97, "CWE-89": 54},
    }
    actual_counts = {
        source: {
            cwe: sum(x["source"] == source and x["cwe_id"] == cwe for x in records)
            for cwe in PLANNED
        }
        for source in expected_counts
    }
    if actual_counts != expected_counts or len(records) != 517:
        raise RuntimeError(f"source count mismatch: {actual_counts}")
    source_manifest = {
        "schema_version": "phase21_model2_activation_source_manifest_v1",
        "status": "FROZEN_BEFORE_ACTIVATION_EXTRACTION",
        "planned_cwes": list(PLANNED), "record_count": len(records),
        "counts": actual_counts,
        "input_files": {
            "train_pairs": {"path": "data/processed/train_pairs.csv", "sha256": sha256_file(TRAIN), "total_rows": len(train)},
            "validation_pairs": {"path": "data/processed/val_pairs.csv", "sha256": sha256_file(VAL), "total_rows": len(val)},
            "model2_b0": {"path": "revision/model2/phase21/outputs/model2_b0_dev_outputs.json", "sha256": sha256_file(B0), "total_rows": len(b0)},
            "model2_scan": {"path": "revision/model2/phase21/outputs/model2_b0_dev_scan.json", "sha256": sha256_file(SCAN), "total_rows": len(scan)},
        },
        "tokenization": {"raw_code_text": True, "chat_template": False, "add_special_tokens": True, "truncation": True, "max_length": 512, "batch_size": 1, "selection": "last index whose attention_mask value is one"},
        "layers": [9, 20, 31], "hook_template": "blocks.<layer>.hook_resid_post",
        "runner": {"path": "revision/model2/phase21/scripts/extract_model2_source_activations.py", "sha256": sha256_file(RUNNER)},
        "records": records,
        "records_canonical_sha256": canonical_hash(records),
        "cve_test_used": False, "cyberseceval_heldout_used": False,
    }
    atomic_json(SOURCE_MANIFEST, source_manifest)
    returned = json.loads(SCAN_RETURN.read_text(encoding="utf-8"))
    rule_payload = {
        name: scanner_literal(name)
        for name in ("DERIVATION_CWES", "EXPLORATORY_CWES", "HELDOUT_CWES", "TARGET_LANGUAGES", "LANG_EXT", "RULE_CWE_MAP", "REGEX_PATTERNS")
    }
    scanner_provenance = {
        "schema_version": "phase21_model2_scanner_provenance_v1",
        "status": "FROZEN_MODEL2_ENVIRONMENT",
        "semgrep_version": returned["semgrep_version"],
        "platform": returned["platform"],
        "scanner_output_sha256": sha256_file(SCAN),
        "scanner_script_path": "phases/phase9/colab_scan_phase9.py",
        "scanner_script_sha256": sha256_file(SCANNER),
        "semgrep_registry_config": "p/security-audit",
        "semgrep_registry_snapshot_hash": "NOT_CONTENT_PINNED",
        "embedded_rule_payload_sha256": canonical_hash(rule_payload),
        "rule_component_sha256": {name: canonical_hash(value) for name, value in rule_payload.items()},
        "scanner_rules_changed": False,
        "model1_semgrep_runtime_byte_identical_claimed": False,
        "required_for_later_model2_scans": "Semgrep 1.175.0 and exact scanner script/rule hashes unless amended prospectively",
        "new_scan_run": False,
    }
    atomic_json(SCANNER_PROVENANCE, scanner_provenance)
    print(json.dumps({
        "source_manifest_sha256": sha256_file(SOURCE_MANIFEST),
        "scanner_provenance_sha256": sha256_file(SCANNER_PROVENANCE),
        "record_count": 517, "counts": actual_counts,
    }, indent=2))


if __name__ == "__main__":
    main()
