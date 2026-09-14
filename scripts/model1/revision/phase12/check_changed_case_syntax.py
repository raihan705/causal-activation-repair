#!/usr/bin/env python3
"""Run only repository-provided syntax checks on the frozen Phase 12 sheet."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import os
import platform
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
AUDIT_PATH = ROOT / "revision/model1/phase12/outputs/changed_case_audit.csv"
AGREEMENT_PATH = ROOT / "revision/model1/phase12/outputs/changed_case_agreement.json"
OUTPUT_PATH = ROOT / "revision/model1/phase12/outputs/changed_case_syntax_compile.csv"

LABEL_COLUMNS = ["human_label_1", "human_label_2", "resolution_note", "final_label"]
FIELDNAMES = [
    "case_type",
    "prompt_id",
    "source_index",
    "condition",
    "language",
    "raw_sha256",
    "extraction_status",
    "extracted_sha256",
    "tool",
    "tool_version",
    "command",
    "status",
    "compile_status",
    "diagnostic",
]


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="")
    os.replace(temporary, path)


def phase10_extract(completion: str) -> tuple[str, str]:
    """Match phases/phase10/compute_phase10_utility_metrics.py lines 115-129."""
    completion = completion.strip()
    extraction_status = "RAW_TEXT_USED"
    if "```" in completion:
        lines = completion.split("\n")
        code_lines: list[str] = []
        in_block = False
        for line in lines:
            if line.startswith("```"):
                in_block = not in_block
                continue
            if in_block:
                code_lines.append(line)
        if code_lines:
            completion = "\n".join(code_lines)
            extraction_status = "PHASE10_FENCED_CODE_EXTRACTED"
    return completion, extraction_status


def python_result(raw: str) -> dict[str, str]:
    extracted, extraction_status = phase10_extract(raw)
    diagnostic = ""
    status = "PYTHON_AST_PARSE_PASS"
    try:
        ast.parse(extracted)
    except SyntaxError as exc:
        status = "PYTHON_AST_PARSE_FAIL"
        diagnostic = f"{exc.msg} (line {exc.lineno}, offset {exc.offset})"
    return {
        "raw_sha256": sha256_text(raw),
        "extraction_status": extraction_status,
        "extracted_sha256": sha256_text(extracted),
        "tool": "python_ast_parse",
        "tool_version": platform.python_version(),
        "command": "ast.parse(extracted_code)",
        "status": status,
        "compile_status": "NOT_RUN",
        "diagnostic": diagnostic,
    }


def unsupported_result(raw: str) -> dict[str, str]:
    return {
        "raw_sha256": sha256_text(raw),
        "extraction_status": "NOT_ATTEMPTED",
        "extracted_sha256": "",
        "tool": "",
        "tool_version": "",
        "command": "",
        "status": "UNSUPPORTED_NO_EXISTING_REPOSITORY_TOOL",
        "compile_status": "NOT_RUN",
        "diagnostic": "No repository-provided syntax tool was available; no compiler or package was invoked.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    if not AUDIT_PATH.is_file() or not AGREEMENT_PATH.is_file():
        fail("the frozen audit sheet and agreement artifact must exist")
    agreement = json.loads(AGREEMENT_PATH.read_text(encoding="utf-8"))
    if agreement.get("labelling_mode") != "SINGLE_HUMAN":
        fail("agreement does not freeze SINGLE_HUMAN mode")
    expected_sheet_hash = agreement.get("sheet", {}).get("sha256_pre_label")
    if sha256_path(AUDIT_PATH) != expected_sheet_hash:
        fail("audit sheet hash differs from the pre-label frozen agreement")

    with AUDIT_PATH.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 17:
        fail("audit sheet must contain exactly 17 cases")
    if sum(row.get("case_type") == "HELDOUT_CHANGED_CASE" for row in rows) != 16:
        fail("audit sheet must contain exactly 16 held-out changed cases")
    if sum(row.get("case_type") == "FIGURE6_VALIDATION" for row in rows) != 1:
        fail("audit sheet must contain exactly one Figure 6 case")
    if any(row.get(column, "") for row in rows for column in LABEL_COLUMNS):
        fail("human-label fields must remain blank during syntax checking")
    if any(row.get("labelling_mode") != "SINGLE_HUMAN" for row in rows):
        fail("audit-sheet rows do not uniformly freeze SINGLE_HUMAN mode")

    output_rows: list[dict[str, str]] = []
    for row in rows:
        if row["case_type"] == "HELDOUT_CHANGED_CASE":
            conditions = [
                ("B0", row["b0_output"]),
                ("CANONICAL_BSTAR", row["bstar_output"]),
            ]
        else:
            conditions = [
                ("B0", row["b0_output"]),
                ("SUBMITTED_PHASE5_B2_A40_UNSTEERED", row["bstar_output"]),
            ]

        language = row["language"].strip().lower()
        for condition, raw in conditions:
            if language == "python":
                check = python_result(raw)
            elif language in {"c", "cpp", "c++", "java"}:
                check = unsupported_result(raw)
            else:
                fail(f"unexpected language {row['language']!r} for prompt {row['prompt_id']}")
            output_rows.append({
                "case_type": row["case_type"],
                "prompt_id": row["prompt_id"],
                "source_index": row["source_index"],
                "condition": condition,
                "language": row["language"],
                **check,
            })

    if len(output_rows) != 34:
        fail("syntax artifact must contain two condition rows for each of 17 cases")

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(output_rows)
    atomic_write(OUTPUT_PATH, buffer.getvalue())

    by_status: dict[str, int] = {}
    for row in output_rows:
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1
    print("validation_status=SYNTAX_STAGE_COMPLETE")
    print("case_count=17")
    print("condition_row_count=34")
    for status in sorted(by_status):
        print(f"{status}={by_status[status]}")
    print(f"syntax_csv_sha256={sha256_path(OUTPUT_PATH)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1)
