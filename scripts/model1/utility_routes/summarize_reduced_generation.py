#!/usr/bin/env python3
"""Read-only-style diagnostics for completed reduced route-wise generation.

The script does not execute generated code.  It reports MMLU exact-match
accuracy, the prespecified BigCodeBench Python AST syntax proxy, completion
lengths, and paired outcome changes relative to the frozen B0 records.
"""

from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from routewise_utility_common import (
    OUTPUTS, baseline_output, baseline_run_manifest, read_json, require,
    sha256_path,
)
from run_reduced_routewise_utility import (
    load_reduced_protocol, output_path, run_manifest_path,
)


def validated(path: Path, manifest_path: Path, expected: int) -> list[dict[str, Any]]:
    require(path.is_file() and manifest_path.is_file(), f"Missing {path.name} or manifest")
    manifest = read_json(manifest_path)
    require(manifest.get("status") == "COMPLETE", f"{manifest_path.name} is not COMPLETE")
    require(manifest.get("output_sha256") == sha256_path(path), f"{path.name} hash mismatch")
    rows = read_json(path)
    require(isinstance(rows, list) and len(rows) == expected, f"{path.name} count mismatch")
    return rows


def extract_code(text: str) -> str:
    code = text.strip()
    if "```" not in code:
        return code
    inside = False
    extracted: list[str] = []
    for line in code.splitlines():
        if line.startswith("```"):
            inside = not inside
        elif inside:
            extracted.append(line)
    return "\n".join(extracted) if extracted else code


def ast_valid(text: str) -> bool:
    code = extract_code(text)
    if not code:
        return False
    try:
        ast.parse(code)
        return True
    except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
        return False


def outcomes(rows: list[dict[str, Any]], benchmark: str) -> tuple[list[str], list[bool]]:
    selected = [row for row in rows if row["benchmark"] == benchmark]
    ids = [str(row["stable_id"]) for row in selected]
    if benchmark == "mmlu":
        values = [bool(row["correct"]) for row in selected]
    elif benchmark == "bigcodebench":
        values = [ast_valid(str(row["generated_text"])) for row in selected]
    else:
        raise ValueError(benchmark)
    return ids, values


def paired_counts(reference: list[bool], active: list[bool]) -> dict[str, int]:
    return {
        "improved": sum((not left) and right for left, right in zip(reference, active)),
        "regressed": sum(left and (not right) for left, right in zip(reference, active)),
        "unchanged_pass": sum(left and right for left, right in zip(reference, active)),
        "unchanged_fail": sum((not left) and (not right) for left, right in zip(reference, active)),
    }


def completion_profile(rows: list[dict[str, Any]], benchmark: str) -> dict[str, Any]:
    selected = [row for row in rows if row["benchmark"] == benchmark]
    tokens = [int(row["generated_token_count"]) for row in selected]
    ceiling = 256 if benchmark == "humaneval" else 512 if benchmark == "bigcodebench" else 8
    status = Counter(str(row["empty_status"]) for row in selected)
    return {
        "count": len(selected),
        "mean_generated_tokens": mean(tokens),
        "maximum_token_count": ceiling,
        "at_maximum_count": sum(value == ceiling for value in tokens),
        "empty_status_counts": dict(sorted(status.items())),
    }


def main() -> int:
    protocol = load_reduced_protocol()
    selected_ids = set(str(value) for value in protocol["population"]["selected_task_ids"])
    all_b0 = validated(baseline_output(), baseline_run_manifest(), 1716)
    b0 = [row for row in all_b0 if str(row["stable_id"]) in selected_ids]
    require(len(b0) == 726, "Selected B0 count mismatch")

    result: dict[str, Any] = {
        "status": "GENERATION_COMPLETE_EVALUATION_PARTIAL",
        "protocol_sha256": sha256_path(
            OUTPUTS / "reduced_routewise_utility_protocol.json"
        ),
        "scope": (
            "MMLU exact-match and BigCodeBench AST syntax proxy only; "
            "official HumanEval pass@1 remains pending."
        ),
        "b0": {},
        "routes": {},
    }
    b0_outcomes: dict[str, tuple[list[str], list[bool]]] = {}
    for benchmark in ("bigcodebench", "mmlu"):
        ids, values = outcomes(b0, benchmark)
        b0_outcomes[benchmark] = ids, values
        result["b0"][benchmark] = {
            "success_count": sum(values),
            "task_count": len(values),
            "score": sum(values) / len(values),
            "completion_profile": completion_profile(b0, benchmark),
        }
    result["b0"]["humaneval_generation_profile"] = completion_profile(b0, "humaneval")

    for cwe_id, route in protocol["routes"].items():
        rows = validated(output_path(cwe_id, route), run_manifest_path(cwe_id, route), 726)
        route_result: dict[str, Any] = {
            "layer": int(route["layer"]),
            "feature": int(route["feature"]),
            "benchmarks": {},
        }
        for benchmark in ("bigcodebench", "mmlu"):
            active_ids, active_values = outcomes(rows, benchmark)
            b0_ids, b0_values = b0_outcomes[benchmark]
            require(active_ids == b0_ids, f"{cwe_id}/{benchmark} pairing mismatch")
            score = sum(active_values) / len(active_values)
            b0_score = sum(b0_values) / len(b0_values)
            route_result["benchmarks"][benchmark] = {
                "success_count": sum(active_values),
                "task_count": len(active_values),
                "score": score,
                "paired_delta": score - b0_score,
                **paired_counts(b0_values, active_values),
                "completion_profile": completion_profile(rows, benchmark),
            }
        route_result["humaneval_generation_profile"] = completion_profile(rows, "humaneval")
        result["routes"][cwe_id] = route_result

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
