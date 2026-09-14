#!/usr/bin/env python3
"""Evaluate B0 and all route-wise utility outputs with official HumanEval.

Run only in an isolated Linux/Colab runtime because the official evaluator
executes generated code.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any


EXPECTED_PACKAGE_VERSION = "1.0.3"
EXPECTED_TASKS = 164


class EvaluationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationError(message)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def route_slug(cwe_id: str, route: dict[str, Any]) -> str:
    return f"{cwe_id.lower().replace('-', '')}_l{int(route['layer'])}_f{int(route['feature'])}_a40_seed42"


def validated_output(path: Path, manifest_path: Path, expected: int) -> list[dict[str, Any]]:
    require(path.is_file() and manifest_path.is_file(), f"Missing generation artifact for {path.name}")
    manifest = read_json(manifest_path)
    require(manifest.get("status") == "COMPLETE", f"{manifest_path.name} is not COMPLETE")
    require(manifest.get("output_sha256") == sha256(path), f"{path.name} hash mismatch")
    rows = read_json(path)
    require(isinstance(rows, list) and len(rows) == expected, f"{path.name} record count mismatch")
    return rows


def prepare(workdir: Path) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, Any]]:
    protocol_path = workdir / "routewise_utility_protocol.json"
    require(protocol_path.is_file(), "routewise_utility_protocol.json is missing")
    protocol = read_json(protocol_path)
    require(protocol.get("schema_version") == "routewise_bstar_utility_protocol_v1",
            "Protocol schema mismatch")
    total = int(protocol["population"]["total_tasks"])
    b0 = validated_output(workdir / "paired_b0_seed42_outputs.json",
                          workdir / "paired_b0_seed42_run_manifest.json", total)
    b0_he = [row for row in b0 if row.get("benchmark") == "humaneval"]
    require(len(b0_he) == EXPECTED_TASKS, "B0 HumanEval count mismatch")
    task_ids = [str(row["stable_id"]) for row in b0_he]
    require(len(set(task_ids)) == EXPECTED_TASKS, "HumanEval task IDs are not unique")

    from human_eval.data import HUMAN_EVAL, read_problems
    problems = read_problems(HUMAN_EVAL)
    require(set(problems) == set(task_ids), "Official HumanEval task population mismatch")

    conditions: dict[str, dict[str, Any]] = {
        "B0": {
            "slug": "b0",
            "source": workdir / "paired_b0_seed42_outputs.json",
            "rows": b0_he,
        }
    }
    for cwe_id, route in protocol["routes"].items():
        slug = route_slug(cwe_id, route)
        source = workdir / f"route_{slug}_outputs.json"
        manifest = workdir / f"route_{slug}_run_manifest.json"
        rows = validated_output(source, manifest, total)
        human_rows = [row for row in rows if row.get("benchmark") == "humaneval"]
        require([str(row["stable_id"]) for row in human_rows] == task_ids,
                f"{cwe_id} HumanEval order mismatch")
        conditions[cwe_id] = {"slug": slug, "source": source, "rows": human_rows,
                              "layer": int(route["layer"]), "feature": int(route["feature"])}

    for condition in conditions.values():
        sample_path = workdir / f"humaneval_{condition['slug']}_samples.jsonl"
        write_jsonl(sample_path, [
            {"task_id": row["stable_id"], "completion": row["generated_text"]}
            for row in condition["rows"]
        ])
        condition["samples"] = sample_path
        condition["task_results"] = workdir / f"humaneval_{condition['slug']}_task_results.json"
    metadata = {"problem_file": Path(HUMAN_EVAL), "protocol": protocol,
                "protocol_sha256": sha256(protocol_path)}
    return task_ids, conditions, metadata


def evaluate(samples: Path, output: Path, task_ids: list[str], resume: bool) -> tuple[int, str]:
    raw_results = Path(str(samples) + "_results.jsonl")
    if raw_results.is_file():
        require(resume, f"{raw_results.name} exists; use --resume to validate and reuse it")
    else:
        from human_eval.evaluation import evaluate_functional_correctness
        scores = evaluate_functional_correctness(str(samples), k=[1], n_workers=4, timeout=3.0)
        require("pass@1" in scores, "Official evaluator did not return pass@1")
    rows = read_jsonl(raw_results)
    require(len(rows) == EXPECTED_TASKS, f"{samples.name} result count mismatch")
    require([str(row["task_id"]) for row in rows] == task_ids, f"{samples.name} result order mismatch")
    require(all(isinstance(row.get("passed"), bool) for row in rows),
            f"{samples.name} contains a non-Boolean passed value")
    preserved = [
        {
            "task_id": row["task_id"],
            "passed": row["passed"],
            "result": row.get("result"),
            "completion_sha256": hashlib.sha256(row["completion"].encode("utf-8")).hexdigest(),
        }
        for row in rows
    ]
    atomic_json(output, preserved)
    return sum(row["passed"] for row in rows), sha256(raw_results)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, default=Path("."))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    workdir = args.workdir.resolve()
    installed = version("human-eval")
    require(installed == EXPECTED_PACKAGE_VERSION,
            f"human-eval version mismatch: expected {EXPECTED_PACKAGE_VERSION}, got {installed}")
    task_ids, conditions, metadata = prepare(workdir)
    preflight = {
        "schema_version": "routewise_bstar_utility_humaneval_preflight_v1",
        "status": "PASS",
        "evaluation_run": False,
        "package": "human-eval",
        "package_version": installed,
        "condition_count": len(conditions),
        "task_count_per_condition": EXPECTED_TASKS,
        "task_ids_sha256": canonical_sha256(task_ids),
        "protocol_sha256": metadata["protocol_sha256"],
        "problem_file_sha256": sha256(metadata["problem_file"]),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }
    atomic_json(workdir / "routewise_humaneval_preflight.json", preflight)
    print(json.dumps(preflight, indent=2, sort_keys=True))
    if args.preflight_only:
        return 0
    require(os.environ.get("HUMAN_EVAL_ALLOW_CODE_EVAL") == "1",
            "Set HUMAN_EVAL_ALLOW_CODE_EVAL=1 only inside the isolated Colab/Linux runtime")

    results: dict[str, Any] = {}
    for name, condition in conditions.items():
        pass_count, raw_hash = evaluate(
            condition["samples"], condition["task_results"], task_ids, args.resume
        )
        results[name] = {
            "layer": condition.get("layer"),
            "feature": condition.get("feature"),
            "source_sha256": sha256(condition["source"]),
            "samples_sha256": sha256(condition["samples"]),
            "raw_results_sha256": raw_hash,
            "task_results_sha256": sha256(condition["task_results"]),
            "pass_count": pass_count,
            "pass_at_1": pass_count / EXPECTED_TASKS,
        }
        print(f"{name}: {pass_count}/{EXPECTED_TASKS}", flush=True)
    returned = {
        "schema_version": "routewise_bstar_utility_humaneval_return_v1",
        "status": "COMPLETE",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_evaluator": True,
        "package": "human-eval",
        "package_version": installed,
        "k": [1],
        "n_workers": 4,
        "timeout_seconds": 3.0,
        "task_count_per_condition": EXPECTED_TASKS,
        "protocol_sha256": metadata["protocol_sha256"],
        "problem_file_sha256": preflight["problem_file_sha256"],
        "conditions": results,
    }
    atomic_json(workdir / "routewise_humaneval_return_manifest.json", returned)
    print(json.dumps(returned, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EvaluationError as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        raise SystemExit(2)

