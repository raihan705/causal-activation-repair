#!/usr/bin/env python3
"""Run official HumanEval for the reduced route-wise experiment in Linux/Colab."""

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
EXPECTED_CONDITIONS = 10


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


def evaluate(samples: Path, output: Path, task_ids: list[str], resume: bool) -> tuple[int, str]:
    raw = Path(str(samples) + "_results.jsonl")
    if raw.is_file():
        require(resume, f"{raw.name} exists; use --resume to validate and reuse")
    else:
        from human_eval.evaluation import evaluate_functional_correctness
        scores = evaluate_functional_correctness(str(samples), k=[1], n_workers=4, timeout=3.0)
        require("pass@1" in scores, "Official evaluator did not return pass@1")
    rows = read_jsonl(raw)
    require(len(rows) == EXPECTED_TASKS, f"{samples.name} result count mismatch")
    require([str(row["task_id"]) for row in rows] == task_ids, f"{samples.name} result order mismatch")
    require(all(isinstance(row.get("passed"), bool) for row in rows), "Non-Boolean passed field")
    preserved = [{"task_id": row["task_id"], "passed": row["passed"], "result": row.get("result"),
                  "completion_sha256": hashlib.sha256(row["completion"].encode("utf-8")).hexdigest()}
                 for row in rows]
    atomic_json(output, preserved)
    return sum(row["passed"] for row in rows), sha256(raw)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, default=Path("."))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    workdir = args.workdir.resolve()
    input_path = workdir / "reduced_humaneval_evaluation_input.json"
    require(input_path.is_file(), "Reduced HumanEval input is missing")
    payload = read_json(input_path)
    require(payload.get("schema_version") == "reduced_routewise_humaneval_input_v1",
            "Input schema mismatch")
    require(payload.get("status") == "FROZEN", "Input is not frozen")
    conditions = payload.get("conditions", {})
    require(len(conditions) == EXPECTED_CONDITIONS, "Condition count mismatch")
    b0_ids = [str(row["task_id"]) for row in conditions["B0"]["records"]]
    require(len(b0_ids) == EXPECTED_TASKS and len(set(b0_ids)) == EXPECTED_TASKS,
            "B0 task population mismatch")
    require(canonical_sha256(b0_ids) == payload["task_ids_sha256"], "Task-ID hash mismatch")
    for name, condition in conditions.items():
        require([str(row["task_id"]) for row in condition["records"]] == b0_ids,
                f"{name} task order mismatch")
    installed = version("human-eval")
    require(installed == EXPECTED_PACKAGE_VERSION,
            f"Expected human-eval {EXPECTED_PACKAGE_VERSION}, found {installed}")
    from human_eval.data import HUMAN_EVAL, read_problems
    problems = read_problems(HUMAN_EVAL)
    require(set(problems) == set(b0_ids), "Official HumanEval population mismatch")
    preflight = {
        "schema_version": "reduced_routewise_humaneval_preflight_v1",
        "status": "PASS",
        "evaluation_run": False,
        "input_sha256": sha256(input_path),
        "protocol_sha256": payload["protocol_sha256"],
        "condition_count": len(conditions),
        "task_count_per_condition": EXPECTED_TASKS,
        "task_ids_sha256": payload["task_ids_sha256"],
        "package": "human-eval",
        "package_version": installed,
        "problem_file_sha256": sha256(Path(HUMAN_EVAL)),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }
    atomic_json(workdir / "reduced_routewise_humaneval_preflight.json", preflight)
    print(json.dumps(preflight, indent=2, sort_keys=True))
    if args.preflight_only:
        return 0
    require(os.environ.get("HUMAN_EVAL_ALLOW_CODE_EVAL") == "1",
            "Set HUMAN_EVAL_ALLOW_CODE_EVAL=1 only inside isolated Linux/Colab")
    results: dict[str, Any] = {}
    for name, condition in conditions.items():
        slug = condition["slug"]
        samples = workdir / f"humaneval_reduced_{slug}_samples.jsonl"
        task_results = workdir / f"humaneval_reduced_{slug}_task_results.json"
        write_jsonl(samples, condition["records"])
        pass_count, raw_hash = evaluate(samples, task_results, b0_ids, args.resume)
        results[name] = {
            "slug": slug,
            "layer": condition.get("layer"),
            "feature": condition.get("feature"),
            "source_output_sha256": condition["source_output_sha256"],
            "samples_sha256": sha256(samples),
            "raw_results_sha256": raw_hash,
            "task_results_sha256": sha256(task_results),
            "pass_count": pass_count,
            "pass_at_1": pass_count / EXPECTED_TASKS,
        }
        print(f"{name}: {pass_count}/{EXPECTED_TASKS}", flush=True)
    returned = {
        "schema_version": "reduced_routewise_humaneval_return_v1",
        "status": "COMPLETE",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_evaluator": True,
        "input_sha256": sha256(input_path),
        "protocol_sha256": payload["protocol_sha256"],
        "package": "human-eval",
        "package_version": installed,
        "k": [1], "n_workers": 4, "timeout_seconds": 3.0,
        "task_count_per_condition": EXPECTED_TASKS,
        "conditions": results,
    }
    atomic_json(workdir / "reduced_routewise_humaneval_return_manifest.json", returned)
    print(json.dumps(returned, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EvaluationError as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        raise SystemExit(2)

