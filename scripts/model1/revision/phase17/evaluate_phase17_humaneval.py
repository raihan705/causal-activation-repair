#!/usr/bin/env python3
"""Run official HumanEval 1.0.3 for preserved B0 and Phase 17 AlwaysOn."""

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


EXPECTED_B0_SHA256 = "039197df171fb393616b417880e2c902b2fe9339a53a6ab08ed37eac7f043515"
EXPECTED_ACTIVE_SHA256 = "d2df7ce232ad4ab46bcf22ee1807d49df738aad5c26067187b3d48496452e158"
EXPECTED_PACKAGE_VERSION = "1.0.3"
EXPECTED_TASKS = 164


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                 ensure_ascii=False).encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prepare(workdir: Path) -> tuple[list[str], dict[str, Path]]:
    b0_path = workdir / "utility_baseline_humaneval.json"
    active_path = workdir / "alwayson_utility_seed42_outputs.json"
    require(b0_path.is_file() and active_path.is_file(), "Required HumanEval source file is missing")
    require(sha256(b0_path) == EXPECTED_B0_SHA256, "Preserved B0 hash mismatch")
    require(sha256(active_path) == EXPECTED_ACTIVE_SHA256, "AlwaysOn utility hash mismatch")
    b0 = read_json(b0_path)
    active_all = read_json(active_path)
    require(isinstance(b0, list) and len(b0) == EXPECTED_TASKS, "B0 HumanEval count mismatch")
    active = [row for row in active_all if row.get("benchmark") == "humaneval"]
    require(len(active) == EXPECTED_TASKS, "AlwaysOn HumanEval count mismatch")
    task_ids = [str(row["task_id"]) for row in b0]
    require(len(set(task_ids)) == EXPECTED_TASKS, "B0 HumanEval IDs are not unique")
    require([str(row["stable_id"]) for row in active] == task_ids, "AlwaysOn HumanEval order mismatch")

    from human_eval.data import HUMAN_EVAL, read_problems
    problems = read_problems(HUMAN_EVAL)
    require(set(problems) == set(task_ids), "Official HumanEval task population mismatch")
    require(all(problems[row["task_id"]]["prompt"] == row["prompt"] for row in b0),
            "Official HumanEval prompt mismatch")

    samples = {
        "b0": workdir / "humaneval_b0_samples.jsonl",
        "alwayson": workdir / "humaneval_alwayson_samples.jsonl",
    }
    write_jsonl(samples["b0"], [{"task_id": row["task_id"], "completion": row["completion"]} for row in b0])
    write_jsonl(samples["alwayson"], [{"task_id": row["stable_id"], "completion": row["generated_text"]}
                                       for row in active])
    return task_ids, {**samples, "problem": Path(HUMAN_EVAL)}


def evaluate(samples: Path, output: Path, task_ids: list[str], resume: bool) -> tuple[float, str]:
    result_jsonl = Path(str(samples) + "_results.jsonl")
    if result_jsonl.is_file():
        require(resume, f"{result_jsonl.name} exists; use --resume to validate and reuse")
    else:
        from human_eval.evaluation import evaluate_functional_correctness
        scores = evaluate_functional_correctness(str(samples), k=[1], n_workers=4, timeout=3.0)
        require("pass@1" in scores, "Official evaluator did not return pass@1")
    rows = read_jsonl(result_jsonl)
    require(len(rows) == EXPECTED_TASKS, f"{samples.name} result count mismatch")
    require([str(row["task_id"]) for row in rows] == task_ids, f"{samples.name} result order mismatch")
    require(all(isinstance(row.get("passed"), bool) for row in rows), "Official passed field is not Boolean")
    preserved = [{"task_id": row["task_id"], "passed": row["passed"], "result": row.get("result"),
                  "completion_sha256": hashlib.sha256(row["completion"].encode("utf-8")).hexdigest()}
                 for row in rows]
    atomic_json(output, preserved)
    return sum(row["passed"] for row in rows) / EXPECTED_TASKS, sha256(result_jsonl)


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
    task_ids, paths = prepare(workdir)
    preflight = {
        "schema_version": "phase17_humaneval_preflight_v1", "status": "PASS",
        "package": "human-eval", "package_version": installed,
        "task_count_per_condition": EXPECTED_TASKS, "task_ids_sha256": canonical_sha256(task_ids),
        "b0_source_sha256": EXPECTED_B0_SHA256, "active_source_sha256": EXPECTED_ACTIVE_SHA256,
        "problem_file_sha256": sha256(paths["problem"]), "platform": platform.platform(),
        "python_version": platform.python_version(), "evaluation_run": False,
    }
    atomic_json(workdir / "phase17_humaneval_preflight.json", preflight)
    print(json.dumps(preflight, indent=2, sort_keys=True))
    if args.preflight_only:
        return 0
    require(os.environ.get("HUMAN_EVAL_ALLOW_CODE_EVAL") == "1",
            "Set HUMAN_EVAL_ALLOW_CODE_EVAL=1 only inside the isolated Colab/Linux runtime")

    b0_output = workdir / "humaneval_b0_task_results.json"
    active_output = workdir / "humaneval_alwayson_task_results.json"
    b0_score, b0_raw_hash = evaluate(paths["b0"], b0_output, task_ids, args.resume)
    active_score, active_raw_hash = evaluate(paths["alwayson"], active_output, task_ids, args.resume)
    result = {
        "schema_version": "phase17_humaneval_return_manifest_v1", "status": "COMPLETE",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "official_evaluator": True,
        "package": "human-eval", "package_version": installed, "k": [1], "n_workers": 4,
        "timeout_seconds": 3.0, "task_count_per_condition": EXPECTED_TASKS,
        "problem_file_sha256": preflight["problem_file_sha256"],
        "samples": {
            "B0": {"source_sha256": EXPECTED_B0_SHA256, "samples_sha256": sha256(paths["b0"]),
                   "raw_results_sha256": b0_raw_hash, "task_results_sha256": sha256(b0_output),
                   "pass_count": round(b0_score * EXPECTED_TASKS), "pass_at_1": b0_score},
            "B*-AlwaysOn-L19": {"source_sha256": EXPECTED_ACTIVE_SHA256,
                   "samples_sha256": sha256(paths["alwayson"]), "raw_results_sha256": active_raw_hash,
                   "task_results_sha256": sha256(active_output),
                   "pass_count": round(active_score * EXPECTED_TASKS), "pass_at_1": active_score},
        },
        "command_contract": "evaluate_functional_correctness(sample_file, k=[1], n_workers=4, timeout=3.0)",
        "lexical_fallback_used": False,
    }
    atomic_json(workdir / "phase17_humaneval_return_manifest.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        raise SystemExit(2)
