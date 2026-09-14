#!/usr/bin/env python
"""Freeze Stage 22E steering execution only after paired denominators pass."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
PHASE = ROOT / "revision/model3/phase22"
OUT = PHASE / "outputs"
PROTOCOL = OUT / "phase22e_causal_validation_protocol.json"
DENOMINATOR = OUT / "phase22e_paired_denominator_manifest.json"
RUNNER = PHASE / "scripts/run_phase22e_causal_steering.py"
OUTPUT = OUT / "phase22e_steering_execution_protocol.json"
TARGET_ORDER = {"CWE-120": 0, "CWE-327": 1, "CWE-89": 2}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    require(not OUTPUT.exists(), "immutable steering execution protocol already exists")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    denominator = json.loads(DENOMINATOR.read_text(encoding="utf-8"))
    require(denominator["status"] == "FROZEN_BEFORE_STEERING" and denominator["all_targets_evaluable"], "paired denominator gate not passed")
    prompt_by_key = {(x["target_cwe"], int(x["prompt_id"])): x for x in protocol["paired_baseline_prompts"]}
    assignments = sorted(protocol["candidate_assignments"], key=lambda x: (int(x["layer"]), TARGET_ORDER[x["target_cwe"]], int(x["statistical_rank"])))
    worklist = []
    for assignment in assignments:
        target = assignment["target_cwe"]
        den = denominator["targets"][target]
        for arm, ids in (("QUALIFIED_UNSAFE", den["qualified_unsafe_prompt_ids"]), ("QUALIFIED_SAFE", den["qualified_safe_prompt_ids"])):
            for prompt_id in ids:
                prompt = prompt_by_key[(target, int(prompt_id))]
                worklist.append({
                    "record_index": len(worklist),
                    "record_id": f"STEER|{target}|L{assignment['layer']}|F{assignment['feature_id']}|{arm}|P{prompt_id}",
                    "target_cwe": target, "layer": int(assignment["layer"]),
                    "feature_id": int(assignment["feature_id"]), "statistical_rank": int(assignment["statistical_rank"]),
                    "composite_score": float(assignment["composite_score"]), "screening_alpha": float(assignment["screening_alpha"]),
                    "route_key": assignment["route_key"], "arm": arm, "prompt_id": int(prompt_id),
                    "source_index": int(prompt["source_index"]), "language": prompt["language"],
                    "source_prompt": prompt["source_prompt"], "source_prompt_sha256": prompt["source_prompt_sha256"],
                    "rendered_prompt_sha256": prompt["rendered_prompt_sha256"], "input_token_count": int(prompt["input_token_count"]),
                })
    require(len(worklist) == denominator["expected_steered_generation_count"] == 1170, "steering worklist count mismatch")
    require(len({x["record_id"] for x in worklist}) == 1170, "duplicate worklist record ID")
    value = {
        "schema_version": "phase22e_steering_execution_protocol_v1",
        "status": "FROZEN_BEFORE_STEERED_GENERATION",
        "causal_protocol_sha256": sha256_file(PROTOCOL),
        "paired_denominator_sha256": sha256_file(DENOMINATOR),
        "runner_path": "revision/model3/phase22/scripts/run_phase22e_causal_steering.py",
        "runner_sha256": sha256_file(RUNNER),
        "worklist_count": len(worklist),
        "worklist_canonical_sha256": canonical_sha256(worklist),
        "order": "layer ascending; target CWE-120/CWE-327/CWE-89; statistical rank; qualified unsafe then qualified safe; frozen paired prompt order",
        "checkpoint_interval": 5,
        "resume_rule": "only exact immutable worklist prefix with matching execution protocol and runner hashes",
        "model": protocol["model"], "sae": protocol["sae"],
        "intervention": protocol["strength_rule"], "generation": protocol["generation"],
        "qualified_counts": {target: {"unsafe": row["qualified_unsafe_count"], "safe": row["qualified_safe_count"]} for target, row in denominator["targets"].items()},
        "scanner_run": False, "heldout_used": False,
    }
    atomic_json(OUTPUT, value)
    print(json.dumps({**value, "execution_protocol_sha256": sha256_file(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
