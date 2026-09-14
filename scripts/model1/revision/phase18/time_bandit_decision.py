#!/usr/bin/env python3
"""Time the frozen B4 policy-decision function without LLM generation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[4]
PHASE = ROOT / "revision/model1/phase18"
SUBSET = PHASE / "outputs/timing_subset_manifest.json"
DATASET = ROOT / "outputs/phase6/offline_bandit_dataset.jsonl"
CHECKPOINT = ROOT / "outputs/phase7/checkpoints/linear_bandit_epoch_01.ckpt"
ACTION_MAP = ROOT / "outputs/phase6/bandit_action_map.json"
RAW_OUTPUT = PHASE / "outputs/raw_timing/b4_policy_timing.json"
CSV_OUTPUT = PHASE / "outputs/phase18_bandit_overhead.csv"

EXPECTED = {
    SUBSET: "b25fc05efc915f1f6fcde7b4a78b06f771605c356ce991551a0c38b451665c3c",
    DATASET: "d251e5d816691041819ed85b770e63845d240c372fa2ade680906421e3cdd1eb",
    CHECKPOINT: "c4c38bc404223cc6e37632ff781487ded42a4c21c4ba0fe7983838a9cd33fa89",
    ACTION_MAP: "67ed4f4d81d8edfe5096a50587937fc0eaeaa4c2d576949e7ec9ef21035a9193",
}


class LinearPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(102, 28)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.fc(state)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_states(ids: list[int]) -> dict[int, list[float]]:
    wanted = set(ids)
    states: dict[int, list[float]] = {}
    with DATASET.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt_id = int(row["prompt_id"])
            if prompt_id not in wanted:
                continue
            state = list(row["state"])
            require(len(state) == 102, f"B4 state width mismatch for prompt {prompt_id}")
            if prompt_id in states:
                require(states[prompt_id] == state, f"B4 states differ across actions for prompt {prompt_id}")
            else:
                states[prompt_id] = state
    require(set(states) == wanted, f"Missing frozen B4 states: {sorted(wanted - set(states))}")
    return states


def preflight() -> dict[str, Any]:
    for path, expected in EXPECTED.items():
        require(path.is_file(), f"Missing B4 input: {path}")
        require(sha256(path) == expected, f"B4 input hash mismatch: {path}")
    subset = read_json(SUBSET)
    ids = list(map(int, subset["ordered_prompt_ids"]))
    require(len(ids) == 50 and len(set(ids)) == 50, "Frozen timing IDs invalid")
    states = load_states(ids)
    return {
        "status": "PASS",
        "method": "B4-policy-decision",
        "device": "cpu",
        "batch_size": 1,
        "prompt_count": len(ids),
        "state_count": len(states),
        "state_source": DATASET.relative_to(ROOT).as_posix(),
        "state_source_sha256": EXPECTED[DATASET],
        "timing_subset_sha256": EXPECTED[SUBSET],
        "checkpoint_sha256": EXPECTED[CHECKPOINT],
        "heldout_used": False,
        "policy_updated": False,
        "llm_generation": False,
    }


def run() -> None:
    gate = preflight()
    subset = read_json(SUBSET)
    ids = list(map(int, subset["ordered_prompt_ids"]))
    states = load_states(ids)

    load_start = time.perf_counter_ns()
    try:
        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    model = LinearPolicy().to("cpu").eval()
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=False)
    require(not incompatible.missing_keys and not incompatible.unexpected_keys, "B4 state-dict incompatibility")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    load_seconds = (time.perf_counter_ns() - load_start) / 1e9

    warmup_id = int(subset["warmup_prompt_id"])
    with torch.inference_mode():
        model(torch.tensor([states[warmup_id]], dtype=torch.float32))

    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for repetition in (1, 2, 3):
            for source_index, prompt_id in enumerate(ids):
                state = torch.tensor([states[prompt_id]], dtype=torch.float32)
                start = time.perf_counter_ns()
                logits = model(state)
                prediction = int(logits.argmax(dim=1).item())
                elapsed_ns = time.perf_counter_ns() - start
                require(tuple(logits.shape) == (1, 28) and bool(torch.isfinite(logits).all()), "Invalid B4 output")
                rows.append({
                    "method": "B4-policy-decision", "repetition": repetition,
                    "source_index": source_index, "prompt_id": prompt_id,
                    "batch_size": 1, "latency_seconds": elapsed_ns / 1e9,
                    "latency_nanoseconds": elapsed_ns, "predicted_action_id": prediction,
                    "failure": None,
                })

    payload = {
        "schema_version": "phase18_b4_policy_timing_v1",
        "status": "COMPLETE",
        "preflight": gate,
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__},
        "load_seconds": load_seconds,
        "warmup": {"prompt_id": warmup_id, "count": 1, "excluded_from_measurements": True},
        "repetitions": 3,
        "records": rows,
    }
    write_json(RAW_OUTPUT, payload)

    CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = CSV_OUTPUT.with_name(CSV_OUTPUT.name + ".tmp")
    fields = list(rows[0])
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, CSV_OUTPUT)
    print(json.dumps({"status": "COMPLETE", "records": len(rows), "raw_sha256": sha256(RAW_OUTPUT), "csv_sha256": sha256(CSV_OUTPUT)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        print(json.dumps(preflight(), indent=2, sort_keys=True))
    else:
        run()


if __name__ == "__main__":
    main()
