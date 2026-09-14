#!/usr/bin/env python3
"""Deterministically summarize complete Phase 18 raw timing artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "revision/model1/phase18/outputs"
RAW = OUT / "raw_timing"
SUBSET = OUT / "timing_subset_manifest.json"
METHODS = ("B0", "B1", "B*", "B3-ungated", "B3-PG", "B1-CWE", "RCI-1", "CAA-CWE")
SLUGS = {method: method.lower().replace("*", "star").replace("-", "_") for method in METHODS}
SUBSET_SHA = "b25fc05efc915f1f6fcde7b4a78b06f771605c356ce991551a0c38b451665c3c"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    require(rows, f"No rows for {path.name}")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def aggregate(method: str, repetition: str | int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(row["latency_seconds"]) for row in rows]
    total_latency = sum(latencies)
    generated = sum(int(row["generated_tokens"]) for row in rows)
    inputs = sum(int(row["input_tokens"]) for row in rows)
    calls = sum(int(row["model_call_count"]) for row in rows)
    return {
        "method": method, "repetition": repetition, "prompt_records": len(rows), "batch_size": 1,
        "total_wall_seconds": total_latency, "mean_latency_seconds": statistics.fmean(latencies),
        "median_latency_seconds": statistics.median(latencies), "min_latency_seconds": min(latencies),
        "max_latency_seconds": max(latencies), "failure_count": sum(row["generation_status"] != "COMPLETED" for row in rows),
        "fallback_count": sum(row.get("fallback_status") not in (None, "NONE") for row in rows),
        "input_tokens": inputs, "generated_tokens": generated, "model_calls": calls,
        "mean_model_calls_per_prompt": calls / len(rows),
        "generated_tokens_per_second": generated / total_latency if total_latency else 0.0,
    }


def preflight() -> dict[str, Any]:
    require(SUBSET.is_file() and sha256(SUBSET) == SUBSET_SHA, "Timing subset hash mismatch")
    files = {}
    for method in METHODS:
        path = RAW / f"{SLUGS[method]}_timing.json"
        require(path.is_file(), f"Missing raw timing: {method}")
        body = read_json(path)
        require(body.get("status") == "COMPLETE" and body.get("method") == method, f"Incomplete raw timing: {method}")
        require(body.get("batch_size") == 1 and body.get("repetitions") == 3, f"Protocol mismatch: {method}")
        rows = body.get("records", [])
        require(len(rows) == 150, f"Expected 150 records for {method}")
        expected = [(rep, prompt_id) for rep in (1, 2, 3) for prompt_id in read_json(SUBSET)["ordered_prompt_ids"]]
        require([(int(row["repetition"]), int(row["prompt_id"])) for row in rows] == expected, f"Order mismatch: {method}")
        files[method] = sha256(path)
    b4 = RAW / "b4_policy_timing.json"
    require(b4.is_file() and read_json(b4).get("status") == "COMPLETE", "B4 timing incomplete")
    require(len(read_json(b4)["records"]) == 150, "B4 record count mismatch")
    scanner = OUT / "phase18_scanner_timing.json"
    require(scanner.is_file() and read_json(scanner).get("status") == "COMPLETE", "Scanner timing incomplete")
    return {"status": "PASS", "generation_raw_hashes": files, "b4_sha256": sha256(b4), "scanner_sha256": sha256(scanner)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    gate = preflight()
    if args.preflight_only:
        print(json.dumps(gate, indent=2, sort_keys=True))
        return

    latency_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    memory_rows: list[dict[str, Any]] = []
    aggregate_by_method: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        body = read_json(RAW / f"{SLUGS[method]}_timing.json")
        rows = body["records"]
        for repetition in (1, 2, 3):
            summary = aggregate(method, repetition, [row for row in rows if int(row["repetition"]) == repetition])
            latency_rows.append({key: summary[key] for key in ("method", "repetition", "prompt_records", "batch_size", "total_wall_seconds", "mean_latency_seconds", "median_latency_seconds", "min_latency_seconds", "max_latency_seconds", "failure_count", "fallback_count")})
            token_rows.append({key: summary[key] for key in ("method", "repetition", "input_tokens", "generated_tokens", "model_calls", "mean_model_calls_per_prompt", "generated_tokens_per_second")})
        overall = aggregate(method, "ALL", rows)
        aggregate_by_method[method] = overall
        latency_rows.append({key: overall[key] for key in ("method", "repetition", "prompt_records", "batch_size", "total_wall_seconds", "mean_latency_seconds", "median_latency_seconds", "min_latency_seconds", "max_latency_seconds", "failure_count", "fallback_count")})
        token_rows.append({key: overall[key] for key in ("method", "repetition", "input_tokens", "generated_tokens", "model_calls", "mean_model_calls_per_prompt", "generated_tokens_per_second")})
        memory_rows.append({
            "method": method, "load_session_count": len(body["load_sessions"]),
            "total_load_seconds": sum(float(row["load_seconds"]) for row in body["load_sessions"]),
            "peak_allocated_bytes": int(body["peak_allocated_bytes"]),
            "peak_reserved_bytes": int(body["peak_reserved_bytes"]),
        })

    b4 = read_json(RAW / "b4_policy_timing.json")
    b4_rows = []
    for repetition in (1, 2, 3):
        rows = [row for row in b4["records"] if int(row["repetition"]) == repetition]
        latencies = [float(row["latency_seconds"]) for row in rows]
        b4_rows.append({"method": "B4-policy-decision", "repetition": repetition, "decision_count": len(rows),
                        "total_wall_seconds": sum(latencies), "mean_latency_seconds": statistics.fmean(latencies),
                        "median_latency_seconds": statistics.median(latencies), "failure_count": sum(row["failure"] is not None for row in rows)})
    all_b4 = [float(row["latency_seconds"]) for row in b4["records"]]
    b4_rows.append({"method": "B4-policy-decision", "repetition": "ALL", "decision_count": len(all_b4),
                    "total_wall_seconds": sum(all_b4), "mean_latency_seconds": statistics.fmean(all_b4),
                    "median_latency_seconds": statistics.median(all_b4), "failure_count": 0})

    scanner = read_json(OUT / "phase18_scanner_timing.json")
    scanner_rows = scanner["measured_repetitions"]
    write_csv(OUT / "phase18_method_latency.csv", latency_rows)
    write_csv(OUT / "phase18_tokens_throughput.csv", token_rows)
    write_csv(OUT / "phase18_memory.csv", memory_rows)
    write_csv(OUT / "phase18_bandit_overhead.csv", b4_rows)
    write_csv(OUT / "phase18_scanner_time.csv", scanner_rows)

    b0 = aggregate_by_method["B0"]
    b1 = aggregate_by_method["B1"]
    summary = {
        "schema_version": "phase18_cost_summary_v1", "phase": 18, "status": "COMPLETE",
        "timing_subset_sha256": SUBSET_SHA, "preflight": gate,
        "methods": aggregate_by_method,
        "overheads": {
            "bstar_vs_b0_mean_latency_ratio": aggregate_by_method["B*"]["mean_latency_seconds"] / b0["mean_latency_seconds"],
            "bstar_vs_b0_mean_latency_percent": 100.0 * (aggregate_by_method["B*"]["mean_latency_seconds"] / b0["mean_latency_seconds"] - 1.0),
            "b1cwe_vs_b1_mean_latency_ratio": aggregate_by_method["B1-CWE"]["mean_latency_seconds"] / b1["mean_latency_seconds"],
            "rci1_vs_b0_mean_latency_ratio": aggregate_by_method["RCI-1"]["mean_latency_seconds"] / b0["mean_latency_seconds"],
            "b3pg_vs_b0_mean_latency_ratio": aggregate_by_method["B3-PG"]["mean_latency_seconds"] / b0["mean_latency_seconds"],
        },
        "b4": b4_rows[-1], "scanner": scanner,
        "output_hashes": {}, "no_optimization": True, "heldout_used": False,
    }
    for name in ("phase18_method_latency.csv", "phase18_tokens_throughput.csv", "phase18_memory.csv", "phase18_bandit_overhead.csv", "phase18_scanner_time.csv"):
        summary["output_hashes"][name] = sha256(OUT / name)
    atomic_json(OUT / "phase18_cost_summary.json", summary)
    print(json.dumps({"status": "COMPLETE", "summary_sha256": sha256(OUT / "phase18_cost_summary.json"), "overheads": summary["overheads"]}, indent=2))


if __name__ == "__main__":
    main()
