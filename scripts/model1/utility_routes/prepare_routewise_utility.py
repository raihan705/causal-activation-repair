#!/usr/bin/env python3
"""Freeze the route-wise active B* utility protocol without loading a model."""

from __future__ import annotations

import json
from pathlib import Path

from routewise_utility_common import (
    ALPHA, BENCHMARK_ORDER, BSTAR_CONFIG, EXPECTED_BSTAR_CONFIG_SHA256,
    EXPECTED_COUNTS, EXPECTED_PHASE17_INPUT_SHA256, MODEL_ID, MODEL_REVISION,
    OUTPUTS, PHASE17_INPUT, PROTOCOL, SAE_RELEASE, SAE_REVISION, SEED,
    atomic_json, canonical_sha256, ordered_tasks, relative, sha256_path,
    validate_sources,
)


def main() -> int:
    source, config = validate_sources()
    tasks = ordered_tasks(source)
    task_ids = [str(row["stable_id"]) for row in tasks]
    protocol = {
        "schema_version": "routewise_bstar_utility_protocol_v1",
        "status": "FROZEN",
        "scientific_role": "ACTIVE_SINGLE_ROUTE_UTILITY_SENSITIVITY",
        "interpretation_boundary": (
            "Each frozen B* coordinate is applied independently to every utility task. "
            "This is an active coordinate-wise stress test, not a deployable CWE router and "
            "not a pooled B* effect estimate."
        ),
        "source_manifest_path": relative(PHASE17_INPUT),
        "source_manifest_sha256": EXPECTED_PHASE17_INPUT_SHA256,
        "source_populations_sha256": canonical_sha256(source["populations"]),
        "bstar_config_path": relative(BSTAR_CONFIG),
        "bstar_config_sha256": EXPECTED_BSTAR_CONFIG_SHA256,
        "routes": config["feature_map"],
        "alpha": ALPHA,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "dtype": "float16"},
        "sae": {
            "release": SAE_RELEASE,
            "revision": SAE_REVISION,
            "ids_by_layer": {"16": "l16r_8x", "19": "l19r_8x", "23": "l23r_8x"},
        },
        "population": {
            "benchmark_order": list(BENCHMARK_ORDER),
            "benchmark_counts": EXPECTED_COUNTS,
            "total_tasks": len(tasks),
            "task_ids_sha256": canonical_sha256(task_ids),
        },
        "task_ids_sha256": canonical_sha256(task_ids),
        "generation": {
            "seed": SEED,
            "temperature": 0.2,
            "top_p": 0.95,
            "do_sample": True,
            "decode_generated_tokens_only": True,
            "skip_special_tokens": True,
            "batch_size": 1,
            "humaneval": {
                "rendering": "RAW_CANONICAL_CONTINUATION",
                "input_truncation": 2048,
                "max_new_tokens": 256,
                "stop_strings": ["\ndef ", "\nclass ", "\n#", "\nif __name__"],
                "eos_policy": "FIRST_TOKEN_OF_EACH_STOP_STRING_PLUS_TOKENIZER_EOS",
            },
            "bigcodebench": {
                "rendering": "LLAMA_CHAT_PHASE17",
                "input_truncation": 2048,
                "max_new_tokens": 512,
            },
            "mmlu": {
                "rendering": "LLAMA_CHAT_PHASE17",
                "input_truncation": 2048,
                "max_new_tokens": 8,
            },
        },
        "paired_rng_contract": {
            "baseline": "Fresh B0 stream seeded once with 42 after model loading",
            "capture": "CPU and all-CUDA RNG states captured immediately before each B0 task",
            "intervention": "The corresponding pre-task state is restored immediately before each route generation",
            "resume": "B0 restores its checkpoint RNG state; route runs use stored per-task B0 states",
            "purpose": "Remove separate stochastic-sampling variation from task-level B0-versus-route contrasts",
        },
        "routing_contract": {
            "task_cwe_fields_consumed": [],
            "route_selection": "EXPERIMENT_CONDITION_ONLY",
            "one_feature_per_condition": True,
            "simultaneous_features": False,
            "artificial_task_cwe_assignment": False,
        },
        "evaluation": {
            "humaneval": "official human-eval 1.0.3 pass@1 in isolated Linux/Colab",
            "bigcodebench": "Python AST syntax-pass proxy retained from Phase 17",
            "mmlu": "exact A/B/C/D accuracy retained from Phase 17",
            "uncertainty": "10,000 paired task-level bootstrap resamples; seed 42; equal-tailed Type-7 95% intervals",
            "tolerances": {"humaneval": 0.015, "bigcodebench": 0.02, "mmlu": 0.01},
            "reporting": "Per-route estimates plus benchmark-wise worst observed route; no pooled B* score",
        },
        "producer": {"path": relative(Path(__file__)), "sha256": sha256_path(Path(__file__))},
    }
    atomic_json(PROTOCOL, protocol)
    result = {
        "status": "PASS_FROZEN_AWAITING_HASH_BOUND_APPROVAL",
        "protocol_path": relative(PROTOCOL),
        "protocol_sha256": sha256_path(PROTOCOL),
        "task_count": len(tasks),
        "route_count": len(config["feature_map"]),
        "active_generation_count": len(tasks) * len(config["feature_map"]),
        "fresh_b0_generation_count": len(tasks),
        "total_generation_count": len(tasks) * (1 + len(config["feature_map"])),
        "outputs_directory": relative(OUTPUTS),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

