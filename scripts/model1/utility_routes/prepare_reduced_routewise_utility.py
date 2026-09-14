#!/usr/bin/env python3
"""Freeze the approved feasibility-reduced active B* utility population."""

from __future__ import annotations

import json
import random
from pathlib import Path

from routewise_utility_common import (
    ALPHA, BENCHMARK_ORDER, BSTAR_CONFIG, EXPECTED_BSTAR_CONFIG_SHA256,
    EXPECTED_PHASE17_INPUT_SHA256, EXPECTED_ROUTES, MODEL_ID, MODEL_REVISION,
    OUTPUTS, PHASE17_INPUT, SAE_RELEASE, SAE_REVISION, SEED, atomic_json,
    canonical_sha256, ordered_tasks, relative, sha256_path, validate_sources,
)


PROTOCOL = OUTPUTS / "reduced_routewise_utility_protocol.json"
BIGCODE_SAMPLE_SIZE = 150


def main() -> int:
    source, config = validate_sources()
    all_tasks = ordered_tasks(source)
    by_benchmark = {
        benchmark: [task for task in all_tasks if task["benchmark"] == benchmark]
        for benchmark in BENCHMARK_ORDER
    }
    rng = random.Random(SEED)
    selected_bigcode_indices = sorted(rng.sample(range(len(by_benchmark["bigcodebench"])),
                                                 BIGCODE_SAMPLE_SIZE))
    selected = (
        by_benchmark["humaneval"]
        + [by_benchmark["bigcodebench"][index] for index in selected_bigcode_indices]
        + by_benchmark["mmlu"]
    )
    selected_ids = [str(task["stable_id"]) for task in selected]
    rebuilt_rng = random.Random(SEED)
    rebuilt_indices = sorted(rebuilt_rng.sample(range(len(by_benchmark["bigcodebench"])),
                                                BIGCODE_SAMPLE_SIZE))
    assert rebuilt_indices == selected_bigcode_indices
    assert len(selected) == 726 and len(set(selected_ids)) == 726
    protocol = {
        "schema_version": "reduced_routewise_bstar_utility_protocol_v1",
        "status": "FROZEN",
        "amendment_reason": (
            "The full 1,716-task by nine-route design was stopped after the first 25 active "
            "records because observed maximum-length generation implied approximately five "
            "days of continuous execution. No utility outcomes were inspected before this amendment."
        ),
        "scientific_role": "PRESPECIFIED_ACTIVE_SINGLE_ROUTE_UTILITY_SENSITIVITY_SUBSET",
        "interpretation_boundary": (
            "All nine frozen B* coordinates are evaluated independently. HumanEval and MMLU "
            "retain their complete populations; BigCodeBench estimates apply to the deterministic "
            "150-task subset and are not presented as full-benchmark estimates."
        ),
        "source_manifest_path": relative(PHASE17_INPUT),
        "source_manifest_sha256": EXPECTED_PHASE17_INPUT_SHA256,
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
            "execution_order": list(BENCHMARK_ORDER),
            "counts": {"humaneval": 164, "bigcodebench": BIGCODE_SAMPLE_SIZE, "mmlu": 412},
            "total_tasks_per_route": len(selected),
            "selection": {
                "humaneval": "COMPLETE_POPULATION",
                "bigcodebench": "random.Random(42).sample(range(1140), 150), then restore source order",
                "mmlu": "COMPLETE_FOUR_SLICE_POPULATION",
            },
            "bigcodebench_selected_source_indices": selected_bigcode_indices,
            "bigcodebench_selected_source_indices_sha256": canonical_sha256(selected_bigcode_indices),
            "selected_task_ids": selected_ids,
            "selected_task_ids_sha256": canonical_sha256(selected_ids),
            "deterministic_rebuild_identical": True,
        },
        "paired_b0": {
            "path": "revision/model1/routewise_bstar_utility/outputs/paired_b0_seed42_outputs.json",
            "sha256": "889b20226cbf74badf30d30ec221ac4a028aeebc58756ecf3bce3a5e28dcf001",
            "full_record_count": 1716,
            "reuse": "Select the 726 approved task records and restore each stored pre-task RNG state",
        },
        "generation": {
            "seed": SEED,
            "temperature": 0.2,
            "top_p": 0.95,
            "do_sample": True,
            "decode_generated_tokens_only": True,
            "skip_special_tokens": True,
            "batch_size": 1,
            "humaneval": {"rendering": "RAW_CANONICAL_CONTINUATION",
                          "input_truncation": 2048, "max_new_tokens": 256,
                          "stop_strings": ["\ndef ", "\nclass ", "\n#", "\nif __name__"],
                          "eos_policy": "FIRST_TOKEN_OF_EACH_STOP_STRING_PLUS_TOKENIZER_EOS"},
            "bigcodebench": {"rendering": "LLAMA_CHAT_PHASE17",
                             "input_truncation": 2048, "max_new_tokens": 512},
            "mmlu": {"rendering": "LLAMA_CHAT_PHASE17",
                     "input_truncation": 2048, "max_new_tokens": 8},
        },
        "routing_contract": {
            "task_cwe_fields_consumed": [],
            "route_selection": "EXPERIMENT_CONDITION_ONLY",
            "one_feature_per_condition": True,
            "simultaneous_features": False,
            "artificial_task_cwe_assignment": False,
            "all_nine_routes_required": True,
        },
        "evaluation": {
            "humaneval": "Official human-eval 1.0.3 pass@1 on all 164 tasks",
            "bigcodebench": "Python AST syntax-pass proxy on the frozen 150-task subset",
            "mmlu": "Exact A/B/C/D accuracy on all 412 tasks",
            "uncertainty": "10,000 paired task-level bootstrap resamples, seed 42",
            "reporting": "Per-route results; no pooled deployable B* estimate",
        },
        "superseded_execution": {
            "full_protocol_sha256": "a471dcae4e527dff2dd98fa1e57b8ebbaaf062f152c7c28e913814ec16519a4a",
            "preserved_checkpoint": "revision/model1/routewise_bstar_utility/outputs/route_cwe120_l19_f14193_a40_seed42_checkpoint.json",
            "checkpoint_record_count": 25,
            "reuse_in_reduced_analysis": False,
        },
        "producer": {"path": relative(Path(__file__)), "sha256": sha256_path(Path(__file__))},
    }
    atomic_json(PROTOCOL, protocol)
    print(json.dumps({
        "status": protocol["status"],
        "protocol_path": relative(PROTOCOL),
        "protocol_sha256": sha256_path(PROTOCOL),
        "counts": protocol["population"]["counts"],
        "tasks_per_route": len(selected),
        "route_count": len(EXPECTED_ROUTES),
        "active_generation_count": len(selected) * len(EXPECTED_ROUTES),
        "bigcodebench_indices_sha256": protocol["population"]["bigcodebench_selected_source_indices_sha256"],
        "selected_task_ids_sha256": protocol["population"]["selected_task_ids_sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
