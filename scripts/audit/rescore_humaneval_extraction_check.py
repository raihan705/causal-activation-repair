#!/usr/bin/env python3
"""Audit HumanEval interfaces and run a no-op extraction sensitivity check.

The script never modifies a preserved generation or result file. In
``--content-only`` mode it performs only JSON/hash/content/provenance checks.
In ``--run-evaluator`` mode it must run under isolated Linux with
``HUMAN_EVAL_ALLOW_CODE_EVAL=1`` and ``human-eval==1.0.3``; generated programs
are executed only by the official HumanEval evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_TASKS = 164
EXPECTED_PACKAGE = "human-eval"
EXPECTED_PACKAGE_VERSION = "1.0.3"
ROLE_MARKER_RULE = (
    "A completion matches when it contains '<|' anywhere, or when a line "
    "consists only of an optional '<|' prefix, one of assistant/system/user "
    "(case-insensitive), an optional '|>' suffix, and an optional colon."
)
ROLE_LINE_RE = re.compile(
    r"(?im)^\s*(?:<\|)?(?:assistant|system|user)(?:\|>)?\s*:?\s*$"
)
FENCED_BLOCK_RE = re.compile(
    r"```[^\r\n]*\r?\n(.*?)(?:\r?\n```|```)", re.DOTALL
)


@dataclass(frozen=True)
class Condition:
    name: str
    panel: str
    source_path: str
    source_sha256: str
    result_path: str
    result_sha256: str
    source_kind: str


CONDITIONS = (
    Condition(
        "panel_b_b0",
        "B_AlwaysOn",
        "outputs/phase10/utility_baseline_humaneval.json",
        "039197df171fb393616b417880e2c902b2fe9339a53a6ab08ed37eac7f043515",
        "revision/model1/phase17/outputs/humaneval_b0_task_results.json",
        "04072354d514d3fe1342e78f3aa89a901641002089dd68ef14c40d3b5b809450",
        "phase10_b0",
    ),
    Condition(
        "panel_b_alwayson",
        "B_AlwaysOn",
        "revision/model1/phase17/outputs/alwayson_utility_seed42_outputs.json",
        "d2df7ce232ad4ab46bcf22ee1807d49df738aad5c26067187b3d48496452e158",
        "revision/model1/phase17/outputs/humaneval_alwayson_task_results.json",
        "86c10ead4ed96b382f196334a5633a9368d8904ecd76752471e0d49823f2025f",
        "benchmark_rows",
    ),
    Condition(
        "panel_a_b0",
        "A_Routewise",
        "revision/model1/routewise_bstar_utility/outputs/paired_b0_seed42_outputs.json",
        "889b20226cbf74badf30d30ec221ac4a028aeebc58756ecf3bce3a5e28dcf001",
        "revision/model1/routewise_bstar_utility/outputs/humaneval_reduced_b0_task_results.json",
        "f82ee280763d1b6960e5103debbd7b121922bdea33e3834447aafbd6fe992617",
        "benchmark_rows",
    ),
    Condition(
        "panel_a_cwe120_l19_f14193_a40_seed42",
        "A_Routewise",
        "revision/model1/routewise_bstar_utility/outputs/reduced_route_cwe120_l19_f14193_a40_seed42_outputs.json",
        "adcd35636f2faf3093f9e596245c1c13483da3ba3dc635a9e24efb12c5bf5384",
        "revision/model1/routewise_bstar_utility/outputs/humaneval_reduced_cwe120_l19_f14193_a40_seed42_task_results.json",
        "a9e8dc83868ba1a162767d40d92aacc7943fa966170467f9352e3299dbacc64a",
        "benchmark_rows",
    ),
    Condition(
        "panel_a_cwe125_l23_f16655_a40_seed42",
        "A_Routewise",
        "revision/model1/routewise_bstar_utility/outputs/reduced_route_cwe125_l23_f16655_a40_seed42_outputs.json",
        "28ceb1e7e78e0ae43f047ce993d68c9ed806d2db34838b3a765bfa8be217900d",
        "revision/model1/routewise_bstar_utility/outputs/humaneval_reduced_cwe125_l23_f16655_a40_seed42_task_results.json",
        "3cfc89f3279e740cebcbdefda0ee4859963ad4e126966c5672a076fcff700d00",
        "benchmark_rows",
    ),
    Condition(
        "panel_a_cwe190_l19_f16897_a40_seed42",
        "A_Routewise",
        "revision/model1/routewise_bstar_utility/outputs/reduced_route_cwe190_l19_f16897_a40_seed42_outputs.json",
        "a309b1bd3b144edea7d18aec544e71d0d4fc181be2ee893f2aa056929c133bd2",
        "revision/model1/routewise_bstar_utility/outputs/humaneval_reduced_cwe190_l19_f16897_a40_seed42_task_results.json",
        "54d69c365bc11f8c88df906a6bf1a83339de19c5dc0f54e912f992cede2b99c3",
        "benchmark_rows",
    ),
    Condition(
        "panel_a_cwe327_l23_f14449_a40_seed42",
        "A_Routewise",
        "revision/model1/routewise_bstar_utility/outputs/reduced_route_cwe327_l23_f14449_a40_seed42_outputs.json",
        "f70244be8b12a077f1c43e90fb66c7f5b0f2d272fbc0d1ec376ec2a18e5011cf",
        "revision/model1/routewise_bstar_utility/outputs/humaneval_reduced_cwe327_l23_f14449_a40_seed42_task_results.json",
        "066586d0b8f479eef0936cd713e6784ead70673481800d02ab6f9e934f09e2b3",
        "benchmark_rows",
    ),
    Condition(
        "panel_a_cwe338_l23_f7533_a40_seed42",
        "A_Routewise",
        "revision/model1/routewise_bstar_utility/outputs/reduced_route_cwe338_l23_f7533_a40_seed42_outputs.json",
        "bde0b3f50ca7ed0d5e7de69a12de67a9fe571918e7c29ba756dbd1a445ce521a",
        "revision/model1/routewise_bstar_utility/outputs/humaneval_reduced_cwe338_l23_f7533_a40_seed42_task_results.json",
        "40533b4ac43e8633b35cb34a86b7b997366be716286fc8f3ec23c27e0848664a",
        "benchmark_rows",
    ),
    Condition(
        "panel_a_cwe476_l23_f18397_a40_seed42",
        "A_Routewise",
        "revision/model1/routewise_bstar_utility/outputs/reduced_route_cwe476_l23_f18397_a40_seed42_outputs.json",
        "c6a60a9e7c5befa85da41a2261a1cd7343c08e58be95f280c2c45413906484f0",
        "revision/model1/routewise_bstar_utility/outputs/humaneval_reduced_cwe476_l23_f18397_a40_seed42_task_results.json",
        "c591b56a26bbc14640133b088718760c4b98c53e612eec5dda6f260e586321de",
        "benchmark_rows",
    ),
    Condition(
        "panel_a_cwe787_l19_f1515_a40_seed42",
        "A_Routewise",
        "revision/model1/routewise_bstar_utility/outputs/reduced_route_cwe787_l19_f1515_a40_seed42_outputs.json",
        "98609bc124324004f7a6ab1e2cc62a03c300d534f6b396d338bc697656979152",
        "revision/model1/routewise_bstar_utility/outputs/humaneval_reduced_cwe787_l19_f1515_a40_seed42_task_results.json",
        "024e2665b1cad24db60fe26adc489f56606273095b8303bd1781d47f0a29c38a",
        "benchmark_rows",
    ),
    Condition(
        "panel_a_cwe79_l16_f9816_a40_seed42",
        "A_Routewise",
        "revision/model1/routewise_bstar_utility/outputs/reduced_route_cwe79_l16_f9816_a40_seed42_outputs.json",
        "7a4db6ff1e3f307f19f0284268dd84f6b38d2b714942d22b7452ce04a0a3ccee",
        "revision/model1/routewise_bstar_utility/outputs/humaneval_reduced_cwe79_l16_f9816_a40_seed42_task_results.json",
        "80a196b398b3a018265a4da855142740e46dbdea60652346941fbd6362bdf181",
        "benchmark_rows",
    ),
    Condition(
        "panel_a_cwe89_l23_f1652_a40_seed42",
        "A_Routewise",
        "revision/model1/routewise_bstar_utility/outputs/reduced_route_cwe89_l23_f1652_a40_seed42_outputs.json",
        "416ab733f5a1e5c39970efe47b2317a6af416dfb4cf0bcfdae54cd1f5b8e4c9e",
        "revision/model1/routewise_bstar_utility/outputs/humaneval_reduced_cwe89_l23_f1652_a40_seed42_task_results.json",
        "3e2069c7ac194beb8aba1bcc7325b2e3daf8021393ed30d9a8e4b94915900288",
        "benchmark_rows",
    ),
)


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def installed_versions(distributions: tuple[str, ...]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in distributions:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def rows_for(condition: Condition, workspace: Path) -> list[dict[str, str]]:
    source = workspace / condition.source_path
    require(source.is_file(), f"Missing source file: {condition.source_path}")
    require(sha256_file(source) == condition.source_sha256, f"Source hash mismatch: {condition.source_path}")
    payload = read_json(source)
    require(isinstance(payload, list), f"Source is not a JSON list: {condition.source_path}")
    if condition.source_kind == "phase10_b0":
        selected = [
            {"task_id": str(row["task_id"]), "completion": str(row["completion"])}
            for row in payload
        ]
    else:
        selected = [
            {"task_id": str(row["stable_id"]), "completion": str(row["generated_text"])}
            for row in payload
            if row.get("benchmark") == "humaneval"
        ]
    require(len(selected) == EXPECTED_TASKS, f"Expected 164 HumanEval rows for {condition.name}")
    require(len({row["task_id"] for row in selected}) == EXPECTED_TASKS, f"Duplicate task IDs: {condition.name}")
    require(
        set(row["task_id"] for row in selected) == {f"HumanEval/{index}" for index in range(EXPECTED_TASKS)},
        f"Unexpected HumanEval task population: {condition.name}",
    )
    return selected


def original_outcomes(condition: Condition, workspace: Path, rows: list[dict[str, str]]) -> dict[str, bool]:
    path = workspace / condition.result_path
    require(path.is_file(), f"Missing original result file: {condition.result_path}")
    require(sha256_file(path) == condition.result_sha256, f"Result hash mismatch: {condition.result_path}")
    payload = read_json(path)
    require(isinstance(payload, list) and len(payload) == EXPECTED_TASKS, f"Bad result population: {condition.name}")
    expected_hashes = {row["task_id"]: sha256_text(row["completion"]) for row in rows}
    outcomes: dict[str, bool] = {}
    for row in payload:
        task_id = str(row["task_id"])
        require(isinstance(row.get("passed"), bool), f"Malformed passed field: {condition.name}/{task_id}")
        require(row.get("completion_sha256") == expected_hashes[task_id], f"Completion hash mismatch: {condition.name}/{task_id}")
        outcomes[task_id] = bool(row["passed"])
    require(set(outcomes) == set(expected_hashes), f"Result task IDs differ: {condition.name}")
    return outcomes


def first_nonblank_line(value: str) -> str:
    for line in value.splitlines():
        if line.strip():
            return line
    return ""


def extract_first_fenced_block(value: str) -> str:
    """Return the first complete fenced block, or the input unchanged."""
    if "```" not in value:
        return value
    match = FENCED_BLOCK_RE.search(value)
    return match.group(1) if match else value


def content_record(condition: Condition, rows: list[dict[str, str]]) -> dict[str, Any]:
    fenced = [row["task_id"] for row in rows if "```" in row["completion"]]
    chat = [
        row["task_id"]
        for row in rows
        if "<|" in row["completion"] or ROLE_LINE_RE.search(row["completion"])
    ]
    empty = [row["task_id"] for row in rows if not row["completion"].strip()]
    top_level_def = [
        row["task_id"]
        for row in rows
        if first_nonblank_line(row["completion"]).startswith("def ")
    ]
    extracted_changed = [
        row["task_id"]
        for row in rows
        if extract_first_fenced_block(row["completion"]) != row["completion"]
    ]
    return {
        "condition": condition.name,
        "panel": condition.panel,
        "source_path": condition.source_path,
        "source_sha256": condition.source_sha256,
        "original_result_path": condition.result_path,
        "original_result_sha256": condition.result_sha256,
        "humaneval_completion_count": len(rows),
        "markdown_fence_count": len(fenced),
        "chat_role_marker_count": len(chat),
        "empty_or_whitespace_count": len(empty),
        "top_level_def_first_nonblank_count": len(top_level_def),
        "top_level_def_example_task_ids": top_level_def[:3],
        "extraction_text_changed_count": len(extracted_changed),
        "extraction_text_changed_task_ids": extracted_changed,
    }


def path_evidence(workspace: Path, relative: str, expected_sha256: str | None = None) -> dict[str, Any]:
    path = workspace / relative
    require(path.is_file(), f"Missing provenance file: {relative}")
    actual = sha256_file(path)
    if expected_sha256 is not None:
        require(actual == expected_sha256, f"Provenance hash mismatch: {relative}")
    stat = path.stat()
    return {
        "path": relative,
        "sha256": actual,
        "size_bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def provenance(workspace: Path) -> dict[str, Any]:
    phase10_output = path_evidence(
        workspace,
        "outputs/phase10/utility_baseline_humaneval.json",
        "039197df171fb393616b417880e2c902b2fe9339a53a6ab08ed37eac7f043515",
    )
    phase10_copy = path_evidence(
        workspace,
        "outputs/phase10/utility_finalmethod_humaneval.json",
        "039197df171fb393616b417880e2c902b2fe9339a53a6ab08ed37eac7f043515",
    )
    fix_script = path_evidence(
        workspace,
        "phases/phase10/fix_humaneval_generation.py",
        "ae41243609081b8f5213ad4ada697a40c5822cd58af648c41dce3e44fa4c120a",
    )
    obsolete_script = path_evidence(
        workspace,
        "phases/phase10/run_phase10_baseline_utility.py",
        "ad803a9ea7cc9951f667f9fd4111ad6549113d7bd156eb5fe595f3b62aada730",
    )
    phase10_audit = path_evidence(
        workspace,
        "revision/model1/phase10/work.md",
        "6184a308ef0c680ce28b03723564a5b1d39c27f6e5379451572956f0b30ffd3f",
    )
    phase17_script = path_evidence(workspace, "revision/model1/phase17/scripts/evaluate_phase17_humaneval.py")
    phase17_return = path_evidence(
        workspace,
        "revision/model1/phase17/outputs/phase17_humaneval_return_manifest.json",
        "b64a032de9d17d817ab64bb4798f5bcd46ef96ecafdff6b79e8dd096040ed64a",
    )
    route_runner = path_evidence(
        workspace,
        "revision/model1/routewise_bstar_utility/scripts/run_routewise_utility.py",
        "b12746f7346ceeb4a80cfa9ebeeb98bbb894567a4797d6d0e032ffe22e0a8705",
    )
    route_common = path_evidence(workspace, "revision/model1/routewise_bstar_utility/scripts/routewise_utility_common.py")
    route_manifest = path_evidence(
        workspace,
        "revision/model1/routewise_bstar_utility/outputs/paired_b0_seed42_run_manifest.json",
        "39912bf06646a31be6ea2fee087f0850a18e7c27b15e027d335e5a2074cc1a6f",
    )
    return {
        "phase10_panel_b_b0": {
            "status": "VERIFIED_WITH_PRESERVED_AUDIT_AND_HASH_CHAIN",
            "statement": (
                "The current Phase 10 HumanEval B0 is the corrected raw-canonical-prompt generation "
                "written by fix_humaneval_generation.py, not the overwritten chat-template output."
            ),
            "executed_semantics": {
                "prompt_rendering": "RAW_CANONICAL_PROMPT_NO_CHAT_TEMPLATE",
                "decode": "GENERATED_SUFFIX_ONLY",
                "input_token_limit": 2048,
                "max_new_tokens": 256,
                "stop_policy": "first token of each frozen stop string plus tokenizer EOS",
            },
            "evidence": [
                fix_script,
                obsolete_script,
                phase10_output,
                phase10_copy,
                phase10_audit,
                phase17_script,
                phase17_return,
            ],
            "evidence_interpretation": (
                "The corrected B0 was modified after both the obsolete runner and correction script were created; "
                "B0 and the copied inactive B* file have identical pinned hashes, matching the correction script's "
                "documented one-generation-plus-copy behavior. The Phase 10 audit identifies that hash as the "
                "corrected raw completion, and the later Phase 17 evaluator pins and records the same source hash."
            ),
            "limitation": (
                "The original correction command log and deleted generation checkpoint are not preserved; "
                "provenance rests on the preserved audit, file chronology, exact B0/B* identity, and downstream hash pinning."
            ),
        },
        "routewise_panel_a_b0": {
            "status": "VERIFIED_BY_RUN_MANIFEST_AND_RUNNER_HASH",
            "statement": (
                "Panel A B0 was generated by run_routewise_utility.py using the HumanEval branch in "
                "routewise_utility_common.py."
            ),
            "executed_semantics": {
                "prompt_expression": "task['prompt']",
                "input_token_limit": 2048,
                "max_new_tokens": 256,
                "stop_policy": "first token of each canonical stop string plus tokenizer EOS",
                "decode": "GENERATED_SUFFIX_ONLY",
            },
            "evidence": [route_runner, route_common, route_manifest],
            "evidence_interpretation": (
                "The COMPLETE run manifest records the exact runner SHA-256, RAW_CANONICAL_CONTINUATION, "
                "2048/256 token limits, suffix-only decoding, frozen stop strings, and the B0 output hash."
            ),
        },
    }


def run_official_evaluator(
    condition: Condition,
    rows: list[dict[str, str]],
    original: dict[str, bool],
    audit_dir: Path,
) -> dict[str, Any]:
    from human_eval.evaluation import evaluate_functional_correctness

    samples = audit_dir / "samples" / f"{condition.name}.jsonl"
    rescored_results = audit_dir / "task_results" / f"{condition.name}.json"
    extracted_rows = [
        {"task_id": row["task_id"], "completion": extract_first_fenced_block(row["completion"])}
        for row in rows
    ]
    write_jsonl(samples, extracted_rows)
    raw_results = Path(str(samples) + "_results.jsonl")
    require(not raw_results.exists(), f"Refusing to overwrite existing raw evaluator output: {raw_results}")
    evaluate_functional_correctness(str(samples), k=[1], n_workers=4, timeout=3.0)
    evaluated = read_jsonl(raw_results)
    require(len(evaluated) == EXPECTED_TASKS, f"Evaluator row count mismatch: {condition.name}")
    rescored: dict[str, bool] = {}
    compact: list[dict[str, Any]] = []
    for row in evaluated:
        task_id = str(row["task_id"])
        require(isinstance(row.get("passed"), bool), f"Evaluator passed field malformed: {condition.name}/{task_id}")
        require(task_id not in rescored, f"Duplicate evaluator task: {condition.name}/{task_id}")
        rescored[task_id] = bool(row["passed"])
        compact.append(
            {
                "task_id": task_id,
                "passed": bool(row["passed"]),
                "result": row.get("result"),
                "completion_sha256": sha256_text(str(row["completion"])),
            }
        )
    require(set(rescored) == set(original), f"Evaluator task population differs: {condition.name}")
    write_json(rescored_results, compact)
    changed = sorted(task_id for task_id in original if original[task_id] != rescored[task_id])
    extracted_changed = {
        row["task_id"]
        for row in rows
        if extract_first_fenced_block(row["completion"]) != row["completion"]
    }
    changed_details = [
        {
            "task_id": task_id,
            "original_passed": original[task_id],
            "rescored_passed": rescored[task_id],
            "reason": (
                "EXTRACTION_CHANGED_COMPLETION"
                if task_id in extracted_changed
                else "IDENTICAL_COMPLETION_DIFFERENT_EVALUATOR_OUTCOME_REQUIRES_INVESTIGATION"
            ),
        }
        for task_id in changed
    ]
    return {
        "condition": condition.name,
        "panel": condition.panel,
        "original_pass_count": sum(original.values()),
        "rescored_pass_count": sum(rescored.values()),
        "changed_outcome_count": len(changed),
        "changed_task_ids": changed,
        "changed_outcomes": changed_details,
        "samples_path": samples.as_posix(),
        "samples_sha256": sha256_file(samples),
        "raw_evaluator_results_path": raw_results.as_posix(),
        "raw_evaluator_results_sha256": sha256_file(raw_results),
        "compact_task_results_path": rescored_results.as_posix(),
        "compact_task_results_sha256": sha256_file(rescored_results),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--content-only", action="store_true")
    mode.add_argument("--run-evaluator", action="store_true")
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    audit_dir = args.audit_dir.resolve()
    output = args.output.resolve()
    audit_dir.mkdir(parents=True, exist_ok=True)

    rows_by_condition: dict[str, list[dict[str, str]]] = {}
    original_by_condition: dict[str, dict[str, bool]] = {}
    content: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        rows = rows_for(condition, workspace)
        original = original_outcomes(condition, workspace, rows)
        rows_by_condition[condition.name] = rows
        original_by_condition[condition.name] = original
        record = content_record(condition, rows)
        record["original_pass_count"] = sum(original.values())
        content.append(record)

    result: dict[str, Any] = {
        "schema_version": "humaneval_interface_audit_v1",
        "status": "AWAITING_ISOLATED_EVALUATOR" if args.content_only else "IN_PROGRESS",
        "verification_only": True,
        "preserved_generation_files_modified": False,
        "frozen_results_modified": False,
        "reported_numbers_modified": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "chat_role_marker_matching_rule": ROLE_MARKER_RULE,
        "extraction_rule": (
            "Use the contents of the first complete Markdown fenced block when one exists; "
            "otherwise return the original completion byte-for-byte. An unmatched fence is passed through unchanged."
        ),
        "content_audit": content,
        "provenance": provenance(workspace),
        "evaluator_protocol": {
            "package": EXPECTED_PACKAGE,
            "required_version": EXPECTED_PACKAGE_VERSION,
            "function": "human_eval.evaluation.evaluate_functional_correctness",
            "k": [1],
            "n_workers": 4,
            "timeout_seconds": 3.0,
            "required_environment_variable": "HUMAN_EVAL_ALLOW_CODE_EVAL=1",
            "required_platform": "isolated Linux runtime",
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "package_versions": installed_versions(
                ("human-eval", "numpy", "tqdm", "fire")
            ),
        },
        "rescoring": [],
    }

    if args.content_only:
        result["rescoring"] = [
            {
                "condition": condition.name,
                "panel": condition.panel,
                "original_pass_count": sum(original_by_condition[condition.name].values()),
                "rescored_pass_count": None,
                "changed_outcome_count": None,
                "changed_task_ids": None,
                "status": "AWAITING_ISOLATED_EVALUATOR",
            }
            for condition in CONDITIONS
        ]
        write_json(output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    require(platform.system() == "Linux", "Official evaluator execution is restricted to isolated Linux")
    require(
        os.environ.get("HUMAN_EVAL_ALLOW_CODE_EVAL") == "1",
        "Set HUMAN_EVAL_ALLOW_CODE_EVAL=1 only inside the isolated runtime",
    )
    installed = importlib.metadata.version(EXPECTED_PACKAGE)
    require(installed == EXPECTED_PACKAGE_VERSION, f"Expected human-eval 1.0.3, found {installed}")
    result["runtime"]["package_versions"] = installed_versions(
        ("human-eval", "numpy", "tqdm", "fire")
    )

    for condition in CONDITIONS:
        rescored = run_official_evaluator(
            condition,
            rows_by_condition[condition.name],
            original_by_condition[condition.name],
            audit_dir,
        )
        result["rescoring"].append(rescored)
        if rescored["changed_outcome_count"]:
            result["status"] = "CHANGED_OUTCOME_DETECTED_STOPPED"
            write_json(output, result)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 2

    result["status"] = "COMPLETE_NO_OP"
    result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
