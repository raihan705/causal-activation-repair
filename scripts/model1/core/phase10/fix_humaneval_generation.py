#!/usr/bin/env python3
"""Corrected Phase 10 HumanEval generation protocol.

This script supersedes only the HumanEval branch in
``run_phase10_baseline_utility.py``. It supplies the canonical HumanEval
function prefix directly to the model, decodes only the generated suffix,
and uses the frozen 2,048/256-token and stopping settings. The selected
Model 1 intervention is inactive on HumanEval, so the same completions are
written for B0 and the selected-method condition.

This is a model-generation script. It does not run HumanEval's executable
tests; evaluate generated programs only in an isolated runtime.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
STOP_STRINGS = ("\ndef ", "\nclass ", "\n#", "\nif __name__")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path(os.environ.get("CAR_ROOT", ".")),
        help="Reconstruction workspace (default: CAR_ROOT or current directory).",
    )
    parser.add_argument(
        "--model-id",
        default=MODEL_ID,
        help="Hugging Face model identifier or local model directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permit replacement of existing HumanEval output files.",
    )
    return parser.parse_args()


def load_problems() -> list[dict[str, str]]:
    """Load the canonical 164-task HumanEval population."""
    try:
        from datasets import load_dataset

        dataset = load_dataset("openai_humaneval", split="test")
        problems = [
            {"task_id": str(row["task_id"]), "prompt": str(row["prompt"])}
            for row in dataset
        ]
    except Exception as dataset_error:
        print(f"datasets loader failed ({dataset_error}); using human-eval package")
        from human_eval.data import read_problems

        raw = read_problems()
        problems = [
            {"task_id": str(task_id), "prompt": str(problem["prompt"])}
            for task_id, problem in raw.items()
        ]
    if len(problems) != 164:
        raise RuntimeError(f"Expected 164 HumanEval tasks, found {len(problems)}")
    return problems


def load_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


def canonical_stop_ids(tokenizer) -> list[int]:
    stop_ids: list[int] = []
    for stop_string in STOP_STRINGS:
        encoded = tokenizer.encode(stop_string, add_special_tokens=False)
        if encoded:
            stop_ids.append(int(encoded[0]))
    stop_ids.append(int(tokenizer.eos_token_id))
    return sorted(set(stop_ids))


def generate_raw_completion(tokenizer, model, prompt: str) -> str:
    """Continue the canonical function prefix without applying a chat template."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            temperature=0.2,
            top_p=0.95,
            max_new_tokens=256,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=canonical_stop_ids(tokenizer),
        )
    prompt_length = inputs["input_ids"].shape[1]
    return tokenizer.decode(
        generated[0][prompt_length:],
        skip_special_tokens=True,
    )


def read_checkpoint(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError(f"Malformed checkpoint: {path}")
    return payload


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.workdir.resolve() / "outputs" / "phase10"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "_ckpt_humaneval_fix.json"
    b0_output = output_dir / "utility_baseline_humaneval.json"
    selected_output = output_dir / "utility_finalmethod_humaneval.json"

    existing = [path for path in (b0_output, selected_output) if path.exists()]
    if existing and not args.overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite preserved outputs: {names}. "
            "Use a fresh workspace or pass --overwrite deliberately."
        )

    problems = load_problems()
    tokenizer, model = load_model(args.model_id)
    results = read_checkpoint(checkpoint)
    completed = {str(row["task_id"]) for row in results}
    remaining = [problem for problem in problems if problem["task_id"] not in completed]

    for index, problem in enumerate(tqdm(remaining, desc="HumanEval-corrected"), start=1):
        results.append(
            {
                "task_id": problem["task_id"],
                "prompt": problem["prompt"],
                "completion": generate_raw_completion(
                    tokenizer,
                    model,
                    problem["prompt"],
                ),
                "steered": False,
            }
        )
        if index % 50 == 0:
            write_json(checkpoint, results)

    if len(results) != 164 or len({row["task_id"] for row in results}) != 164:
        raise RuntimeError("Corrected HumanEval generation did not produce 164 unique tasks")
    write_json(b0_output, results)
    write_json(selected_output, [{**row, "steered": False} for row in results])
    checkpoint.unlink(missing_ok=True)
    print(f"Wrote corrected B0 completions: {b0_output}")
    print(f"Wrote inactive selected-method copy: {selected_output}")


if __name__ == "__main__":
    main()
