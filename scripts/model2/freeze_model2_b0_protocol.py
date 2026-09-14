#!/usr/bin/env python
"""Freeze the approved Model2 B0 development protocol before generation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[4]
DEV = ROOT / "data/cyberseceval/dev_prompts.json"
RUNNER = Path(__file__).with_name("run_model2_b0_dev.py")
OUTPUT = ROOT / "revision/model2/phase21/outputs/model2_b0_dev_protocol.json"
MODEL_ID = "google/gemma-2-9b-it"
MODEL_REVISION = "11c9b309abf73637e4b6f9a3fa1e92e615547819"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    if OUTPUT.exists():
        raise RuntimeError(f"protocol already frozen: {OUTPUT}")
    if not RUNNER.is_file():
        raise RuntimeError(f"runner missing: {RUNNER}")
    rows = json.loads(DEV.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != 1341:
        raise RuntimeError("development source must contain exactly 1341 prompts")
    ids = [int(x["prompt_id"]) for x in rows]
    if len(set(ids)) != 1341:
        raise RuntimeError("development prompt IDs are not unique")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    rendered_records = []
    for source_index, row in enumerate(rows):
        source_prompt = row["test_case_prompt"]
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": source_prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        token_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        rendered_records.append({
            "source_index": source_index,
            "prompt_id": int(row["prompt_id"]),
            "source_prompt_sha256": hashlib.sha256(source_prompt.encode()).hexdigest(),
            "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
            "rendered_input_token_count": len(token_ids),
        })
    value = {
        "schema_version": "phase21_model2_b0_protocol_v1",
        "status": "FROZEN_BEFORE_GENERATION",
        "method": "B0",
        "split": "DEVELOPMENT",
        "source": {"path": "data/cyberseceval/dev_prompts.json", "sha256": sha256_file(DEV), "record_count": 1341},
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "dtype": "bfloat16", "device": "cuda:0", "quantization": False, "offload": False},
        "tokenizer": {"id": MODEL_ID, "revision": MODEL_REVISION, "class": type(tokenizer).__name__, "padding_side": tokenizer.padding_side},
        "prompt_interface": {
            "chat_template_used": True,
            "messages": [{"role": "user", "content": "<test_case_prompt>"}],
            "add_generation_prompt": True,
            "tokenize_template_call": False,
            "tokenizer_add_special_tokens_after_render": False,
            "input_truncation": False,
            "rendered_prompt_aggregate_sha256": canonical_hash([{"prompt_id": x["prompt_id"], "rendered_prompt_sha256": x["rendered_prompt_sha256"]} for x in rendered_records]),
        },
        "generation": {"seed": 42, "temperature": 0.2, "top_p": 0.95, "max_new_tokens": 512, "do_sample": True, "decode_generated_tokens_only": True, "skip_special_tokens": True},
        "checkpoint": {"source_order_prefix_required": True, "rng_state_saved": True, "interval_records": 10, "failure_preservation": True, "changed_setting_retry": False},
        "runner": {"path": "revision/model2/phase21/scripts/run_model2_b0_dev.py", "sha256": sha256_file(RUNNER)},
        "records": rendered_records,
        "heldout_accessed": False,
        "sae_loaded": False,
        "security_intervention": False,
    }
    atomic_json(OUTPUT, value)
    print(json.dumps({"status": value["status"], "record_count": 1341, "protocol_path": str(OUTPUT), "protocol_sha256": sha256_file(OUTPUT), "max_input_tokens": max(x["rendered_input_token_count"] for x in rendered_records)}, indent=2))


if __name__ == "__main__":
    main()
