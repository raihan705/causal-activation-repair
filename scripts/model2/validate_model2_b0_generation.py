#!/usr/bin/env python
"""Independent deterministic validation of the completed Model2 B0 generation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "revision/model2/phase21/outputs"
DEV = ROOT / "data/cyberseceval/dev_prompts.json"
PROTOCOL = OUT / "model2_b0_dev_protocol.json"
GENERATION = OUT / "model2_b0_dev_outputs.json"
CHECKPOINT = OUT / "model2_b0_dev_checkpoint.json"
RUN_MANIFEST = OUT / "model2_b0_dev_run_manifest.json"
FINAL_MANIFEST = OUT / "model2_b0_dev_generation_manifest.json"
VALIDATION = OUT / "model2_b0_dev_local_validation.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    required = [DEV, PROTOCOL, GENERATION, CHECKPOINT, RUN_MANIFEST, FINAL_MANIFEST]
    require(all(x.is_file() for x in required), "one or more required B0 artifacts are missing")
    source = json.loads(DEV.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    generated = json.loads(GENERATION.read_text(encoding="utf-8"))
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    run = json.loads(RUN_MANIFEST.read_text(encoding="utf-8"))
    manifest = json.loads(FINAL_MANIFEST.read_text(encoding="utf-8"))
    require(len(source) == len(protocol["records"]) == len(generated) == len(checkpoint["records"]) == 1341, "record-count mismatch")
    source_ids = [int(x["prompt_id"]) for x in source]
    generated_ids = [int(x["prompt_id"]) for x in generated]
    require(len(set(source_ids)) == 1341 and generated_ids == source_ids, "ID uniqueness/order mismatch")
    require(manifest["completed_prompt_ids_source_order"] == source_ids, "manifest ID order mismatch")
    require(checkpoint["records"] == generated, "final checkpoint records differ from final generation")
    require(run["status"] == "COMPLETE" and run["completed_records"] == 1341, "run manifest incomplete")
    require(manifest["status"] == "COMPLETE", "generation manifest incomplete")
    generation_hash = sha256_file(GENERATION)
    require(generation_hash == run["output_sha256"] == manifest["generation_output"]["sha256"], "generation hash chain mismatch")
    require(sha256_file(PROTOCOL) == run["condition"]["protocol_sha256"] == manifest["protocol"]["sha256"], "protocol hash chain mismatch")
    require(sha256_file(FINAL_MANIFEST) == run["generation_manifest_sha256"], "final manifest hash mismatch")
    expected_settings = {"seed": 42, "temperature": 0.2, "top_p": 0.95, "max_new_tokens": 512, "do_sample": True, "decode_generated_tokens_only": True, "skip_special_tokens": True}
    status_counts: dict[str, int] = {}
    token_counts = []
    generated_hashes = []
    for i, (src, frozen, rec) in enumerate(zip(source, protocol["records"], generated, strict=True)):
        pid = int(src["prompt_id"])
        require(int(frozen["source_index"]) == i and int(rec["source_index"]) == i, f"source index mismatch at {i}")
        require(int(frozen["prompt_id"]) == pid == int(rec["prompt_id"]), f"prompt ID mismatch at {i}")
        require(rec["prompt_text"] == src["test_case_prompt"], f"prompt text mismatch for {pid}")
        require(rec["source_prompt_sha256"] == frozen["source_prompt_sha256"] == hashlib.sha256(src["test_case_prompt"].encode()).hexdigest(), f"source prompt hash mismatch for {pid}")
        require(rec["rendered_prompt_sha256"] == frozen["rendered_prompt_sha256"], f"rendered hash mismatch for {pid}")
        require(rec["input_token_count"] == frozen["rendered_input_token_count"], f"token count mismatch for {pid}")
        require(rec["source_cwe"] == rec["cwe_id"] == src.get("cwe_identifier", ""), f"CWE mismatch for {pid}")
        require(rec["language"] == src.get("language", ""), f"language mismatch for {pid}")
        require(rec["method"] == "B0" and rec["seed"] == 42, f"method/seed mismatch for {pid}")
        require(rec["generation_settings"] == expected_settings, f"generation settings mismatch for {pid}")
        require(rec["chat_template_used"] is True, f"chat template flag mismatch for {pid}")
        require(rec["model_id"] == "google/gemma-2-9b-it" and rec["model_revision"] == "11c9b309abf73637e4b6f9a3fa1e92e615547819", f"model mismatch for {pid}")
        require(rec["tokenizer_id"] == rec["model_id"] and rec["tokenizer_revision"] == rec["model_revision"], f"tokenizer mismatch for {pid}")
        require(rec["sae_intervention_applied"] is False and rec["causal_steering_applied"] is False, f"intervention flag mismatch for {pid}")
        require(rec["generated_text"] == rec["generated_code"], f"scanner text alias mismatch for {pid}")
        require(isinstance(rec["generated_token_count"], int) and 0 <= rec["generated_token_count"] <= 512, f"generated token count out of bounds for {pid}")
        status_counts[rec["generation_status"]] = status_counts.get(rec["generation_status"], 0) + 1
        token_counts.append(rec["generated_token_count"])
        generated_hashes.append({"prompt_id": pid, "generated_text_sha256": hashlib.sha256(rec["generated_text"].encode()).hexdigest()})
    require(status_counts == {"SUCCESS": 1341}, "generation contains preserved failures or unknown statuses")
    require(all(x["failure_status"] is None for x in generated), "failure reason exists on successful record")
    require(all(x["empty_status"] == "NONEMPTY" and x["generated_text"].strip() for x in generated), "empty output detected")
    require(manifest["heldout_ids_used"] is False and manifest["heldout_accessed"] is False, "manifest reports held-out use")
    require(manifest["sae_intervention_applied"] is False and manifest["causal_steering_applied"] is False, "manifest reports intervention")
    value = {
        "schema_version": "phase21_model2_b0_local_validation_v1",
        "validation_status": "PASS",
        "generation_sha256": generation_hash,
        "generation_manifest_sha256": sha256_file(FINAL_MANIFEST),
        "protocol_sha256": sha256_file(PROTOCOL),
        "source_sha256": sha256_file(DEV),
        "record_count": 1341,
        "unique_prompt_id_count": 1341,
        "source_order_preserved": True,
        "checkpoint_records_equal_final_output": True,
        "rendered_prompt_hashes_match_frozen_protocol": True,
        "configuration_uniform": True,
        "generation_status_counts": status_counts,
        "empty_or_whitespace_count": 0,
        "failure_count": 0,
        "generated_token_count": {"minimum": min(token_counts), "maximum": max(token_counts), "mean": sum(token_counts) / len(token_counts), "at_maximum_512": sum(x == 512 for x in token_counts)},
        "generated_text_hash_aggregate_sha256": canonical_hash(generated_hashes),
        "heldout_file_opened_by_validator": False,
        "heldout_accessed_by_generator": False,
        "scanner_run": False,
        "sae_loaded": False,
        "causal_intervention_run": False,
        "model1_modified": False,
    }
    atomic_json(VALIDATION, value)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
