"""Revision Phase 3: verify dense residual-addition hook readiness.

This is a fixed technical wiring smoke, not CAA vector construction or tuning.
It loads no SAE and writes only the revision Phase 3 readiness artifact.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import transformers
from huggingface_hub.constants import HF_HUB_CACHE
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_PATH = Path(__file__).resolve()
PROMPTS_PATH = PROJECT_ROOT / "data/cyberseceval/dev_prompts.json"
OUTPUT_PATH = PROJECT_ROOT / "revision/model1/phase3/dense_hook_readiness.json"

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LAYERS = (16, 19, 23)
HIDDEN_WIDTH = 4096
PROMPT_ID = 1866
MAX_INPUT_TOKENS = 1024
MAX_NEW_TOKENS = 4
MULTIPLIER = 0.125
DELTA_ATOL_FLOOR = 0.03125
SEED = 42
COMMAND = (
    "conda run -n svrac python "
    "revision/model1/phase3/scripts/verify_dense_hook_readiness.py"
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    tmp_path = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def current_hf_snapshot(repo_id: str) -> str | None:
    repo_dir = Path(HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}"
    ref = repo_dir / "refs/main"
    if not ref.is_file():
        return None
    value = ref.read_text(encoding="utf-8").strip()
    return value or None


def load_fixed_prompt() -> dict[str, Any]:
    prompts = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    matches = [item for item in prompts if item.get("prompt_id") == PROMPT_ID]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one development prompt {PROMPT_ID}; found {len(matches)}"
        )
    prompt = matches[0]
    if not isinstance(prompt.get("test_case_prompt"), str) or not prompt["test_case_prompt"]:
        raise ValueError(f"Development prompt {PROMPT_ID} has no test_case_prompt")
    return prompt


def make_probe_vector() -> torch.Tensor:
    # Fixed alternating values: deterministic, dense, and unrelated to any outcome.
    vector = torch.ones(HIDDEN_WIDTH, dtype=torch.float32)
    vector[1::2] = -1.0
    return vector


def make_hook(vector_cpu: torch.Tensor, call_records: list[dict[str, Any]]):
    def hook_fn(module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[-1] != HIDDEN_WIDTH:
            raise ValueError(
                f"Expected hidden width {HIDDEN_WIDTH}; got {hidden.shape[-1]}"
            )

        vector = vector_cpu.to(device=hidden.device, dtype=hidden.dtype)
        if tuple(vector.shape) != (hidden.shape[-1],):
            raise ValueError(
                f"Vector shape {tuple(vector.shape)} cannot target width {hidden.shape[-1]}"
            )

        delta = MULTIPLIER * vector.view(1, 1, -1)
        edited = hidden + delta
        actual_delta = edited - hidden
        delta_error = (actual_delta - delta).abs()
        expected_error = (edited - (hidden + delta)).abs()
        hidden_max_abs = float(hidden.abs().max().item())
        edited_max_abs = float(edited.abs().max().item())
        # Float16 spacing grows with activation magnitude. This bound checks the
        # observed delta without falsely rejecting correctly rounded additions.
        rounding_bound = torch.finfo(hidden.dtype).eps * max(
            hidden_max_abs, edited_max_abs, 1.0
        )
        delta_tolerance = max(DELTA_ATOL_FLOOR, rounding_bound)

        record = {
            "call_index": len(call_records),
            "shape": list(hidden.shape),
            "tuple_output": isinstance(output, tuple),
            "dtype": str(hidden.dtype),
            "device": str(hidden.device),
            "hidden_finite": bool(torch.isfinite(hidden).all().item()),
            "vector_finite": bool(torch.isfinite(vector).all().item()),
            "delta_finite": bool(torch.isfinite(delta).all().item()),
            "edited_finite": bool(torch.isfinite(edited).all().item()),
            "hidden_max_abs": hidden_max_abs,
            "edited_max_abs": edited_max_abs,
            "intended_delta_max_abs": float(delta.abs().max().item()),
            "observed_delta_max_abs": float(actual_delta.abs().max().item()),
            "delta_max_abs_error": float(delta_error.max().item()),
            "delta_absolute_tolerance": delta_tolerance,
            "output_formula_max_abs_error": float(expected_error.max().item()),
            "changed_fraction": float((actual_delta != 0).float().mean().item()),
        }
        call_records.append(record)

        if isinstance(output, tuple):
            return (edited,) + output[1:]
        return edited

    return hook_fn


def verify_layer(
    layer: int,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    tokenized: dict[str, torch.Tensor],
    vector_cpu: torch.Tensor,
) -> dict[str, Any]:
    module = model.model.layers[layer]
    baseline_hook_count = len(module._forward_hooks)
    call_records: list[dict[str, Any]] = []
    input_ids_before = tokenized["input_ids"].detach().clone()
    handle = None
    error: dict[str, str] | None = None
    generated_text = ""
    generated_token_ids: list[int] = []

    try:
        handle = module.register_forward_hook(make_hook(vector_cpu, call_records))
        with torch.inference_mode():
            generated = model.generate(
                **tokenized,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated_ids = generated[0, tokenized["input_ids"].shape[1] :]
        generated_token_ids = [int(value) for value in generated_ids.tolist()]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    except Exception as exc:  # Preserve exact failure evidence in the JSON.
        error = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        if handle is not None:
            handle.remove()

    hook_count_after = len(module._forward_hooks)
    input_ids_unchanged = bool(torch.equal(input_ids_before, tokenized["input_ids"]))
    prefill_seen = bool(call_records and call_records[0]["shape"][1] > 1)
    cached_decode_seen = any(record["shape"][1] == 1 for record in call_records[1:])
    widths_valid = bool(call_records) and all(
        record["shape"][-1] == HIDDEN_WIDTH for record in call_records
    )
    dtype_device_valid = bool(call_records) and all(
        record["dtype"] == "torch.float16" and record["device"].startswith("cuda")
        for record in call_records
    )
    numerical_check = bool(call_records) and all(
        record["hidden_finite"]
        and record["vector_finite"]
        and record["delta_finite"]
        and record["edited_finite"]
        and record["delta_max_abs_error"] <= record["delta_absolute_tolerance"]
        and record["output_formula_max_abs_error"] == 0.0
        and record["changed_fraction"] > 0.0
        for record in call_records
    )
    generation_valid = bool(generated_token_ids) and bool(generated_text.strip())
    hook_removed = hook_count_after == baseline_hook_count

    checks = {
        "no_exception": error is None,
        "hook_fired": bool(call_records),
        "prefill_seen": prefill_seen,
        "cached_decode_seen": cached_decode_seen,
        "hidden_width_valid": widths_valid,
        "dtype_device_compatible": dtype_device_valid,
        "numerical_check": numerical_check,
        "input_ids_unchanged": input_ids_unchanged,
        "generation_nonempty": generation_valid,
        "hook_removed": hook_removed,
    }
    supported = all(checks.values())
    return {
        "layer": layer,
        "supported": supported,
        "hook_location": f"model.model.layers[{layer}]",
        "hook_count_before": baseline_hook_count,
        "hook_count_after": hook_count_after,
        "checks": checks,
        "hook_calls": call_records,
        "generated_token_ids": generated_token_ids,
        "generated_text": generated_text,
        "error": error,
    }


def main() -> int:
    started = time.perf_counter()
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats()

    prompt = load_fixed_prompt()
    prompt_text = prompt["test_case_prompt"]
    vector_cpu = make_probe_vector()
    script_hash = sha256_file(SCRIPT_PATH)
    vector_hash = sha256_bytes(vector_cpu.contiguous().numpy().tobytes())
    model_snapshot = current_hf_snapshot(MODEL_ID)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )
    model.eval()
    input_device = model.get_input_embeddings().weight.device
    tokenized = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    )
    tokenized = {key: value.to(input_device) for key, value in tokenized.items()}

    layer_results = [
        verify_layer(layer, model, tokenizer, tokenized, vector_cpu)
        for layer in LAYERS
    ]
    layers_verified = [
        result["layer"] for result in layer_results if result["supported"]
    ]
    supported = layers_verified == list(LAYERS)
    elapsed = time.perf_counter() - started

    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": 3,
        "purpose": "dense residual-addition hook readiness technical smoke",
        "verification_status": "VERIFIED" if supported else "FAILED",
        "supported": supported,
        "layers_requested": list(LAYERS),
        "layers_verified": layers_verified,
        "wrapper_required": True,
        "hook_location": "model.model.layers[layer] forward hook",
        "vector_shape": [HIDDEN_WIDTH],
        "dtype": str(model.dtype),
        "device": str(input_device),
        "token_position_policy": {
            "policy": "all_positions_present_in_each_forward_call",
            "prefill": "add to every prompt position",
            "cached_decode": "add to the current one-token position on each decode call",
        },
        "operation": "edited = hidden + multiplier * dense_vector[None, None, :]",
        "model": {
            "id": MODEL_ID,
            "requested_revision": "main",
            "cache_snapshot": model_snapshot,
            "reported_commit_hash": getattr(model.config, "_commit_hash", None),
            "dtype": str(model.dtype),
            "device_map": "auto",
            "input_device": str(input_device),
            "hidden_size": int(model.config.hidden_size),
        },
        "tokenizer": {
            "id": MODEL_ID,
            "reported_commit_hash": tokenizer.init_kwargs.get("_commit_hash"),
            "raw_prompt": True,
            "chat_template_applied": False,
            "padding_side": tokenizer.padding_side,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "truncation": True,
            "max_input_tokens": MAX_INPUT_TOKENS,
            "input_token_count": int(tokenized["input_ids"].shape[1]),
        },
        "prompt": {
            "source": str(PROMPTS_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "prompt_id": PROMPT_ID,
            "text_sha256": sha256_bytes(prompt_text.encode("utf-8")),
            "cwe_identifier": prompt.get("cwe_identifier"),
            "language": prompt.get("language"),
        },
        "vector_kind": "synthetic_technical_probe",
        "vector_construction_status": "NOT_PERFORMED_PHASE3",
        "vector": {
            "construction": "float32 alternating +1,-1 beginning with +1",
            "seed": None,
            "shape": list(vector_cpu.shape),
            "vector_shape": HIDDEN_WIDTH,
            "dtype": str(vector_cpu.dtype),
            "l2_norm": float(torch.linalg.vector_norm(vector_cpu).item()),
            "sha256": vector_hash,
            "multiplier": MULTIPLIER,
            "delta_absolute_tolerance_policy": (
                "max(0.03125, torch.float16.eps * "
                "max(hidden_max_abs, edited_max_abs, 1.0))"
            ),
            "delta_absolute_tolerance_floor": DELTA_ATOL_FLOOR,
        },
        "numerical_check": {
            "required": (
                "finite tensors, exact agreement with the float16 hidden + delta "
                "operation, and max_abs((edited-hidden)-delta) within the "
                "magnitude-aware float16 rounding bound"
            ),
            "absolute_tolerance_policy": (
                "max(0.03125, torch.float16.eps * "
                "max(hidden_max_abs, edited_max_abs, 1.0))"
            ),
            "all_layers_passed": all(
                result["checks"]["numerical_check"] for result in layer_results
            ),
        },
        "generation": {
            "do_sample": False,
            "max_new_tokens": MAX_NEW_TOKENS,
            "layer_isolation": True,
        },
        "layer_results": layer_results,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
            ),
        },
        "execution": {
            "command": COMMAND,
            "exit_status": 0 if supported else 1,
            "elapsed_seconds": elapsed,
            "script_path": str(SCRIPT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "script_sha256": script_hash,
            "output_path": str(OUTPUT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
        "notes": [
            "This smoke verifies wiring only; the vector is not a CAA direction.",
            "No SAE was loaded or used.",
            "No held-out prompt, safety score, feature selection, or multiplier tuning was used.",
        ],
    }
    atomic_write_json(OUTPUT_PATH, payload)
    print(json.dumps({
        "supported": supported,
        "layers_verified": layers_verified,
        "output": str(OUTPUT_PATH),
        "elapsed_seconds": elapsed,
    }, indent=2))
    return 0 if supported else 1


if __name__ == "__main__":
    sys.exit(main())
