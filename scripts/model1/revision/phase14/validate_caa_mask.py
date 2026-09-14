#!/usr/bin/env python3
"""Numerically validate only the Phase 14 CAA position-mask adaptation."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any

from caa_position_mask import CAAResidualPositionHook
from phase14_common import (
    CAA_LAYERS, EXPECTED_SOURCE_RECOVERY_SHA256, OUTPUTS, SOURCE_RECOVERY,
    atomic_json, relative, require_hash, sha256_path, validate_preparation,
)


RUNNER = Path(__file__).resolve()
WRAPPER = RUNNER.with_name("caa_position_mask.py")
PHASE3 = RUNNER.parents[1].parent / "phase3/dense_hook_readiness.json"
EXPECTED_PHASE3_SHA256 = "75d1b494b2604d91aa36aaf04006bfd22c0f673c3317d0cf9d07dafc95e73855"
DEFAULT_OUTPUT = OUTPUTS / "dense_hook_readiness_revision.json"


def validate_layer(layer: int, torch: Any) -> dict[str, Any]:
    class TupleBlock(torch.nn.Module):
        def forward(self, value: Any) -> tuple[Any, str]:
            return value.clone(), "tuple-tail-sentinel"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    width = 8
    vector = torch.tensor([1.0, -1.0] * 4, dtype=torch.float32)
    multiplier = 0.125
    block = TupleBlock().to(device)
    before_hooks = len(block._forward_hooks)
    hook = CAAResidualPositionHook(vector, multiplier, expected_width=width)
    handle = block.register_forward_hook(hook)
    error = None
    checks: dict[str, bool] = {}
    observations: dict[str, Any] = {}
    try:
        prefill = torch.arange(1 * 5 * width, dtype=dtype, device=device).reshape(1, 5, width) / 16
        prefill_before = prefill.clone()
        prefill_out, tail = block(prefill)
        delta = (multiplier * vector).to(device=device, dtype=dtype)
        checks["tuple_tail_preserved"] = tail == "tuple-tail-sentinel"
        checks["prefill_earlier_positions_unchanged"] = bool(torch.equal(prefill_out[:, :-1], prefill_before[:, :-1]))
        checks["prefill_final_prompt_position_added"] = bool(torch.equal(prefill_out[:, -1], prefill_before[:, -1] + delta))
        checks["hook_inputs_unchanged"] = bool(torch.equal(prefill, prefill_before))

        cached = torch.ones((1, 1, width), dtype=dtype, device=device)
        cached_before = cached.clone()
        cached_out, _ = block(cached)
        checks["cached_current_position_added"] = bool(torch.equal(cached_out, cached_before + delta.view(1, 1, width)))

        uncached = torch.ones((1, 7, width), dtype=dtype, device=device)
        uncached_before = uncached.clone()
        uncached_out, _ = block(uncached)
        checks["uncached_earlier_prompt_positions_unchanged"] = bool(torch.equal(uncached_out[:, :4], uncached_before[:, :4]))
        checks["uncached_post_prompt_positions_added"] = bool(torch.equal(uncached_out[:, 4:], uncached_before[:, 4:] + delta.view(1, 1, width)))
        checks["shape_preserved"] = tuple(uncached_out.shape) == tuple(uncached.shape)
        checks["dtype_preserved"] = uncached_out.dtype == dtype
        checks["finite"] = bool(torch.isfinite(prefill_out).all() and torch.isfinite(cached_out).all() and torch.isfinite(uncached_out).all())
        observations = {"device": str(device), "dtype": str(dtype), "call_records": hook.call_records}
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        handle.remove()
    checks["hook_removed"] = len(block._forward_hooks) == before_hooks

    identity_input = torch.arange(40, dtype=dtype, device=device).reshape(1, 5, width) / 32
    checks["inactive_identity"] = bool(torch.equal(block(identity_input)[0], identity_input))

    zero_vector_hook = CAAResidualPositionHook(torch.zeros(width), 1.0, expected_width=width)
    zero_handle = block.register_forward_hook(zero_vector_hook)
    zero_vector_output = block(identity_input)[0]
    zero_handle.remove()
    checks["zero_vector_identity"] = bool(torch.equal(zero_vector_output, identity_input))

    zero_multiplier_hook = CAAResidualPositionHook(vector, 0.0, expected_width=width)
    zero_multiplier_handle = block.register_forward_hook(zero_multiplier_hook)
    zero_multiplier_output = block(identity_input)[0]
    zero_multiplier_handle.remove()
    checks["zero_multiplier_identity"] = bool(torch.equal(zero_multiplier_output, identity_input))
    checks["no_hook_leak_after_identity_tests"] = len(block._forward_hooks) == before_hooks
    passed = error is None and all(checks.values())
    return {"layer": layer, "status": "PASS" if passed else "FAIL", "supported": passed,
            "checks": checks, "observations": observations, "error": error}


def main() -> int:
    import torch

    validate_preparation()
    require_hash(PHASE3, EXPECTED_PHASE3_SHA256, "Phase 3 dense-hook readiness")
    phase3 = json.loads(PHASE3.read_text(encoding="utf-8"))
    if phase3.get("verification_status") != "VERIFIED" or phase3.get("supported") is not True:
        raise RuntimeError("Phase 3 dense-hook readiness is not VERIFIED")
    results = [validate_layer(layer, torch) for layer in CAA_LAYERS]
    passed = all(result["supported"] for result in results)
    payload = {
        "schema_version": "phase14_dense_hook_readiness_revision_v1",
        "phase": 14, "purpose": "minimal CAA after-prompt position-mask numerical validation",
        "verification_status": "PASS" if passed else "FAIL", "supported": passed,
        "layers_requested": list(CAA_LAYERS),
        "layers_verified": [result["layer"] for result in results if result["supported"]],
        "source_linkage": {
            "source_recovery_path": relative(SOURCE_RECOVERY),
            "source_recovery_sha256": EXPECTED_SOURCE_RECOVERY_SHA256,
            "recovered_elements": ["application_token_positions", "hook_injection_semantics"],
            "phase3_readiness_path": relative(PHASE3), "phase3_readiness_sha256": EXPECTED_PHASE3_SHA256,
        },
        "semantics": {
            "prefill": "final prompt position inclusively; earlier prompt positions unchanged",
            "cached_decode": "current one-token position",
            "uncached_decode": "positions from frozen final-prompt index onward",
            "operation": "hidden + multiplier * dense_vector at masked positions only",
        },
        "layer_results": results,
        "execution_boundary": {"model_loaded": False, "generation_executed": False,
                               "vector_construction_executed": False, "security_outcomes_evaluated": False,
                               "held_out_accessed": False, "scanner_executed": False},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "cuda_available": torch.cuda.is_available(),
                        "test_device": "cuda:0" if torch.cuda.is_available() else "cpu"},
        "wrapper": {"path": relative(WRAPPER), "sha256": sha256_path(WRAPPER)},
        "validator": {"path": relative(RUNNER), "sha256": sha256_path(RUNNER)},
    }
    atomic_json(DEFAULT_OUTPUT, payload)
    print(json.dumps({"status": payload["verification_status"], "layers": payload["layers_verified"],
                      "output_sha256": sha256_path(DEFAULT_OUTPUT)}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

