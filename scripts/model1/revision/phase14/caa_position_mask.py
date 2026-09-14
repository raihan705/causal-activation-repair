#!/usr/bin/env python3
"""Source-faithful dense CAA residual hook for the DSAR raw-prompt path.

Source linkage: `baseline_source_recovery.json`, CAA-CWE elements
`application_token_positions` and `hook_injection_semantics`, recovered from
Rimsky et al. ACL 2024 and the official nrimsky/CAA implementation.

The first forward call is treated as prompt prefill and edits only the final
prompt token.  Cached one-token calls edit that current token.  A later
uncached full-sequence call edits positions beginning at the frozen prompt-end
index.  Earlier prompt positions are never changed.
"""

from __future__ import annotations

from typing import Any


class CAAResidualPositionHook:
    """Stateful forward hook implementing final-prompt-inclusive CAA masking."""

    def __init__(self, vector: Any, multiplier: float, expected_width: int | None = None):
        self.vector = vector
        self.multiplier = float(multiplier)
        self.expected_width = expected_width
        self.prompt_length: int | None = None
        self.call_records: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.prompt_length = None
        self.call_records.clear()

    def _mask(self, hidden: Any) -> Any:
        import torch

        batch, sequence, _ = hidden.shape
        if self.prompt_length is None:
            self.prompt_length = int(sequence)
            mask = torch.zeros((batch, sequence, 1), dtype=torch.bool, device=hidden.device)
            mask[:, sequence - 1 :, :] = True
            policy = "PREFILL_FINAL_PROMPT_POSITION_ONLY"
        elif sequence == 1:
            mask = torch.ones((batch, 1, 1), dtype=torch.bool, device=hidden.device)
            policy = "CACHED_CURRENT_TOKEN"
        else:
            start = self.prompt_length - 1
            if sequence < self.prompt_length:
                raise ValueError("uncached sequence became shorter than frozen prompt length")
            mask = torch.zeros((batch, sequence, 1), dtype=torch.bool, device=hidden.device)
            mask[:, start:, :] = True
            policy = "UNCACHED_FROM_FINAL_PROMPT_POSITION"
        return mask, policy

    def __call__(self, module: Any, inputs: tuple[Any, ...], output: Any) -> Any:
        import torch

        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.ndim != 3:
            raise ValueError(f"CAA hook expects [batch, sequence, width], got {tuple(hidden.shape)}")
        width = int(hidden.shape[-1])
        if self.expected_width is not None and width != self.expected_width:
            raise ValueError(f"CAA vector expects width {self.expected_width}, got {width}")
        vector = self.vector.to(device=hidden.device, dtype=hidden.dtype)
        if tuple(vector.shape) != (width,):
            raise ValueError(f"CAA vector shape {tuple(vector.shape)} cannot target width {width}")
        if not torch.isfinite(vector).all():
            raise ValueError("CAA vector contains NaN/Inf")

        mask, policy = self._mask(hidden)
        delta = self.multiplier * vector.view(1, 1, width)
        edited = torch.where(mask, hidden + delta, hidden)
        if not torch.isfinite(edited).all():
            raise ValueError("CAA injection introduced NaN/Inf")
        self.call_records.append({
            "call_index": len(self.call_records), "shape": list(hidden.shape),
            "dtype": str(hidden.dtype), "device": str(hidden.device), "policy": policy,
            "masked_position_count_per_batch": int(mask[0, :, 0].sum().item()),
            "multiplier": self.multiplier, "tuple_output": isinstance(output, tuple),
        })
        if isinstance(output, tuple):
            return (edited,) + output[1:]
        return edited

