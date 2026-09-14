#!/usr/bin/env python3
"""Run the frozen Phase21 causal validator with the returned manifest binding."""

from __future__ import annotations

import finalize_model2_causal_validation_core as core


core.EXPECTED[core.RETURN] = "f32f77b62140e2d104a41f43fbc614f44c1114c312ab089fbdb4d522a4d4ac62"


if __name__ == "__main__":
    raise SystemExit(core.main())
