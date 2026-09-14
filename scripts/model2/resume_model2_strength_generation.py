#!/usr/bin/env python3
"""Resume the frozen strength runner after its top-level multi-alpha preflight bug.

The underlying generation runner is imported without modification. This wrapper
validates the frozen inputs and exact alpha-20 checkpoint, then invokes its
existing resume implementation. It does not change generation semantics.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


RUNNER = Path(__file__).with_name("run_model2_strength_generation.py")
EXPECTED_RUNNER_SHA256 = "dfb82ec6ce8fa40361afde75f410d7f5ca47fc3a307d6c4f789fc37cb0dae127"


def main() -> None:
    spec = importlib.util.spec_from_file_location("frozen_model2_strength_runner", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load frozen strength runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.require(module.sha256_file(RUNNER) == EXPECTED_RUNNER_SHA256, "Frozen generation runner hash changed")
    _, population, _ = module.load_and_validate_frozen()
    module.require(not module.validate_complete_condition(20, population), "Alpha20 is already complete; wrapper is no longer required")
    records, elapsed = module.load_checkpoint(20, population, True)
    module.require(0 < len(records) < 180, "Alpha20 resumable checkpoint cardinality invalid")
    module.require(not module.output_path(40).exists() and not module.manifest_path(40).exists(), "Unexpected alpha40 final artifact")
    print(json.dumps({
        "schema_version": "phase21_model2_strength_resume_wrapper_preflight_v1",
        "status": "PASS",
        "frozen_runner_sha256": EXPECTED_RUNNER_SHA256,
        "alpha20_checkpoint_records": len(records),
        "alpha20_checkpoint_elapsed_seconds": elapsed,
        "generation_settings_changed": False,
        "routes_changed": False,
        "population_changed": False,
        "heldout_accessed": False,
        "model1_modified": False,
    }, indent=2), flush=True)
    module.run(True)


if __name__ == "__main__":
    main()
