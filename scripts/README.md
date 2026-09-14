# Script index

The released scripts preserve the experiment's executable stage boundaries. Phase numbers remain in filenames to connect code with the corresponding configurations and results.

## Model 1

- `model1/core/phase0–phase4`: provenance, data preparation, baseline scanning, activation extraction, feature discovery, and causal validation.
- `model1/core/phase5`: static and prompting comparisons.
- `model1/core/phase6–phase7`: offline bandit construction, training, and validation.
- `model1/core/phase8–phase9`: configuration freeze and held-out evaluation.
- `model1/core/phase10–phase11`: utility and ablation studies.
- `model1/revision/phase12–phase19`: extended multi-seed, baseline, utility, timing, and statistical analyses.
- `model1/utility_routes`: routewise B\* utility evaluation.

The final multi-seed metric entry point is `model1/revision/phase16/compute_phase16_metrics.py`. Sensitivity and CWE-support summaries are produced by the scripts in `model1/revision/phase19/`.

## Model 2

Model 2 uses Gemma 2 9B Instruct and Gemma Scope SAEs. The corrected causal generation runner is `model2/run_model2_causal_steering_v2.py`; its frozen protocol is `configs/model2/model2_causal_execution_protocol_v3.json`. The superseded hook-signature runner is not released.

Static-strength results are generated and scanned with `run_model2_strength_generation.py` and `scan_model2_strength.py`, then finalized by `finalize_model2_strength.py`.

## Model 3

Model 3 uses Qwen 2.5 7B Instruct and BatchTopK SAEs. The sequence is represented directly by the Phase 22B–22H prefixes. The prospective held-out runner is `model3/run_phase22h_paired_evaluation.py`, and final metrics are produced by `model3/finalize_phase22h_evaluation.py`.

## Paper figures

The final figure scripts are in `paper/`. They read committed compact evidence and write publication-ready PDF and PNG files to the ignored `build/paper_figures/` directory. Set `CAR_FIGURE_OUTPUT_DIR` to use another output directory. The committed manuscript figures in `results/paper_figures/` remain immutable reference artifacts.

## Path convention

Run scripts from a reconstruction workspace that contains `configs/`, `data/`, and `outputs/` in the expected logical layout. Legacy Model 1 scripts that accept the compatibility variable can use `CAR_ROOT=/path/to/workspace`. Model 2 and Model 3 scripts generally expose explicit command-line paths through `argparse`.

Use `scripts/setup_workspace.py --workdir /path/to/car-workspace` to create this layout without copying any raw data or model weights.
