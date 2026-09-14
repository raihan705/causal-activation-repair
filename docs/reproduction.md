# Reproduction guide

## 1. Verify the compact release

From the repository root:

```bash
python -m pip install -r environment/requirements-analysis.txt
python tests/validate_release.py
```

This checks release hashes, JSON/CSV readability, excluded-file policy, and accidental credential or local-path leakage.

## 2. Reconstruct external data

Follow `docs/data_access.md`. Keep downloaded data outside version control. Reconstructed paths should be placed beneath a local working root with the same logical names used by the scripts, or supplied through the scripts' command-line path arguments.

Create the expected logical layout in a dedicated directory:

```bash
python scripts/setup_workspace.py --workdir /path/to/car-workspace
export CAR_ROOT=/path/to/car-workspace
```

The setup command copies only released code and configuration. It does not download data or weights and does not write into the release repository.

To create a metadata-only identity table from reconstructed CVE pair files:

```bash
python scripts/data/export_split_metadata.py \
  --train /path/to/train_pairs.csv \
  --validation /path/to/val_pairs.csv \
  --test /path/to/test_pairs.csv \
  --output data/split_ids/cve_pair_split_metadata.csv
```

## 3. Install model dependencies

Install the appropriate CUDA build of PyTorch and then:

```bash
python -m pip install -r environment/requirements-models.txt
```

Authenticate separately for gated assets. Do not store credentials in configuration files.

Utility evaluation can execute model-generated programs. Run HumanEval or BigCodeBench evaluators only in an isolated container or disposable environment without credentials or access to sensitive files.

## 4. Model 1 workflow

The primary pipeline is preserved in `scripts/model1/core/phase0` through `phase11`. Extended evaluation scripts are preserved in `scripts/model1/revision/`, and routewise utility scripts are in `scripts/model1/utility_routes/`.

The high-level order is:

1. acquire and construct CVE pairs;
2. construct and scan CyberSecEval development data;
3. extract hidden activations and evaluate SAE reconstruction;
4. rank and causally validate sparse features;
5. evaluate static, prompting, dense, grouped, and adaptive alternatives;
6. freeze the selected route map;
7. perform held-out multi-seed evaluation;
8. evaluate utility, ablations, and timing;
9. consolidate statistical evidence.

Use `CAR_ROOT` to point legacy Model 1 scripts to a reconstruction workspace when necessary.

## 5. Model 2 workflow

Run the scripts in `scripts/model2/` in the following order:

1. compatibility and reconstruction;
2. B0 development generation and scanning;
3. source activation extraction and feature ranking;
4. paired causal baseline qualification;
5. corrected causal steering (`run_model2_causal_steering_v2.py`) and validation;
6. prospective route selection and strength evaluation;
7. denominator and result verification.

The frozen execution protocol is `configs/model2/model2_causal_execution_protocol_v3.json`.

## 6. Model 3 workflow

The public sequence is represented by the Phase 22A–22H scripts in `scripts/model3/`:

1. provenance and compatibility;
2. B0 support measurement;
3. activation extraction and discovery;
4. live causal validation;
5. development-only strength calibration;
6. prospective protocol freeze;
7. paired held-out generation, scanning, and analysis.

The Model 3 prospective protocol freezes the route map, population, seeds, generation settings, denominators, and statistics before held-out execution.

## 7. Scanner execution

Install the protocol-specific Semgrep requirement and run scans under Linux. Follow `docs/scanner_protocol.md`; do not change the scanner or repair denominator after observing outcomes.

## 8. Paper artifacts

The compact results are sufficient to inspect the paper-facing values without model execution. Figure-generation scripts are in `scripts/paper/`, while their source results are in `results/`. Run the scripts from the repository root:

```bash
python scripts/paper/plot_fig3_cross_cwe_recurrence.py
python scripts/paper/plot_fig5_bstar_per_cwe_profile.py
python scripts/paper/plot_fig7_intervention_boundaries.py
```

Generated files are written to the ignored `build/paper_figures/` directory, so reproducing a figure does not overwrite the committed reference artifact or invalidate the release checksums.
