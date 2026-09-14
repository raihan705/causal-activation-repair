# Reporting provenance

Manuscript values are reported from canonical result artifacts rather than from performance fields in execution configurations. The public mapping is recorded in `results/model1/model1_reporting_provenance.json`.

## Model 1

- `results/model1/scanner_audit_summary.json` reports the 50-decision scanner audit and the 16% false-positive rate.
- `results/model1/development_method_selection.csv` reports the paired development comparison used to select B\*. Its denominators distinguish 60 B0-positive records, 863 scanner-observed B0-safe records, and all 1,341 prompts used for validity.
- `results/model1/sae_reconstruction_summary.csv` and `results/model1/layer_sensitivity_metrics.csv` report reconstruction and perturbation sensitivity.
- `results/model1/causal_revalidation_summary.csv` reports the four scanner-evaluable routes and 29/29 qualified target-finding removals.
- `results/model1/strength_sweep_overall.csv` and `results/model1/intervention_boundary_summary.json` report strength and design-boundary analyses.
- `results/model1/phase16_per_seed_results.csv`, `phase16_aggregate_results.csv`, and `phase16_pairwise_intervals.csv` report the held-out estimates. B0, B1, and B\* form the primary repeated-seed comparison. B2-alpha20 is a descriptive strength comparison.

## Cross-model replications

- `results/model2/model2_causal_*` and `model2_strength_*` report the Gemma causal and static-strength analyses.
- `results/model3/phase22e_*`, `phase22f_*`, and `phase22h_*` report the Qwen causal, calibration, and prospective held-out analyses.

## Construct review

`results/construct_review_summary.json` contains code-free aggregate judgments for the changed-case analyses reported in the manuscript. These judgments do not replace the scanner labels or alter the scanner-primary metrics.
