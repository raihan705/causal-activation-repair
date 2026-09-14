# Public artifact map

| Paper evidence | Principal public artifact |
|---|---|
| Model/SAE identities and reconstruction | `docs/provenance.md` and model-specific compatibility results |
| CyberSecEval split | `data/split_ids/cyberseceval_split_manifest.json` |
| CVE-pair construction and split | `data/dataset_construction_summary.json` and `data/split_ids/cve_pair_split_metadata.csv` |
| Scanner audit | `results/model1/scanner_audit_summary.json` |
| Model 1 reconstruction and layer sensitivity | `results/model1/sae_reconstruction_summary.csv` and `results/model1/layer_sensitivity_metrics.csv` |
| Model 1 feature discovery | `artifacts/model1/selected_features/` |
| Model 1 corrective causal validation | `results/model1/causal_revalidation_summary.csv` |
| Frozen B\* route map | `configs/model1/bstar_config.json` |
| Full intervention library | `configs/model1/intervention_library.json` |
| Model 1 development selection | `results/model1/development_method_selection.csv` |
| Static-method and multi-seed results | `results/model1/phase16_*` |
| Strength and intervention boundaries | `results/model1/strength_sweep_overall.csv` and `results/model1/intervention_boundary_summary.json` |
| Utility and routewise utility | `results/model1/*utility*` |
| Timing and resource cost | `results/model1/phase18_*` |
| Adaptive-control checkpoint evidence | `artifacts/model1/bandit_checkpoints/` |
| Consolidated sensitivity/statistical evidence | `results/statistics/` |
| Gemma causal replication | `results/model2/model2_causal_*` |
| Gemma static-strength boundary | `results/model2/model2_strength_*` |
| Qwen causal replication | `results/model3/phase22e_*` |
| Qwen prospective held-out evaluation | `results/model3/phase22h_*` |
| Changed-case construct review | `results/construct_review_summary.json` |
| Figure-generation code | `scripts/paper/` |

All distributed artifacts are indexed with sizes and SHA-256 digests in `manifests/release_manifest.json`.
