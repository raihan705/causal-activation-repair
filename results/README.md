# Result artifacts

Only compact study evidence is included. Raw generations, prompt-bearing scanner inputs, model checkpoints, activation tensors, and transfer bundles are excluded.

- `model1/` contains the scanner audit, reconstruction and causal summaries, development selection, multi-seed security results, utility results, intervention-boundary results, and timing evidence.
- `model2/` contains Gemma reconstruction, causal-validation, denominator, and prospective static-strength results.
- `model3/` contains Qwen reconstruction, causal-validation, calibration, and prospective held-out results.
- `statistics/` contains the consolidated Model 1 statistical and sensitivity manifests.
- `paper_tables/` contains available figure/table source data.
- `paper_figures/` contains figures regenerated from the released scripts and data.

The scanner-primary machine metrics remain distinct from any secondary manual construct review. `construct_review_summary.json` provides the code-free aggregate judgments reported in the manuscript; manual-review worksheets and source-bearing changed-case material are not redistributed.

Canonical reporting sources are identified in [`../docs/reporting_provenance.md`](../docs/reporting_provenance.md).
