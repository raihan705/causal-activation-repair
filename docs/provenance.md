# Model and SAE provenance

## Model 1: primary analysis

- Model: `meta-llama/Meta-Llama-3.1-8B-Instruct`
- Model revision: `0e9e39f249a16976918f6564b8830bc894c89659`
- SAE Lens release: `llama_scope_lxr_8x`
- SAE IDs: `l16r_8x`, `l19r_8x`, and `l23r_8x`
- Recorded SAE snapshot: `8dbc1d85edfced43081c03c38b05514dbab1368b`
- Hidden dimension: 4,096
- SAE dimension: 32,768

The frozen route map is in `configs/model1/bstar_config.json`. The selected intervention uses CWE-routed single sparse coordinates with alpha 40.

## Model 2: Gemma replication

- Model: `google/gemma-2-9b-it`
- Model revision: `11c9b309abf73637e4b6f9a3fa1e92e615547819`
- SAE repository: `google/gemma-scope-9b-it-res`
- SAE revision: `e86af97a5b6fbbccca28ab654f2fda1b0768f770`
- Candidate SAE IDs:
  - `layer_9/width_16k/average_l0_47`
  - `layer_20/width_16k/average_l0_47`
  - `layer_31/width_16k/average_l0_43`

The final selected causal routes use layer 20. Their identifiers are stored in `artifacts/model2/selected_routes/model2_selected_routes.json`.

## Model 3: Qwen replication

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Model revision: `a09a35458c702b33eeacc393d103063234e8bc28`
- SAE repository: `andyrdt/saes-qwen2.5-7b-instruct`
- SAE revision: `c37e53c4bb07127ad17ab88f28b93d4e87142e59`
- SAE architecture: BatchTopK, dictionary size 131,072, `k=32`
- Candidate layers: 7, 15, and 23
- Loader provenance: `andyrdt/dictionary_learning` at `de9138b02fffdf9919c53cee828beb4e05049741`

Model 3 uses direct addition of the selected unit-norm decoder direction. The prospective route map is in `artifacts/model3/selected_routes/phase22f_selected_routes.json`.

## Integrity convention

Every distributed file is covered by `manifests/SHA256SUMS`. External weights remain identified by repository revision and, where recorded, individual weight hashes in the corresponding protocol manifest.
