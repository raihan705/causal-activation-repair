# Causal Activation Repair

Research artifacts for **Causal Activation Repair for Reducing CWE-Level Vulnerabilities in LLM-Generated Code**.

This repository contains the code, frozen intervention definitions, compact checkpoints, protocol manifests, and machine-readable results needed to inspect and reproduce the study. Large model weights, SAE weights, third-party benchmark prompts, raw vulnerability databases, activation tensors, and prompt-bearing generation files are not redistributed. Exact upstream identifiers and reconstruction instructions are provided instead.

## Study organization

- **Model 1 — Llama:** complete feature-discovery, causal-validation, method-comparison, held-out, utility, adaptive-control, and timing pipeline.
- **Model 2 — Gemma:** independent feature discovery and causal replication, followed by prospective static-strength evaluation.
- **Model 3 — Qwen:** independent discovery, causal validation, development-only strength calibration, and prospective held-out evaluation.

The three model–SAE systems are heterogeneous replications. Their feature identities and effect sizes are not pooled as if they were interchangeable coordinates.

## Repository contents

| Directory | Contents |
|---|---|
| `scripts/` | Experiment and analysis entry points |
| `configs/` | Frozen model, scanner, generation, selection, and intervention definitions |
| `data/` | Dataset source information, schemas, split IDs, and hashes—no source code or prompt text |
| `artifacts/` | Validated features and routes, plus compact bandit checkpoints |
| `results/` | Compact results and statistical summaries |
| `docs/` | Data access, provenance, scanner, artifact, and reproduction documentation |
| `manifests/` | Distribution scope and SHA-256 release inventory |

## Quick verification

Create a Python 3.11 environment, install the analysis dependencies, and validate the release integrity and canonical reported results:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r environment/requirements-analysis.txt
python tests/validate_release.py
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

Full model execution requires a CUDA-capable system, acceptance of any applicable model-provider terms, and separate installation of the appropriate GPU build of PyTorch. See [`docs/reproduction.md`](docs/reproduction.md).

## Data access

The study data are reconstructed from their upstream sources. The release does not include:

- the CVEFixes database or source/fixed code fields;
- CyberSecEval, HumanEvalPack, BigCodeBench, or MMLU prompt text;
- model or SAE weights;
- raw or encoded activation tensors.

The split manifests preserve record identities and counts. CVE-pair metadata can be regenerated with hashes after the source dataset has been reconstructed. See [`docs/data_access.md`](docs/data_access.md).

The complete inclusion boundary and reconstruction path for omitted material are recorded in [`manifests/excluded_materials.csv`](manifests/excluded_materials.csv).

## Reproducibility scope

Three reproducibility levels are distinguished:

1. **Result inspection:** use the committed CSV/JSON summaries to inspect reported denominators, metrics, intervals, routes, and ablations.
2. **Analysis reproduction:** recompute tables and statistical summaries from reconstructed scanner outputs.
3. **End-to-end reproduction:** reacquire external datasets, models, and SAEs; regenerate outputs; scan them; and recompute metrics.

The repository provides executable scripts, frozen configurations, input/output identity hashes where available, and public reconstruction commands. Canonical manuscript values and their public evidence files are identified in [`docs/reporting_provenance.md`](docs/reporting_provenance.md).

## Security and credentials

Never commit Hugging Face tokens, GitHub tokens, cached credentials, model caches, or dataset caches. Use environment variables or the Hugging Face CLI for authentication.

## License status

The authors must select a license for the original study code before making this repository public. Third-party datasets, models, SAEs, scanners, and source-code records remain governed by their upstream licenses and terms.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). Replace the repository URL, DOI, and publication metadata after archival publication.
