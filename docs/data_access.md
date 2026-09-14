# Data access and reconstruction

## Release policy

This repository does not redistribute third-party source code, vulnerability databases, benchmark prompts, model weights, or SAE weights. It provides source identifiers, deterministic split identities, schemas, hashes, preprocessing code, and compact derived evidence.

## CVEFixes-derived pairs

1. Clone `https://github.com/secureIT-project/CVEFixes` at commit `9283b50b3f04e3c5b0a17fc419ab0feea23fc438`.
2. Follow the upstream instructions to obtain/build the CVEFixes database.
3. Run the scripts in `scripts/model1/core/phase1/` in the order described in `reproduction.md`.
4. Validate the reconstructed split against `data/split_ids/cve_pair_split_metadata.csv`.

The source metadata record database SHA-256 `96ebfabb6ebd4960539e73c0840d74607d56e8d18465ba97cdbe1898b7fc30f5`. This value identifies the source used by the study; it is not presented as an independently repeated integrity measurement of the complete 20 GB database.

The released metadata omit `vulnerable_code` and `fixed_code`. Their SHA-256 values allow a locally reconstructed row to be checked without redistributing the code.

The CVE split unit was the effective CVE identifier. The audit also found exact code-pair duplicates across nominal splits, so the release does not claim complete content-level isolation.

## CyberSecEval

Acquire the dataset from `walledai/CyberSecEval`. The study used 1,916 records, split deterministically with seed 42 into 1,341 development records and 575 held-out records. The exact prompt IDs are stored in `data/split_ids/cyberseceval_split_manifest.json`.

Prompt text is not redistributed. After downloading the source, reconstruct the two sets by prompt ID and verify their counts and zero ID overlap. Protocol artifacts retain source IDs and SHA-256 values rather than prompt text.

## Utility benchmarks

Acquire the following datasets from their upstream repositories:

- `bigcode/humanevalpack`
- `bigcode/bigcodebench`
- `cais/mmlu`

The MMLU audit recorded revision `c30699e8356da336a370243923dbaf21066bb9fe` and evaluated the computer-science/security-related slices specified in the utility scripts.

## Model and SAE assets

Weights are obtained from Hugging Face or through SAE Lens. Authenticate with `huggingface-cli login` or a process-scoped `HF_TOKEN`; never place a token in the repository.

Model and SAE revisions are listed in `docs/provenance.md`. Some Llama and Gemma assets require acceptance of provider terms.

## Redistribution boundaries

The release makes no new license claim over upstream data. The CVEFixes repository license does not by itself establish redistribution rights for code copied from every underlying project. The same principle applies to benchmark prompts and model-generated records that embed prompt text.
