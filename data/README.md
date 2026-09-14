# Data directory

This directory contains record identities, schemas, and hashes only. It does not contain benchmark prompts, source/fixed code, raw databases, generated code, or model activations.

- `source_manifest.json` records upstream sources and revisions.
- `dataset_construction_summary.json` records the manuscript's accepted construction and split counts, together with hashes of the non-redistributed processed sources.
- `split_ids/cyberseceval_split_manifest.json` records the exact development and held-out prompt IDs.
- `split_ids/study_split_manifest.json` records the frozen split structure.
- `split_ids/cve_pair_split_metadata.csv` records CVE-pair metadata and code hashes without redistributing code.
- `schemas/cve_pair_metadata.schema.json` defines the released metadata columns.

See `docs/data_access.md` for reconstruction instructions and licensing boundaries.
