# Frozen artifacts

This directory contains the compact artifacts that define the evaluated interventions:

- the Model 1 validated features and feature summaries;
- all 12 compact offline-bandit checkpoints together with their action/state definitions and aggregate validation metrics;
- the prospectively selected Model 2 routes;
- the prospectively selected Model 3 routes and freeze verification.

Model and SAE weight files are not included. Obtain them at the exact revisions in `docs/provenance.md`.

The frozen Model 1 B\* route map, feature groups, and complete intervention library are stored once in `configs/model1/`, which is also their executable runtime location.

The `.ckpt` files contain PyTorch state dictionaries and compact training metadata. Load downloaded checkpoint files with `torch.load(..., weights_only=True)` and verify their release hashes before use.
