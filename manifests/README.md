# Release inventories

This directory contains two complementary records:

- `release_manifest.json` inventories every distributed file except itself and `SHA256SUMS`, with its SHA-256 digest and size.
- `SHA256SUMS` is the corresponding checksum list for command-line verification.
- `excluded_materials.csv` documents material intentionally not redistributed and gives the corresponding reconstruction or access route.

Both records describe the public distribution itself and are covered by the release checksums.
