# Scanner protocol

Semgrep with registry configuration `p/security-audit` is the primary detector used by the released analyses. Scans were executed under Linux because the original workflow observed unreliable behavior on Windows.

## Core rules

- Do not infer vulnerability from a dataset CWE label alone.
- A baseline target-CWE finding must be observed before an output can count as repaired.
- Valid generation alone is not repair.
- Scanner-skipped records are not treated as safe.
- A different CWE cannot serve as a proxy for an unsupported target CWE.
- Repair, corruption, validity, target regression, and other findings use their protocol-specific denominators.

The executable scanner wrappers are included with the relevant model workflow. The protocol manifests define the expected input schema, Semgrep version, scanner settings, and result hashes.

## Version scope

- Model 1's original environment recorded Semgrep 1.154.0.
- Final cross-model protocols record Semgrep 1.175.0.

Use the recorded version for the experiment being reproduced.

## Registry limitation

The remote `p/security-audit` registry was not retained as a complete content-addressed snapshot for every experiment stage. The repository therefore supports exact verification of the saved scan artifacts and wrapper behavior, but a future registry download may resolve to changed rule content. This limitation must remain visible in any exact-reproduction claim.
