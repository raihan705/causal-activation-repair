"""
Figure 3 — Cross-CWE Feature Sharing Matrix
Academic dot-matrix version for IST-style paper figures.

Input:
    phase4_causal_validation.csv

Output:
    fig3_cross_cwe_feature_matrix.pdf
    fig3_cross_cwe_feature_matrix.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


# =============================================================================
# Paths
# =============================================================================
INPUT_CSV = (Path.cwd() / "outputs/phase4/phase4_causal_validation.csv")

OUTPUT_DIR = (Path.cwd() / "outputs/phase4/figs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PDF = OUTPUT_DIR / "fig3_cross_cwe_feature_matrix.pdf"
OUTPUT_PNG = OUTPUT_DIR / "fig3_cross_cwe_feature_matrix.png"


# =============================================================================
# Publication style
# =============================================================================
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.5,
})


# =============================================================================
# Load data
# =============================================================================
df = pd.read_csv(INPUT_CSV)

if df["validated"].dtype != bool:
    df["validated"] = (
        df["validated"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y"])
    )

validated = df[df["validated"]].copy()


# =============================================================================
# CWE order
# =============================================================================
CYBER_CWES = ["CWE-120", "CWE-327", "CWE-89", "CWE-338"]
CVE_CWES = ["CWE-79", "CWE-125", "CWE-787", "CWE-190", "CWE-476"]
CWE_ORDER = CYBER_CWES + CVE_CWES


# =============================================================================
# Select shared features: validated in >= 3 CWEs
# =============================================================================
shared = (
    validated
    .groupby(["layer", "feature_id"], as_index=False)["cwe"]
    .nunique()
    .rename(columns={"cwe": "n_cwes"})
)

shared = shared[shared["n_cwes"] >= 3].copy()

layer_priority = {19: 0, 16: 1, 23: 2}
shared["layer_priority"] = shared["layer"].map(layer_priority).fillna(9)

shared = shared.sort_values(
    by=["n_cwes", "layer_priority", "feature_id"],
    ascending=[False, True, True]
).reset_index(drop=True)

shared["row_label"] = shared.apply(
    lambda r: f"L{int(r['layer'])}-F{int(r['feature_id'])}",
    axis=1
)


# =============================================================================
# Build matrix
# =============================================================================
rows = []

for _, row in shared.iterrows():
    feature_rows = validated[
        (validated["layer"] == row["layer"]) &
        (validated["feature_id"] == row["feature_id"])
    ]

    validated_cwes = set(feature_rows["cwe"].tolist())

    rows.append({
        "Feature": row["row_label"],
        "n_cwes": row["n_cwes"],
        **{cwe: 1 if cwe in validated_cwes else 0 for cwe in CWE_ORDER}
    })

matrix_df = pd.DataFrame(rows).set_index("Feature")
n_cwes = matrix_df["n_cwes"].astype(int)
matrix = matrix_df[CWE_ORDER]

n_rows, n_cols = matrix.shape


# =============================================================================
# Plot dot matrix
# =============================================================================
fig_width = 6.8
fig_height = max(2.4, 0.32 * n_rows + 1.15)

fig, ax = plt.subplots(figsize=(fig_width, fig_height))

# Light background grid points for all possible cells
for i in range(n_rows):
    for j in range(n_cols):
        ax.scatter(
            j, i,
            s=34,
            facecolors="none",
            edgecolors="#d0d0d0",
            linewidths=0.6,
            zorder=1
        )

# Filled dots for validated feature-CWE pairs
for i in range(n_rows):
    for j in range(n_cols):
        if matrix.iloc[i, j] == 1:
            ax.scatter(
                j, i,
                s=58,
                color="#2F5D8C",
                edgecolors="#1F3F60",
                linewidths=0.35,
                zorder=3
            )

# Horizontal guide lines: stop before the summary count column
for i in range(n_rows):
    ax.plot(
        [-0.55, n_cols - 0.35],
        [i, i],
        color="#eeeeee",
        linewidth=0.45,
        zorder=0
    )

# Vertical separator between evidence sources
sep = len(CYBER_CWES) - 0.5
ax.axvline(sep, color="#666666", linestyle="--", linewidth=0.75, zorder=2)

# Light divider before the summary count column
ax.axvline(
    n_cols - 0.05,
    color="#b5b5b5",
    linewidth=0.55,
    linestyle="-",
    zorder=1
)

# Axis ticks and labels
ax.set_xticks(np.arange(n_cols))
ax.set_xticklabels(CWE_ORDER, rotation=35, ha="right", rotation_mode="anchor")

ax.set_yticks(np.arange(n_rows))
ax.set_yticklabels(matrix.index)

ax.invert_yaxis()

ax.set_xlabel("Studied CWE category", labelpad=10)
ax.set_ylabel("SAE latent feature", labelpad=8)

ax.tick_params(axis="both", length=0)

# Add right-side count column
for i, count in enumerate(n_cwes):
    ax.text(
        n_cols + 0.35,
        i,
        str(count),
        va="center",
        ha="center",
        fontsize=7,
        color="#333333"
    )

# Summary count header
ax.text(
    n_cols + 0.35,
    -0.58,
     "Validated\nCWEs",
    va="bottom",
    ha="center",
    fontsize=7,
    color="#333333"
)

# Source-group labels placed close to matrix
trans = ax.get_xaxis_transform()

ax.text(
    (len(CYBER_CWES) - 1) / 2,
    1.015,
    "CyberSecEval-detected",
    transform=trans,
    ha="center",
    va="bottom",
    fontsize=7
)

ax.text(
    len(CYBER_CWES) + (len(CVE_CWES) - 1) / 2,
    1.015,
    "CVE-pair proxy",
    transform=trans,
    ha="center",
    va="bottom",
    fontsize=7
)

# Clean frame
ax.set_xlim(-0.6, n_cols + 0.85)
ax.set_ylim(n_rows - 0.4, -0.68)

for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout(pad=0.35)

plt.savefig(OUTPUT_PDF, bbox_inches="tight")
plt.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight")
plt.close()

print(f"Saved PDF: {OUTPUT_PDF}")
print(f"Saved PNG: {OUTPUT_PNG}")
print(matrix)