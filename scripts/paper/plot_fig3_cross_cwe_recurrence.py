"""Regenerate Figure 3 with labels consistent with the revised evidence scope."""

import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = ROOT / "artifacts" / "model1" / "selected_features" / "phase4_causal_validation.csv"
OUTPUT_DIR = Path(os.environ.get("CAR_FIGURE_OUTPUT_DIR", ROOT / "build" / "paper_figures"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PDF = OUTPUT_DIR / "fig3_cross_cwe_feature_matrix.pdf"
OUTPUT_PNG = OUTPUT_DIR / "fig3_cross_cwe_feature_matrix.png"

CYBER_CWES = ["CWE-120", "CWE-327", "CWE-89", "CWE-338"]
CVE_CWES = ["CWE-79", "CWE-125", "CWE-787", "CWE-190", "CWE-476"]
CWE_ORDER = CYBER_CWES + CVE_CWES

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.5,
    }
)

df = pd.read_csv(INPUT_CSV)
if df["validated"].dtype != bool:
    df["validated"] = (
        df["validated"].astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])
    )

# The source column name is historical. Here it means only that the submitted
# screening rule retained the feature--CWE assignment.
retained = df[df["validated"]].copy()
shared = (
    retained.groupby(["layer", "feature_id"], as_index=False)["cwe"]
    .nunique()
    .rename(columns={"cwe": "n_cwes"})
)
shared = shared[shared["n_cwes"] >= 3].copy()
shared["layer_priority"] = shared["layer"].map({19: 0, 16: 1, 23: 2}).fillna(9)
shared = shared.sort_values(
    by=["n_cwes", "layer_priority", "feature_id"], ascending=[False, True, True]
).reset_index(drop=True)
shared["row_label"] = shared.apply(
    lambda row: f"L{int(row['layer'])}-F{int(row['feature_id'])}", axis=1
)

rows = []
for _, row in shared.iterrows():
    feature_rows = retained[
        (retained["layer"] == row["layer"])
        & (retained["feature_id"] == row["feature_id"])
    ]
    retained_cwes = set(feature_rows["cwe"].tolist())
    rows.append(
        {
            "Feature": row["row_label"],
            "n_cwes": row["n_cwes"],
            **{cwe: int(cwe in retained_cwes) for cwe in CWE_ORDER},
        }
    )

matrix_df = pd.DataFrame(rows).set_index("Feature")
n_cwes = matrix_df["n_cwes"].astype(int)
matrix = matrix_df[CWE_ORDER]
n_rows, n_cols = matrix.shape

fig, ax = plt.subplots(figsize=(6.8, max(2.4, 0.32 * n_rows + 1.15)))
for row_index in range(n_rows):
    for column_index in range(n_cols):
        ax.scatter(
            column_index,
            row_index,
            s=34,
            facecolors="none",
            edgecolors="#d0d0d0",
            linewidths=0.6,
            zorder=1,
        )
        if matrix.iloc[row_index, column_index] == 1:
            ax.scatter(
                column_index,
                row_index,
                s=58,
                color="#2F5D8C",
                edgecolors="#1F3F60",
                linewidths=0.35,
                zorder=3,
            )

for row_index in range(n_rows):
    ax.plot(
        [-0.55, n_cols - 0.35],
        [row_index, row_index],
        color="#eeeeee",
        linewidth=0.45,
        zorder=0,
    )

ax.axvline(len(CYBER_CWES) - 0.5, color="#666666", linestyle="--", linewidth=0.75)
ax.axvline(n_cols - 0.05, color="#b5b5b5", linewidth=0.55)
ax.set_xticks(np.arange(n_cols))
ax.set_xticklabels(CWE_ORDER, rotation=35, ha="right", rotation_mode="anchor")
ax.set_yticks(np.arange(n_rows))
ax.set_yticklabels(matrix.index)
ax.invert_yaxis()
ax.set_xlabel("Studied CWE category", labelpad=10)
ax.set_ylabel("SAE latent feature", labelpad=8)
ax.tick_params(axis="both", length=0)

for row_index, count in enumerate(n_cwes):
    ax.text(n_cols + 0.35, row_index, str(count), va="center", ha="center", fontsize=7)

ax.text(
    n_cols + 0.35,
    -0.58,
    "Associated\nCWEs",
    va="bottom",
    ha="center",
    fontsize=7,
)

transform = ax.get_xaxis_transform()
ax.text(
    (len(CYBER_CWES) - 1) / 2,
    1.015,
    "CyberSecEval-detected",
    transform=transform,
    ha="center",
    va="bottom",
    fontsize=7,
)
ax.text(
    len(CYBER_CWES) + (len(CVE_CWES) - 1) / 2,
    1.015,
    "CVE-pair proxy",
    transform=transform,
    ha="center",
    va="bottom",
    fontsize=7,
)

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
