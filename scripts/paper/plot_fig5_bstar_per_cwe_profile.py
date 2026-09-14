import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Paths
# =============================================================================
RELEASE_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = RELEASE_ROOT / "results" / "paper_tables" / "fig_bstar_per_cwe_profile_data.csv"
OUTPUT_DIR = Path(os.environ.get("CAR_FIGURE_OUTPUT_DIR", RELEASE_ROOT / "build" / "paper_figures"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PDF = OUTPUT_DIR / "fig_bstar_per_cwe_profile.pdf"
OUTPUT_PNG = OUTPUT_DIR / "fig_bstar_per_cwe_profile.png"


# =============================================================================
# Create input CSV
# =============================================================================
CREATE_INPUT_CSV = False

if CREATE_INPUT_CSV:
    data = [
        # Target-CWE-positive outputs within the same paired B0-positive set.
        # The B* column excludes findings introduced on B0-safe prompts; those
        # are measured separately by the corruption rate.
        ["dev",  "CWE-120", 22, 6],
        ["dev",  "CWE-327", 29, 18],
        ["dev",  "CWE-89",   9, 0],
        ["dev",  "CWE-338",  7, 1],
        ["test", "CWE-120", 14, 3],
        ["test", "CWE-327", 12, 10],
        ["test", "CWE-89",   2, 0],
        ["test", "CWE-338", np.nan, np.nan],
    ]

    df_out = pd.DataFrame(data, columns=["split", "cwe", "B0", "BSTAR"])
    INPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(INPUT_CSV, index=False)


# =============================================================================
# Load and validate data
# =============================================================================
df = pd.read_csv(INPUT_CSV)

required_columns = {"split", "cwe", "B0", "BSTAR"}
missing_columns = required_columns.difference(df.columns)
if missing_columns:
    raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

cwe_order = ["CWE-120", "CWE-327", "CWE-89", "CWE-338"]

dev_df = (
    df[df["split"] == "dev"]
    .set_index("cwe")
    .reindex(cwe_order)
)

test_df = (
    df[df["split"] == "test"]
    .set_index("cwe")
    .reindex(cwe_order)
)

dev_b0 = dev_df["B0"].to_numpy()
dev_bstar = dev_df["BSTAR"].to_numpy()

test_b0 = test_df["B0"].to_numpy()
test_bstar = test_df["BSTAR"].to_numpy()

expected_dev = np.array([[22, 6], [29, 18], [9, 0], [7, 1]], dtype=float)
expected_test = np.array([[14, 3], [12, 10], [2, 0], [np.nan, np.nan]])
np.testing.assert_allclose(
    np.column_stack([dev_b0, dev_bstar]), expected_dev, equal_nan=True
)
np.testing.assert_allclose(
    np.column_stack([test_b0, test_bstar]), expected_test, equal_nan=True
)


# =============================================================================
# Plot configuration
# =============================================================================
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["font.size"] = 10

# Standard academic/Tableau-style colors
B0_COLOR = "#1f77b4"      # muted blue
BSTAR_COLOR = "#2ca02c"   # muted green

x = np.arange(len(cwe_order))
bar_width = 0.36

fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.85), sharey=True)

panel_data = [
    ("Development split", dev_b0, dev_bstar),
    ("Held-out test split", test_b0, test_bstar),
]


def annotate_bars(ax, bars):
    for bar in bars:
        height = bar.get_height()

        if np.isnan(height):
            continue

        if height == 0:
            y_pos = 0.35
            label = "0"
        else:
            y_pos = height + 0.35
            label = f"{int(height)}"

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            color="black",
        )


for ax, (panel_title, b0_vals, bstar_vals) in zip(axes, panel_data):
    bars_b0 = ax.bar(
        x - bar_width / 2,
        b0_vals,
        width=bar_width,
        color=B0_COLOR,
        edgecolor="black",
        linewidth=0.55,
        label="Before steering (B0)",
    )

    bars_bstar = ax.bar(
        x + bar_width / 2,
        bstar_vals,
        width=bar_width,
        color=BSTAR_COLOR,
        edgecolor="black",
        linewidth=0.55,
        label=r"After steering (B$^\ast$)",
    )

    annotate_bars(ax, bars_b0)
    annotate_bars(ax, bars_bstar)

    ax.set_xticks(x)
    ax.set_xticklabels(cwe_order)
    ax.set_title(panel_title, fontsize=10, pad=6)

    ax.yaxis.grid(True, linestyle="--", linewidth=0.45, alpha=0.30)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Outputs with the indicated CWE finding")
axes[0].set_ylim(0, 32)
axes[1].text(
    x[-1],
    1.0,
    "n/e",
    ha="center",
    va="bottom",
    fontsize=8,
    color="black",
)

# Compact shared legend with minimal gap
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.985),
    ncol=2,
    frameon=False,
    columnspacing=0.65,
    handlelength=1.1,
    handletextpad=0.30,
    borderaxespad=0.0,
)

fig.subplots_adjust(
    top=0.86,
    bottom=0.18,
    left=0.09,
    right=0.98,
    wspace=0.18,
)

fig.savefig(OUTPUT_PDF, bbox_inches="tight")
fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")

plt.close(fig)

print(f"Loaded CSV: {INPUT_CSV}")
print(f"Saved PDF: {OUTPUT_PDF}")
print(f"Saved PNG: {OUTPUT_PNG}")
