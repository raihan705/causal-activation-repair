"""Reproduce the four-panel intervention-boundary figure with audited values.

The visual design follows ``group_steering_limits.py`` and
``layer_sensitivity.py``.  Panels (a)--(b) report group-steering diagnostics;
Panels (c)--(d) report the layer-sensitivity ablation.
"""

import json
import os
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
INPUT_JSON = ROOT / "results" / "model1" / "intervention_boundary_summary.json"
OUTPUT_DIR = Path(os.environ.get("CAR_FIGURE_OUTPUT_DIR", ROOT / "build" / "paper_figures"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PDF = OUTPUT_DIR / "fig_intervention_boundaries.pdf"
OUTPUT_PNG = OUTPUT_DIR / "fig_intervention_boundaries.png"


# Audited Model 1 values.
with INPUT_JSON.open("r", encoding="utf-8") as handle:
    evidence = json.load(handle)

group = evidence["group_intervention"]
gate = evidence["plausibility_gate"]
layer = evidence["layer_placement"]

METHOD_LABELS = [r"B$^\ast$", "B3-ungated"]
TOTAL_ABSENT = np.array(
    [group["bstar"]["absent_or_whitespace_outputs"], group["b3_ungated"]["absent_or_whitespace_outputs"]],
    dtype=float,
)
B0_POSITIVE_ABSENT = np.array(
    [
        group["bstar"]["b0_positive_absent_or_whitespace_outputs"],
        group["b3_ungated"]["b0_positive_absent_or_whitespace_outputs"],
    ],
    dtype=float,
)

ROUTED_STEPS = gate["generation_steps"]
NONFINITE_KL_STEPS = gate["nonfinite_kl_steps"]
NONZERO_STEERING_STEPS = gate["nonzero_steering_steps"]
GATE_EXECUTION_PERCENT = np.array(
    [
        100.0 * NONFINITE_KL_STEPS / ROUTED_STEPS,
        100.0 * NONZERO_STEERING_STEPS / ROUTED_STEPS,
    ]
)

LAYER_LABELS = [r"B$^\ast$ layer-sensitive", "Fixed-L19 ablation"]
CORRVRR = np.round(100.0 * np.array(
    [layer["bstar_layer_sensitive"]["corrvrr"], layer["fixed_layer_19"]["corrvrr"]]
), 2)
CORRUPTION = np.round(100.0 * np.array(
    [layer["bstar_layer_sensitive"]["corruption_rate"], layer["fixed_layer_19"]["corruption_rate"]]
), 2)


# Fail loudly if a future edit changes the frozen values.
np.testing.assert_array_equal(TOTAL_ABSENT, [26, 68])
np.testing.assert_array_equal(B0_POSITIVE_ABSENT, [0, 9])
np.testing.assert_allclose(GATE_EXECUTION_PERCENT, [100.0, 0.0])
np.testing.assert_allclose(CORRVRR, [60.00, 53.33])
np.testing.assert_allclose(CORRUPTION, [0.35, 0.70])


matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["font.size"] = 9

BLUE = "#1f77b4"
RED = "#d62728"
COLORS = [BLUE, RED]


def format_axis(ax):
    ax.yaxis.grid(True, linestyle="--", linewidth=0.45, alpha=0.30)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def annotate_counts(ax, bars):
    for bar in bars:
        value = bar.get_height()
        y_pos = value + 1.2 if value > 0 else 1.0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos,
            f"{int(value)}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.9))


# Panel (a): output-instability evidence.
ax = axes[0, 0]
x = np.arange(len(METHOD_LABELS))
bar_width = 0.34
bars_total = ax.bar(
    x - bar_width / 2,
    TOTAL_ABSENT,
    width=bar_width,
    label="Total empty outputs",
    color=BLUE,
    edgecolor="black",
    linewidth=0.6,
)
bars_b0 = ax.bar(
    x + bar_width / 2,
    B0_POSITIVE_ABSENT,
    width=bar_width,
    label="B0-positive empty outputs",
    color=RED,
    edgecolor="black",
    linewidth=0.6,
)
ax.set_xticks(x)
ax.set_xticklabels(METHOD_LABELS)
ax.set_ylabel("Output count")
ax.set_title("(a) Generation-instability signal", fontsize=10, pad=6)
ax.set_ylim(0, 76)
annotate_counts(ax, bars_total)
annotate_counts(ax, bars_b0)
ax.legend(
    frameon=False,
    loc="upper left",
    fontsize=8,
    handlelength=1.0,
    handletextpad=0.35,
    borderaxespad=0.25,
)
format_axis(ax)


# Panel (b): verified plausibility-gate execution trace.
ax = axes[0, 1]
gate_labels = ["Non-finite KL", "Nonzero steering"]
x_gate = np.arange(len(gate_labels))
bars_gate = ax.bar(
    x_gate,
    GATE_EXECUTION_PERCENT,
    width=0.46,
    color=[RED, BLUE],
    edgecolor="black",
    linewidth=0.6,
)
ax.set_xticks(x_gate)
ax.set_xticklabels(gate_labels)
ax.set_ylabel("Share of routed steps (%)")
ax.set_title("(b) B3-PG execution trace", fontsize=10, pad=6)
ax.set_ylim(0, 112)
gate_annotations = ["100.0%\n(182,965)", "0.0%\n(0)"]
for bar, label in zip(bars_gate, gate_annotations):
    value = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + (2.0 if value > 0 else 1.5),
        label,
        ha="center",
        va="bottom",
        fontsize=8,
    )
format_axis(ax)


# Panel (c): paired repair under layer-sensitive and fixed-layer steering.
ax = axes[1, 0]
x_layer = np.arange(len(LAYER_LABELS))
bars_vrr = ax.bar(
    x_layer,
    CORRVRR,
    width=0.48,
    color=COLORS,
    edgecolor="black",
    linewidth=0.6,
)
ax.set_xticks(x_layer)
ax.set_xticklabels(LAYER_LABELS, fontsize=8)
ax.set_ylabel("Corrected VRR (%)")
ax.set_title("(c) Repair effectiveness", fontsize=10, pad=6)
ax.set_ylim(0, 68)
for bar, value in zip(bars_vrr, CORRVRR):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1.2,
        f"{value:.1f}%",
        ha="center",
        va="bottom",
        fontsize=8,
    )
format_axis(ax)


# Panel (d): collateral corruption under the same ablation.
ax = axes[1, 1]
bars_corruption = ax.bar(
    x_layer,
    CORRUPTION,
    width=0.48,
    color=COLORS,
    edgecolor="black",
    linewidth=0.6,
)
ax.set_xticks(x_layer)
ax.set_xticklabels(LAYER_LABELS, fontsize=8)
ax.set_ylabel("Corruption rate (%)")
ax.set_title("(d) Collateral corruption", fontsize=10, pad=6)
ax.set_ylim(0, 0.85)
for bar, value in zip(bars_corruption, CORRUPTION):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.035,
        f"{value:.2f}%",
        ha="center",
        va="bottom",
        fontsize=8,
    )
format_axis(ax)


fig.subplots_adjust(
    top=0.94,
    bottom=0.12,
    left=0.08,
    right=0.98,
    hspace=0.40,
    wspace=0.30,
)

fig.savefig(OUTPUT_PDF, bbox_inches="tight")
fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved PDF: {OUTPUT_PDF}")
print(f"Saved PNG: {OUTPUT_PNG}")
