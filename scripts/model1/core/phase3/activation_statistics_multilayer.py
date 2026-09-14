"""
Phase 3, Step 3.7 — Latent activation statistics
Computes feature activation frequency, mean/variance of magnitudes,
and sparsity distributions per layer per source.
Outputs:
  outputs/phase3/latent_activation_statistics_layer_{l}.csv  (per layer)
  outputs/phase3/latent_activation_statistics_summary.csv    (combined)
"""

import csv
import logging
from pathlib import Path

import torch

# ── constants ──────────────────────────────────────────────────────────────────
INTERVENTION_LAYERS = [16, 19, 23]
SOURCES             = ["cve_vuln", "cve_fixed", "cyber_dev"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAT_ROOT     = PROJECT_ROOT / "data/activations/latent"
OUT_DIR      = PROJECT_ROOT / "outputs/phase3"
LOG_PATH     = OUT_DIR / "activation_statistics.log"

# ── logging ────────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"),
              logging.StreamHandler()],
)
log = logging.getLogger(__name__)


# ── compute stats for one latent tensor ───────────────────────────────────────
def compute_stats(latents: torch.Tensor, layer: int, source: str) -> dict:
    """
    latents: (N, d_sae)
    Returns a flat dict of aggregate statistics.
    """
    N, d_sae = latents.shape
    active   = latents > 0                          # (N, d_sae) bool

    # per-sample sparsity
    n_active_per_sample = active.float().sum(dim=1)  # (N,)
    sparsity_per_sample = n_active_per_sample / d_sae

    # per-feature activation frequency across samples
    feat_freq = active.float().mean(dim=0)           # (d_sae,)

    # per-feature mean magnitude (only over active entries)
    # replace zeros with nan for masked mean
    vals      = latents.clone()
    vals[~active] = float("nan")
    feat_mean_magnitude = vals.nanmean(dim=0)        # (d_sae,) — nan where never active
    # fill nan with 0 for summary stats
    feat_mean_magnitude = feat_mean_magnitude.nan_to_num(0.0)

    # overall magnitude stats across all non-zero entries
    nonzero_vals = latents[active]
    overall_mean_mag = float(nonzero_vals.mean()) if nonzero_vals.numel() > 0 else 0.0
    overall_std_mag  = float(nonzero_vals.std())  if nonzero_vals.numel() > 0 else 0.0
    overall_max_mag  = float(nonzero_vals.max())  if nonzero_vals.numel() > 0 else 0.0

    # number of features active in at least 1 sample
    n_features_ever_active = int((feat_freq > 0).sum().item())

    # top-20 most frequently active features
    top20_freq_idx = feat_freq.topk(20).indices.tolist()
    top20_freq_val = feat_freq.topk(20).values.tolist()

    return {
        "layer":                    layer,
        "source":                   source,
        "n_samples":                N,
        "d_sae":                    d_sae,
        "mean_active_per_sample":   round(float(n_active_per_sample.mean()), 4),
        "std_active_per_sample":    round(float(n_active_per_sample.std()), 4),
        "mean_sparsity_active":     round(float(sparsity_per_sample.mean()), 6),
        "overall_mean_magnitude":   round(overall_mean_mag, 4),
        "overall_std_magnitude":    round(overall_std_mag, 4),
        "overall_max_magnitude":    round(overall_max_mag, 4),
        "n_features_ever_active":   n_features_ever_active,
        "top20_feature_ids":        str(top20_freq_idx),
        "top20_feature_freqs":      str([round(v, 4) for v in top20_freq_val]),
    }


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    all_rows = []

    for l in INTERVENTION_LAYERS:
        log.info(f"=== Layer {l} ===")
        layer_rows = []

        for src in SOURCES:
            lat_path = LAT_ROOT / f"layer_{l}" / f"{src}_latents.pt"
            if not lat_path.exists():
                log.warning(f"  [{src}] latent file not found: {lat_path} — skipped")
                continue

            data    = torch.load(lat_path, map_location="cpu")
            latents = data["latents"].float()   # (N, d_sae)

            if latents.numel() == 0:
                log.warning(f"  [{src}] empty latents — skipped")
                continue

            stats = compute_stats(latents, l, src)
            layer_rows.append(stats)
            all_rows.append(stats)

            log.info(f"  [{src}] n={stats['n_samples']}, "
                     f"mean_active={stats['mean_active_per_sample']}, "
                     f"mean_mag={stats['overall_mean_magnitude']}, "
                     f"n_features_ever_active={stats['n_features_ever_active']}")

        # per-layer CSV
        if layer_rows:
            layer_csv = OUT_DIR / f"latent_activation_statistics_layer_{l}.csv"
            with open(layer_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=layer_rows[0].keys())
                writer.writeheader()
                writer.writerows(layer_rows)
            log.info(f"  Saved {layer_csv}")

    # summary CSV (all layers combined)
    if all_rows:
        summary_csv = OUT_DIR / "latent_activation_statistics_summary.csv"
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        log.info(f"Saved {summary_csv}")

    # checkpoint: files exist and non-empty
    per_layer_ok = all(
        (OUT_DIR / f"latent_activation_statistics_layer_{l}.csv").exists()
        for l in INTERVENTION_LAYERS
    )
    summary_ok = (OUT_DIR / "latent_activation_statistics_summary.csv").exists()

    if per_layer_ok and summary_ok:
        log.info("CHECKPOINT PASS: all statistics files saved.")
    else:
        log.warning("CHECKPOINT: some files missing.")

    log.info("Step 3.7 complete.")


if __name__ == "__main__":
    main()