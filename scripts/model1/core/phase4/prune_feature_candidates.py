"""
Phase 4 — Step 4.3
Prune feature candidates per (layer, CWE).
Pruning rules (any one → remove):
  - freq_unsafe < 0.02
  - freq_unsafe > 0.95 AND freq_safe > 0.95  (not discriminative)
  - score < 0.15
  - NaN/Inf or extreme magnitudes (|mean_activation| > 1e4)
"""

import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path.cwd()
PHASE4_DIR   = PROJECT_ROOT / "outputs/phase4"
OUT_DIR      = PHASE4_DIR

# ── Config ─────────────────────────────────────────────────────────────────────
LAYERS     = [16, 19, 23]
CYBER_CWES = [120, 327, 89, 338]
CVE_CWES   = [79, 125, 787, 190, 476]
ALL_CWES   = CYBER_CWES + CVE_CWES

ACTIVE_THRESH     = 0.01   # feature considered active if value > this
FREQ_MIN          = 0.02   # freq_unsafe floor
FREQ_BOTH_MAX     = 0.95   # both unsafe and safe active → not discriminative
SCORE_MIN_CYBER = 0.15
SCORE_MIN_CVE   = 0.05
MAG_MAX           = 1e4    # extreme magnitude guard

# ── Helpers ────────────────────────────────────────────────────────────────────
def load_latents(path):
    t = torch.load(path, map_location="cpu")
    if isinstance(t, dict):
        return t["latents"].float()
    return t.float()

def compute_freq(latents, thresh=ACTIVE_THRESH):
    """Fraction of samples where feature activation > thresh. Returns (D,) array."""
    return (latents > thresh).float().mean(dim=0).numpy()

def compute_mean_mag(latents):
    return latents.mean(dim=0).abs().numpy()

# ── Pruning ────────────────────────────────────────────────────────────────────
all_stats_rows = []
all_summary_rows = []

for layer in LAYERS:
    print(f"\n=== Layer {layer} ===")
    rankings = json.load(open(PHASE4_DIR / f"statistical_feature_rankings_layer_{layer}.json"))
    pruned_layer = {}

    for cwe in ALL_CWES:
        key = f"CWE-{cwe}"
        if key not in rankings:
            print(f"  [SKIP] {key}: not in rankings")
            continue

        candidates = rankings[key]  # list of dicts, top-200

        unsafe_path = PHASE4_DIR / f"unsafe_latents_layer_{layer}_cwe{cwe}.pt"
        safe_path   = PHASE4_DIR / f"safe_latents_layer_{layer}_cwe{cwe}.pt"

        if not unsafe_path.exists() or not safe_path.exists():
            print(f"  [SKIP] {key}: latent files missing")
            continue

        unsafe = load_latents(unsafe_path)
        safe   = load_latents(safe_path)

        freq_unsafe_all = compute_freq(unsafe)
        freq_safe_all   = compute_freq(safe)
        mean_mag_all    = compute_mean_mag(unsafe)

        kept, pruned = [], []
        for rec in candidates:
            fid   = rec["feature_id"]
            score = rec["score"]

            fu = float(freq_unsafe_all[fid])
            fs = float(freq_safe_all[fid])
            mm = float(mean_mag_all[fid])

            # Pruning rules
            reasons = []
            if fu < FREQ_MIN:
                reasons.append(f"freq_unsafe={fu:.3f}<{FREQ_MIN}")
            if fu > FREQ_BOTH_MAX and fs > FREQ_BOTH_MAX:
                reasons.append(f"both_high(fu={fu:.2f},fs={fs:.2f})")
            score_min = SCORE_MIN_CYBER if cwe in CYBER_CWES else SCORE_MIN_CVE
            if score < score_min:
                reasons.append(f"score={score:.4f}<{score_min}")
            if np.isnan(score) or np.isinf(score):
                reasons.append("nan_inf_score")
            if np.isnan(mm) or np.isinf(mm) or mm > MAG_MAX:
                reasons.append(f"extreme_mag={mm:.1f}")

            row = {
                "feature_id":   fid,
                "layer":        layer,
                "cwe_id":       key,
                "rank":         rec["rank"],
                "score":        round(score, 4),
                "hist_dist":    round(rec["hist_dist"], 4),
                "delta_mu":     round(rec["delta_mu"], 4),
                "ks_stat":      round(rec["ks_stat"], 4),
                "freq_unsafe":  round(fu, 4),
                "freq_safe":    round(fs, 4),
                "mean_mag":     round(mm, 4),
                "kept":         len(reasons) == 0,
                "prune_reason": "; ".join(reasons) if reasons else "",
            }
            all_stats_rows.append(row)

            if reasons:
                pruned.append(fid)
            else:
                kept.append(rec)

        pruned_layer[key] = kept

        print(f"  {key}: {len(candidates)} candidates → {len(kept)} kept, {len(pruned)} pruned")

        all_summary_rows.append({
            "layer":       layer,
            "cwe_id":      key,
            "n_input":     len(candidates),
            "n_kept":      len(kept),
            "n_pruned":    len(pruned),
            "source":      "cyber_dev" if cwe in CYBER_CWES else "cve_pairs",
        })

    # Save pruned candidates
    out_json = OUT_DIR / f"pruned_feature_candidates_layer_{layer}.json"
    with open(out_json, "w") as f:
        json.dump(pruned_layer, f, indent=2)
    print(f"  Saved {out_json.name}")

# Save activation stats
stats_df = pd.DataFrame(all_stats_rows)
stats_csv = OUT_DIR / "feature_activation_stats_all.csv"
stats_df.to_csv(stats_csv, index=False)

# Save summary
summary_df = pd.DataFrame(all_summary_rows)
summary_csv = OUT_DIR / "pruning_summary.csv"
summary_df.to_csv(summary_csv, index=False)

print(f"\nPruning summary:")
print(summary_df.to_string(index=False))

# Checkpoint check
n_cwes_with_candidates = summary_df[summary_df["n_kept"] > 0]["cwe_id"].nunique()
print(f"\nCWEs with pruned candidates (kept > 0): {n_cwes_with_candidates}/9")
if n_cwes_with_candidates >= 6:
    print("PASS: >= 6/8 derivation CWEs have pruned candidates (checkpoint criterion met)")
else:
    print("WARN: < 6 CWEs have pruned candidates — review pruning thresholds")

print("\nStep 4.3 complete.")