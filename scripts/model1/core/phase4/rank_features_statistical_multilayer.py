"""
Phase 4 — Step 4.2
Statistical feature ranking per (layer, CWE).
Compares unsafe vs safe latent distributions and scores each SAE feature.
Score = alpha * hist_dist + beta * |delta_mu| + gamma * KS
Keeps top-200 candidates per (layer, CWE).
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ks_2samp

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path.cwd()
PHASE4_DIR   = PROJECT_ROOT / "outputs/phase4"
OUT_DIR      = PHASE4_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
LAYERS = [16, 19, 23]
CYBER_CWES  = [120, 327, 89, 338]
CVE_CWES    = [79, 125, 787, 190, 476]
ALL_CWES    = CYBER_CWES + CVE_CWES

TOP_K       = 200
ALPHA       = 0.4   # histogram distance weight
BETA        = 0.4   # |delta_mu| weight
GAMMA       = 0.2   # KS weight

# KS is expensive on 32768 dims — subsample features for KS if needed
KS_SUBSAMPLE = 500  # run KS on top-500 by hist+mu, then merge

# ── Helpers ────────────────────────────────────────────────────────────────────
def load_latents(path):
    t = torch.load(path, map_location="cpu")
    if isinstance(t, dict):
        return t["latents"].float()
    return t.float()

def histogram_distance(a, b, bins=20):
    """
    Per-feature histogram overlap distance (1 - overlap).
    a, b: (N, D) tensors → returns (D,) array
    """
    a = a.numpy()
    b = b.numpy()
    D = a.shape[1]
    dist = np.zeros(D, dtype=np.float32)
    for d in range(D):
        all_vals = np.concatenate([a[:, d], b[:, d]])
        lo, hi = all_vals.min(), all_vals.max()
        if hi - lo < 1e-8:
            dist[d] = 0.0
            continue
        edges = np.linspace(lo, hi, bins + 1)
        ha, _ = np.histogram(a[:, d], bins=edges, density=True)
        hb, _ = np.histogram(b[:, d], bins=edges, density=True)
        bin_w = edges[1] - edges[0]
        overlap = np.sum(np.minimum(ha, hb)) * bin_w
        dist[d] = 1.0 - overlap
    return dist

def delta_mu(a, b):
    """Mean activation difference per feature."""
    return (a.mean(dim=0) - b.mean(dim=0)).abs().numpy()

def ks_scores(a, b, feature_indices):
    """KS statistic for a subset of feature indices."""
    a = a.numpy()
    b = b.numpy()
    scores = np.zeros(len(feature_indices), dtype=np.float32)
    for i, idx in enumerate(feature_indices):
        stat, _ = ks_2samp(a[:, idx], b[:, idx])
        scores[i] = stat
    return scores

# ── Main ranking ───────────────────────────────────────────────────────────────
def rank_cwe(layer, cwe):
    unsafe_path = PHASE4_DIR / f"unsafe_latents_layer_{layer}_cwe{cwe}.pt"
    safe_path   = PHASE4_DIR / f"safe_latents_layer_{layer}_cwe{cwe}.pt"

    if not unsafe_path.exists() or not safe_path.exists():
        print(f"  [SKIP] layer {layer} CWE-{cwe}: latent files missing")
        return None

    unsafe = load_latents(unsafe_path)
    safe   = load_latents(safe_path)

    if unsafe.shape[0] == 0:
        print(f"  [SKIP] layer {layer} CWE-{cwe}: 0 unsafe samples")
        return None

    # For CVE CWEs: safe tensor is cve_fixed (same size as unsafe)
    # For cyber CWEs: safe tensor is much larger — subsample to balance
    n_unsafe = unsafe.shape[0]
    n_safe   = safe.shape[0]
    if n_safe > 5 * n_unsafe:
        idx = torch.randperm(n_safe, generator=torch.Generator().manual_seed(42))[:5 * n_unsafe]
        safe = safe[idx]

    print(f"  layer {layer} CWE-{cwe}: unsafe={n_unsafe} safe={safe.shape[0]} D={unsafe.shape[1]}")

    # Step 1: histogram distance + delta_mu (all features)
    hd   = histogram_distance(unsafe, safe)
    dmu  = delta_mu(unsafe, safe)

    # Step 2: partial score without KS to select candidates for KS
    partial_score = ALPHA * hd + BETA * dmu
    top_ks_idx = np.argsort(partial_score)[::-1][:KS_SUBSAMPLE]

    # Step 3: KS on top candidates
    ks_full = np.zeros(unsafe.shape[1], dtype=np.float32)
    ks_vals = ks_scores(unsafe, safe, top_ks_idx)
    ks_full[top_ks_idx] = ks_vals

    # Step 4: composite score
    score = ALPHA * hd + BETA * dmu + GAMMA * ks_full

    # Step 5: top-200
    top_idx = np.argsort(score)[::-1][:TOP_K]

    records = []
    for rank, feat_id in enumerate(top_idx):
        records.append({
            "feature_id":    int(feat_id),
            "rank":          rank + 1,
            "score":         float(score[feat_id]),
            "hist_dist":     float(hd[feat_id]),
            "delta_mu":      float(dmu[feat_id]),
            "ks_stat":       float(ks_full[feat_id]),
            "layer":         layer,
            "cwe_id":        f"CWE-{cwe}",
            "n_unsafe":      n_unsafe,
            "n_safe":        int(safe.shape[0]),
        })

    return records

# ── Run all ────────────────────────────────────────────────────────────────────
all_summary_rows = []

for layer in LAYERS:
    print(f"\n=== Layer {layer} ===")
    layer_records = {}

    for cwe in ALL_CWES:
        records = rank_cwe(layer, cwe)
        if records is None:
            continue
        layer_records[f"CWE-{cwe}"] = records

        # summary row
        top1 = records[0]
        all_summary_rows.append({
            "layer":      layer,
            "cwe_id":     f"CWE-{cwe}",
            "n_candidates": len(records),
            "top1_feat":  top1["feature_id"],
            "top1_score": round(top1["score"], 4),
            "top1_hist":  round(top1["hist_dist"], 4),
            "top1_dmu":   round(top1["delta_mu"], 4),
            "top1_ks":    round(top1["ks_stat"], 4),
        })

    # Save per-layer JSON
    out_json = OUT_DIR / f"statistical_feature_rankings_layer_{layer}.json"
    with open(out_json, "w") as f:
        json.dump(layer_records, f, indent=2)
    print(f"  Saved {out_json.name}")

# Summary CSV
summary_df = pd.DataFrame(all_summary_rows)
summary_csv = OUT_DIR / "statistical_feature_rankings_summary.csv"
summary_df.to_csv(summary_csv, index=False)
print(f"\nSummary saved to {summary_csv}")
print(summary_df.to_string(index=False))
print("\nStep 4.2 complete.")