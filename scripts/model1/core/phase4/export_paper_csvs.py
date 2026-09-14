"""
Phase 4 — Paper CSV Export
Produces two paper-ready CSVs from Phase 4 raw outputs:

  1. phase4_causal_validation.csv  — feature-level causal validation (Step 4.4)
  2. phase4_feature_ranking.csv    — statistical ranking scores (Step 4.3)
"""

import json
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path.cwd()
PHASE4_DIR   = PROJECT_ROOT / "outputs/phase4"
OUT_DIR      = PHASE4_DIR

LAYERS     = [16, 19, 23]
CYBER_CWES = {120, 327, 89, 338}
CVE_CWES   = {79, 125, 787, 190, 476}

# ── 1. phase4_causal_validation.csv ───────────────────────────────────────────
rows = []
for layer in LAYERS:
    path = PHASE4_DIR / f"feature_validation_results_layer_{layer}.json"
    if not path.exists():
        print(f"[WARN] {path.name} not found")
        continue
    data = json.load(open(path))
    for cwe_key, features in data.items():
        cwe_num = int(cwe_key.replace("CWE-", ""))
        source  = "cyber_dev" if cwe_num in CYBER_CWES else "cve_pairs"
        for f in features:
            validated = (
                f.get("repair_rate", 0)  > 0    and
                f.get("corrupt_rate", 1) < 0.20 and
                f.get("invalid_rate", 1) < 0.50
            )
            rows.append({
                "cwe":             cwe_key,
                "layer":           layer,
                "feature_id":      f["feature_id"],
                "repair_rate":     f["repair_rate"],
                "corruption_rate": f["corrupt_rate"],
                "invalid_rate":    f["invalid_rate"],
                "validated":       validated,
                "n_prompts":       f.get("n_unsafe_tested", 0),
                "n_repair":        f.get("n_repair", 0),
                "n_corrupt":       f.get("n_corrupt", 0),
                "source":          source,
            })

val_df = pd.DataFrame(rows)
val_path = OUT_DIR / "phase4_causal_validation.csv"
val_df.to_csv(val_path, index=False)
print(f"Saved phase4_causal_validation.csv  ({len(val_df)} rows)")

# Quick summary check
print("\nValidated features per (CWE, layer):")
summary = val_df[val_df["validated"]].groupby(["cwe","layer"]).size().reset_index(name="n_validated")
print(summary.to_string(index=False))

# Per-CWE totals across layers
print("\nPer-CWE totals:")
totals = val_df[val_df["validated"]].groupby("cwe").agg(
    N_validated=("feature_id","count"),
    Best_repair=("repair_rate","max"),
    Post_corrupt=("corruption_rate","max"),
    N_L16=("layer", lambda x: (x==16).sum()),
    N_L19=("layer", lambda x: (x==19).sum()),
    N_L23=("layer", lambda x: (x==23).sum()),
).reset_index()
print(totals.to_string(index=False))

# ── 2. phase4_feature_ranking.csv ─────────────────────────────────────────────
stats_path = PHASE4_DIR / "feature_activation_stats_all.csv"
if stats_path.exists():
    rank_df = pd.read_csv(stats_path)
    # Keep only columns needed + add source
    keep = ["feature_id", "layer", "cwe_id", "score", "hist_dist",
            "delta_mu", "ks_stat", "freq_unsafe", "freq_safe", "kept"]
    rank_df = rank_df[[c for c in keep if c in rank_df.columns]].copy()
    rank_df["source"] = rank_df["cwe_id"].apply(
        lambda x: "cyber_dev" if int(x.replace("CWE-","")) in CYBER_CWES else "cve_pairs"
    )
    rank_path = OUT_DIR / "phase4_feature_ranking.csv"
    rank_df.to_csv(rank_path, index=False)
    print(f"\nSaved phase4_feature_ranking.csv  ({len(rank_df)} rows)")

    # Top composite score per CWE (across all layers)
    print("\nTop composite score per CWE:")
    top = rank_df.groupby("cwe_id")["score"].max().reset_index()
    top.columns = ["cwe", "top_composite_score"]
    print(top.to_string(index=False))
else:
    print("[WARN] feature_activation_stats_all.csv not found")

print("\nExport complete.")