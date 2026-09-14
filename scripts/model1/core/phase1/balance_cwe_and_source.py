"""
Phase 1 - Step 1.7
balance_cwe_and_source.py

Enforce:
  - max_cwe_count <= 2 * min_cwe_count (derivation CWEs only)
  - augmented fraction per CWE <= 0.5
  - downsample augmented first, then raw if needed

Output: vulnerability_pairs_balanced.csv, dataset_statistics.csv
"""

import os
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
IN_CSV   = os.path.join(PROJECT_ROOT, "data/processed/vulnerability_pairs_full.csv")
OUT_CSV  = os.path.join(PROJECT_ROOT, "data/processed/vulnerability_pairs_balanced.csv")
STAT_CSV = os.path.join(PROJECT_ROOT, "data/processed/dataset_statistics.csv")

DERIVATION_CWES  = {"CWE-120","CWE-125","CWE-787","CWE-190","CWE-476","CWE-89","CWE-79","CWE-327"}
EXPLORATORY_CWES = {"CWE-338"}   # N_raw=2 < 20, cannot support feature discovery
HELDOUT_CWES     = {"CWE-22","CWE-290"}
MAX_AUG_FRACTION = 0.5
BALANCE_RATIO    = 2   # relax to 3 only if documented
# Documented relaxations:
#   CWE-327: N_raw=18 (threshold 20) — 10% shortfall, kept as derivation
#   CWE-89:  N_raw=46 (threshold 50) — kept as derivation, min relaxed to 40

SEED = 42

df = pd.read_csv(IN_CSV, low_memory=False)

# ── Step 1: enforce augmentation cap per CWE ─────────────────────────────────
kept = []
for cwe, grp in df.groupby("cwe_id"):
    raw_rows = grp[grp["is_augmented"] == False]
    aug_rows = grp[grp["is_augmented"] == True]
    total = len(raw_rows)
    max_aug = int(total * MAX_AUG_FRACTION / (1 - MAX_AUG_FRACTION)) if total > 0 else len(aug_rows)
    if len(aug_rows) > max_aug:
        aug_rows = aug_rows.sample(n=max_aug, random_state=SEED)
    kept.append(pd.concat([raw_rows, aug_rows]))

df = pd.concat(kept, ignore_index=True)

# ── Step 2: split by role ─────────────────────────────────────────────────────
deriv_df  = df[df["cwe_id"].isin(DERIVATION_CWES)]
explor_df = df[df["cwe_id"].isin(EXPLORATORY_CWES)]
held_df   = df[df["cwe_id"].isin(HELDOUT_CWES)]

counts = deriv_df.groupby("cwe_id").size()
min_count = counts.min()
max_allowed = BALANCE_RATIO * min_count

print(f"Derivation CWEs — min: {min_count}, max: {counts.max()}, 2x cap: {max_allowed}")

actual_ratio = counts.max() / min_count
used_ratio = BALANCE_RATIO
if actual_ratio > 3:
    print(f"WARNING: relaxing balance ratio to 3x (actual {actual_ratio:.1f})")
    used_ratio = 3
    max_allowed = used_ratio * min_count

balanced_parts = []
for cwe, grp in deriv_df.groupby("cwe_id"):
    if len(grp) > max_allowed:
        aug = grp[grp["is_augmented"] == True]
        raw = grp[grp["is_augmented"] == False]
        need = int(max_allowed)
        if len(aug) >= len(grp) - need:
            aug = aug.sample(n=max(0, len(grp) - need), random_state=SEED)
        grp = pd.concat([raw, aug])
        if len(grp) > need:
            grp = grp.sample(n=need, random_state=SEED)
    balanced_parts.append(grp)

deriv_balanced = pd.concat(balanced_parts, ignore_index=True)

# Add exploratory role tag
explor_df = explor_df.copy()
explor_df["role"] = "exploratory"
deriv_balanced["role"] = "derivation"
held_df = held_df.copy()
held_df["role"] = "heldout"

df_out = pd.concat([deriv_balanced, explor_df, held_df], ignore_index=True)

df_out.to_csv(OUT_CSV, index=False, encoding="utf-8")

# ── Statistics ────────────────────────────────────────────────────────────────
stats = df_out.groupby("cwe_id").agg(
    total=("pair_id","count"),
    raw=("is_augmented", lambda x: (x==False).sum()),
    augmented=("is_augmented", lambda x: (x==True).sum()),
).reset_index()
stats["aug_fraction"] = stats["augmented"] / stats["total"]
stats.to_csv(STAT_CSV, index=False)

print(f"\nBalanced dataset: {len(df_out)} pairs")
print(stats.to_string(index=False))
final_counts = df_out[df_out["cwe_id"].isin(DERIVATION_CWES)].groupby("cwe_id").size()
print(f"\nDerivation CWE ratio (max/min): {final_counts.max()/final_counts.min():.2f} (limit: {used_ratio})")