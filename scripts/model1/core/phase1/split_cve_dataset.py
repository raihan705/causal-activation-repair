"""
Phase 1 - Step 1.8
split_cve_dataset.py

CVE-unit split: 60% train / 20% val / 20% test
- Split unit: cve_id for raw; seed_cve_id for augmented
- Held-out CWEs (CWE-22, CWE-290) forced to test only
- Exploratory CWEs (CWE-338) split normally
Output: train_pairs.csv, val_pairs.csv, test_pairs.csv
"""

import os
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
IN_CSV     = os.path.join(PROJECT_ROOT, "data/processed/vulnerability_pairs_balanced.csv")
TRAIN_CSV  = os.path.join(PROJECT_ROOT, "data/processed/train_pairs.csv")
VAL_CSV    = os.path.join(PROJECT_ROOT, "data/processed/val_pairs.csv")
TEST_CSV   = os.path.join(PROJECT_ROOT, "data/processed/test_pairs.csv")

HELDOUT_CWES = {"CWE-22", "CWE-290"}
SEED = 42
TRAIN_FRAC = 0.60
VAL_FRAC   = 0.20
# TEST_FRAC  = 0.20

df = pd.read_csv(IN_CSV, low_memory=False)

# Assign split unit: seed_cve_id for augmented, cve_id for raw
df["split_unit"] = df.apply(
    lambda r: str(r["seed_cve_id"]) if r["is_augmented"] else str(r["cve_id"]), axis=1
)

# ── Held-out CWEs go entirely to test ────────────────────────────────────────
held_mask = df["cwe_id"].isin(HELDOUT_CWES)
df_held   = df[held_mask].copy()
df_split  = df[~held_mask].copy()

# ── Get unique split units for non-held-out rows ─────────────────────────────
rng   = np.random.RandomState(SEED)
units = sorted(df_split["split_unit"].unique())
rng.shuffle(units)

n_train = int(len(units) * TRAIN_FRAC)
n_val   = int(len(units) * VAL_FRAC)

train_units = set(units[:n_train])
val_units   = set(units[n_train:n_train + n_val])
test_units  = set(units[n_train + n_val:])

df_train = df_split[df_split["split_unit"].isin(train_units)].copy()
df_val   = df_split[df_split["split_unit"].isin(val_units)].copy()
df_test  = pd.concat([
    df_split[df_split["split_unit"].isin(test_units)],
    df_held
], ignore_index=True)

# Drop helper column
for d in [df_train, df_val, df_test]:
    d.drop(columns=["split_unit"], inplace=True)

df_train.to_csv(TRAIN_CSV, index=False, encoding="utf-8")
df_val.to_csv(VAL_CSV,     index=False, encoding="utf-8")
df_test.to_csv(TEST_CSV,   index=False, encoding="utf-8")

print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
print("\nTrain CWE distribution:")
print(df_train["cwe_id"].value_counts().to_string())
print("\nVal CWE distribution:")
print(df_val["cwe_id"].value_counts().to_string())
print("\nTest CWE distribution:")
print(df_test["cwe_id"].value_counts().to_string())

# Verify no CVE unit crosses splits
train_units_check = set(df_train["cve_id"].unique())
val_units_check   = set(df_val["cve_id"].unique())
test_units_check  = set(df_test[~df_test["cwe_id"].isin(HELDOUT_CWES)]["cve_id"].unique())
tv = train_units_check & val_units_check
tt = train_units_check & test_units_check
vt = val_units_check   & test_units_check
print(f"\nCVE overlap train/val: {len(tv)}, train/test: {len(tt)}, val/test: {len(vt)}")