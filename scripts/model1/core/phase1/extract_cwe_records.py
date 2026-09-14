"""
Phase 1 - Step 1.5
extract_cwe_records.py

Merge cve_pairs_filtered.csv with crypto_augmented_pairs.csv,
retain only target CWEs, output vulnerability_pairs.csv.
"""

import os
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
FILTERED_CSV = os.path.join(PROJECT_ROOT, "data/processed/cve_pairs_filtered.csv")
AUGMENTED_CSV = os.path.join(PROJECT_ROOT, "data/processed/crypto_augmented_pairs.csv")
OUT_CSV = os.path.join(PROJECT_ROOT, "data/processed/vulnerability_pairs.csv")

DERIVATION_CWES = {"CWE-120","CWE-125","CWE-787","CWE-190","CWE-476","CWE-89","CWE-79","CWE-338","CWE-327"}
HELDOUT_CWES   = {"CWE-22","CWE-290"}
TARGET_CWES    = DERIVATION_CWES | HELDOUT_CWES

df_raw = pd.read_csv(FILTERED_CSV, low_memory=False)
df_aug = pd.read_csv(AUGMENTED_CSV, low_memory=False)

# keep only target CWEs from raw
df_raw = df_raw[df_raw["cwe_id"].isin(TARGET_CWES)].copy()

# align columns: augmented already has all required fields
df_merged = pd.concat([df_raw, df_aug], ignore_index=True)

# final column set
cols = ["pair_id","cve_id","cwe_id","file_change_id","hash","filename",
        "language","vulnerable_code","fixed_code","num_lines_added",
        "num_lines_deleted","mono_decision","source","is_augmented","seed_cve_id"]
for c in cols:
    if c not in df_merged.columns:
        df_merged[c] = ""
df_merged = df_merged[cols]

df_merged.to_csv(OUT_CSV, index=False, encoding="utf-8")

print(f"Total pairs: {len(df_merged)}")
print(df_merged["cwe_id"].value_counts().to_string())
print(f"\nAugmented: {df_merged['is_augmented'].sum()}")
print(f"Raw: {(df_merged['is_augmented'] == False).sum()}")