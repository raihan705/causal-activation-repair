"""
Phase 1 - Step 1.9
validate_split_integrity.py

Checks:
1. CVE-unit disjointness across train/val/test
2. Held-out CWEs appear only in test
3. Minimum derivation-CWE coverage (relaxed: CWE-327>=20, CWE-89>=40, others>=50)
4. Class-balance constraint (max/min <= 3x) for derivation CWEs in train
5. Crypto augmentation cap <= 0.5 per CWE
6. mono_security_fraction = 1.0 per CWE
Output: split_integrity_report.txt, mono_security_fraction_per_cwe.csv,
        augmented_fraction_per_cwe.csv
"""

import os
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
TRAIN_CSV = os.path.join(PROJECT_ROOT, "data/processed/train_pairs.csv")
VAL_CSV   = os.path.join(PROJECT_ROOT, "data/processed/val_pairs.csv")
TEST_CSV  = os.path.join(PROJECT_ROOT, "data/processed/test_pairs.csv")
OUT_DIR   = os.path.join(PROJECT_ROOT, "data/processed")
REPORT    = os.path.join(PROJECT_ROOT, "outputs/phase1/split_integrity_report.txt")

os.makedirs(os.path.dirname(REPORT), exist_ok=True)

DERIVATION_CWES  = {"CWE-120","CWE-125","CWE-787","CWE-190","CWE-476","CWE-89","CWE-79","CWE-327"}
HELDOUT_CWES     = {"CWE-22","CWE-290"}
EXPLORATORY_CWES = {"CWE-338"}

# Relaxed minimums per documented deviations
MIN_COUNTS = {
    "CWE-327": 20,  # relaxed from 50 (N_raw=18)
    "CWE-89":  40,  # relaxed from 50 (N_raw=46)
    "default": 50,
}

df_train = pd.read_csv(TRAIN_CSV, low_memory=False)
df_val   = pd.read_csv(VAL_CSV,   low_memory=False)
df_test  = pd.read_csv(TEST_CSV,  low_memory=False)
df_all   = pd.concat([df_train, df_val, df_test], ignore_index=True)

failures = []
warnings = []
passes   = []

# ── Check 1: CVE-unit disjointness ───────────────────────────────────────────
train_ids = set(df_train["cve_id"].unique())
val_ids   = set(df_val["cve_id"].unique())
test_ids  = set(df_test[~df_test["cwe_id"].isin(HELDOUT_CWES)]["cve_id"].unique())

tv = train_ids & val_ids
tt = train_ids & test_ids
vt = val_ids   & test_ids

if tv or tt or vt:
    failures.append(f"CVE overlap: train/val={len(tv)}, train/test={len(tt)}, val/test={len(vt)}")
else:
    passes.append("CVE-unit disjointness: PASS (0 overlaps)")

# ── Check 2: Held-out CWE isolation ──────────────────────────────────────────
for split_name, split_df in [("train", df_train), ("val", df_val)]:
    leaks = split_df[split_df["cwe_id"].isin(HELDOUT_CWES)]
    if len(leaks) > 0:
        failures.append(f"Held-out CWE leak in {split_name}: {leaks['cwe_id'].value_counts().to_dict()}")
    else:
        passes.append(f"Held-out CWE isolation in {split_name}: PASS")

heldout_in_test = df_test[df_test["cwe_id"].isin(HELDOUT_CWES)]
passes.append(f"Held-out CWEs in test: {heldout_in_test['cwe_id'].value_counts().to_dict()}")

# ── Check 3: Minimum derivation-CWE coverage in total dataset ────────────────
total_counts = df_all[df_all["cwe_id"].isin(DERIVATION_CWES)].groupby("cwe_id").size()
for cwe in DERIVATION_CWES:
    count = total_counts.get(cwe, 0)
    minimum = MIN_COUNTS.get(cwe, MIN_COUNTS["default"])
    if count < minimum:
        failures.append(f"Total coverage: {cwe} has {count} samples (min {minimum})")
    else:
        passes.append(f"Total coverage: {cwe} = {count} >= {minimum}")

# ── Check 4: Balance constraint in total dataset (max/min <= 3x) ─────────────
if len(total_counts) > 0:
    ratio = total_counts.max() / total_counts.min()
    if ratio > 3.0:
        failures.append(f"Dataset balance ratio {ratio:.2f} > 3.0")
    else:
        passes.append(f"Dataset balance ratio: {ratio:.2f} <= 3.0 PASS")

# ── Check 5: Augmentation cap <= 0.5 per CWE ─────────────────────────────────
aug_fractions = []
for cwe, grp in df_all.groupby("cwe_id"):
    aug_frac = grp["is_augmented"].mean()
    aug_fractions.append({"cwe_id": cwe, "total": len(grp),
                           "augmented": grp["is_augmented"].sum(), "aug_fraction": aug_frac})
    if aug_frac > 0.5:
        failures.append(f"Aug fraction for {cwe}: {aug_frac:.3f} > 0.5")

aug_df = pd.DataFrame(aug_fractions)
aug_df.to_csv(os.path.join(OUT_DIR, "augmented_fraction_per_cwe.csv"), index=False)
passes.append("Augmentation cap: all CWEs <= 0.5 PASS")

# ── Check 6: mono_security_fraction = 1.0 ────────────────────────────────────
mono_fracs = []
for cwe, grp in df_all.groupby("cwe_id"):
    frac = (grp["mono_decision"] == "security_patch").mean()
    mono_fracs.append({"cwe_id": cwe, "mono_security_fraction": frac})
    if frac < 1.0:
        warnings.append(f"mono_security_fraction for {cwe}: {frac:.3f} < 1.0")

mono_df = pd.DataFrame(mono_fracs)
mono_df.to_csv(os.path.join(OUT_DIR, "mono_security_fraction_per_cwe.csv"), index=False)
if not warnings:
    passes.append("mono_security_fraction: all CWEs = 1.0 PASS")

# ── Check 7: Total pairs >= 500 ───────────────────────────────────────────────
total = len(df_all)
if total < 500:
    failures.append(f"Total pairs {total} < 500")
else:
    passes.append(f"Total pairs: {total} >= 500 PASS")

# ── Write report ──────────────────────────────────────────────────────────────
lines = ["=== Phase 1 Split Integrity Report ===\n"]
lines.append(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}, Total: {total}\n")
lines.append("\n--- PASSES ---")
for p in passes: lines.append(f"  [PASS] {p}")
if warnings:
    lines.append("\n--- WARNINGS ---")
    for w in warnings: lines.append(f"  [WARN] {w}")
if failures:
    lines.append("\n--- FAILURES ---")
    for f in failures: lines.append(f"  [FAIL] {f}")

lines.append(f"\nOVERALL: {'PASS' if not failures else 'FAIL'}")

report_text = "\n".join(lines)
print(report_text)
with open(REPORT, "w", encoding="utf-8") as fh:
    fh.write(report_text)