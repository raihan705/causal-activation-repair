"""
Phase 4 — Step 4.5
Semantic assignment for validated features per (layer, CWE).

For each validated feature:
  - Collect high-activation contexts from CyberSecEval dev prompts + CVE pairs
  - Compute P(CWE label | feature active) and AUC / mutual information
  - Assign primary CWE label + optional root-cause tags

Outputs: feature_semantics_layer_{l}.json, feature_tag_metrics_layer_{l}.csv
"""

import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path.cwd()
PHASE2_DIR   = PROJECT_ROOT / "outputs/phase2"
PHASE4_DIR   = PROJECT_ROOT / "outputs/phase4"
DATA_DIR     = PROJECT_ROOT / "data"
LATENT_DIR   = DATA_DIR / "activations/latent"
OUT_DIR      = PHASE4_DIR

# ── Config ─────────────────────────────────────────────────────────────────────
LAYERS      = [16, 19, 23]
CYBER_CWES  = [120, 327, 89, 338]
CVE_CWES    = [79, 125, 787, 190, 476]
ALL_CWES    = CYBER_CWES + CVE_CWES

ACTIVE_THRESH = 0.01   # feature considered active if value > this
AUC_MIN       = 0.55   # minimum AUC to assign a CWE label confidently
MI_MIN        = 0.01   # minimum mutual information

# Root-cause tag taxonomy
CWE_TAGS = {
    120: ["buffer_overflow", "memory_unsafe", "bounds_check"],
    327: ["crypto_weak", "insecure_algorithm", "crypto_api"],
    89:  ["sql_injection", "input_validation", "string_format"],
    338: ["prng_weak", "crypto_random", "predictable_seed"],
    79:  ["xss", "input_validation", "output_encoding"],
    125: ["out_of_bounds_read", "memory_unsafe", "bounds_check"],
    787: ["out_of_bounds_write", "memory_unsafe", "buffer_overflow"],
    190: ["integer_overflow", "arithmetic_error", "type_check"],
    476: ["null_dereference", "pointer_check", "memory_unsafe"],
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def load_latents(path):
    t = torch.load(path, map_location="cpu")
    if isinstance(t, dict):
        return t["latents"].float()
    return t.float()

def get_feature_activations(latents, feature_id):
    """Returns (N,) activation values for one feature."""
    return latents[:, feature_id].numpy()

def compute_auc(activations, labels):
    """AUC of feature activation as a binary classifier for label."""
    if len(np.unique(labels)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(labels, activations))
    except Exception:
        return 0.5

def compute_mi(activations, labels):
    """Mutual information between feature activation and binary label."""
    try:
        mi = mutual_info_classif(
            activations.reshape(-1, 1), labels, discrete_features=False, random_state=42
        )
        return float(mi[0])
    except Exception:
        return 0.0

def build_multi_cwe_dataset(layer, feature_id):
    """
    Build a dataset of (activation_value, cwe_label) across all CWEs
    using both cyber_dev and CVE latents.
    Returns dict: {cwe: {"activations": np.array, "labels": np.array}}
    """
    cwe_data = {}
    for cwe in ALL_CWES:
        unsafe_path = PHASE4_DIR / f"unsafe_latents_layer_{layer}_cwe{cwe}.pt"
        safe_path   = PHASE4_DIR / f"safe_latents_layer_{layer}_cwe{cwe}.pt"
        if not unsafe_path.exists() or not safe_path.exists():
            continue
        unsafe = load_latents(unsafe_path)
        safe   = load_latents(safe_path)

        # Balance: subsample safe to match unsafe size
        n_unsafe = unsafe.shape[0]
        n_safe   = safe.shape[0]
        if n_safe > n_unsafe:
            idx = torch.randperm(n_safe,
                generator=torch.Generator().manual_seed(42))[:n_unsafe]
            safe = safe[idx]

        unsafe_act = get_feature_activations(unsafe, feature_id)
        safe_act   = get_feature_activations(safe,   feature_id)

        activations = np.concatenate([unsafe_act, safe_act])
        labels      = np.concatenate([
            np.ones(len(unsafe_act)), np.zeros(len(safe_act))
        ])
        cwe_data[cwe] = {"activations": activations, "labels": labels}
    return cwe_data

# ── Main ───────────────────────────────────────────────────────────────────────
all_semantics = {}   # layer → {cwe → [feature_semantic_dicts]}
all_tag_rows  = []

for layer in LAYERS:
    print(f"\n=== Layer {layer} ===")
    validated = json.load(
        open(PHASE4_DIR / f"validated_features_layer_{layer}.json")
    )
    layer_semantics = {}

    for cwe in ALL_CWES:
        key = f"CWE-{cwe}"
        features = validated.get(key, [])
        if not features:
            continue

        print(f"  {key}: assigning semantics to {len(features)} validated features")
        cwe_semantics = []

        for rec in features:
            fid = rec["feature_id"]

            # Build multi-CWE activation dataset
            cwe_data = build_multi_cwe_dataset(layer, fid)

            # Compute AUC and MI per CWE
            cwe_aucs = {}
            cwe_mis  = {}
            for c, data in cwe_data.items():
                auc = compute_auc(data["activations"], data["labels"])
                mi  = compute_mi(data["activations"],  data["labels"])
                cwe_aucs[c] = auc
                cwe_mis[c]  = mi

            # Primary CWE: highest AUC
            if cwe_aucs:
                primary_cwe = max(cwe_aucs, key=cwe_aucs.get)
                primary_auc = cwe_aucs[primary_cwe]
            else:
                primary_cwe = cwe
                primary_auc = 0.5

            # Check if primary CWE matches expected CWE
            cwe_match = (primary_cwe == cwe)

            # Assign root-cause tags from primary CWE
            # If AUC is above threshold use primary CWE tags
            # otherwise assign tags from the validated CWE (fallback)
            if primary_auc >= AUC_MIN:
                assigned_cwe  = primary_cwe
                assigned_tags = CWE_TAGS.get(primary_cwe, ["generic_security"])
            else:
                assigned_cwe  = cwe  # fallback to validated CWE
                assigned_tags = CWE_TAGS.get(cwe, ["generic_security"])

            # Cross-CWE activation: list all CWEs where AUC > AUC_MIN
            cross_cwes = [c for c, a in cwe_aucs.items() if a >= AUC_MIN and c != assigned_cwe]

            semantic_rec = {
                "feature_id":       fid,
                "layer":            layer,
                "validated_cwe":    f"CWE-{cwe}",
                "primary_cwe":      f"CWE-{primary_cwe}",
                "primary_auc":      round(primary_auc, 4),
                "cwe_match":        cwe_match,
                "assigned_cwe":     f"CWE-{assigned_cwe}",
                "assigned_tags":    assigned_tags,
                "cross_cwes":       [f"CWE-{c}" for c in cross_cwes],
                "is_cross_cwe":     len(cross_cwes) > 0,
                "repair_rate":      rec.get("repair_rate", 0),
                "corrupt_rate":     rec.get("corrupt_rate", 0),
                "score":            rec.get("score", 0),
                "all_cwe_aucs":     {f"CWE-{c}": round(a, 4) for c, a in cwe_aucs.items()},
                "all_cwe_mis":      {f"CWE-{c}": round(m, 4) for c, m in cwe_mis.items()},
            }
            cwe_semantics.append(semantic_rec)

            # Tag metrics row
            all_tag_rows.append({
                "feature_id":    fid,
                "layer":         layer,
                "validated_cwe": f"CWE-{cwe}",
                "primary_cwe":   f"CWE-{primary_cwe}",
                "primary_auc":   round(primary_auc, 4),
                "cwe_match":     cwe_match,
                "n_cross_cwes":  len(cross_cwes),
                "assigned_tags": "|".join(assigned_tags),
                "repair_rate":   rec.get("repair_rate", 0),
                "corrupt_rate":  rec.get("corrupt_rate", 0),
            })

            match_str = "match" if cwe_match else f"→CWE-{primary_cwe}"
            print(f"    feat {fid:6d}: primary_auc={primary_auc:.3f} "
                  f"({match_str}) cross={len(cross_cwes)} tags={assigned_tags[:2]}")

        layer_semantics[key] = cwe_semantics
    all_semantics[layer] = layer_semantics

    # Save per-layer semantics
    out_json = OUT_DIR / f"feature_semantics_layer_{layer}.json"
    with open(out_json, "w") as f:
        json.dump(layer_semantics, f, indent=2)
    print(f"  Saved {out_json.name}")

# Save tag metrics
tag_df = pd.DataFrame(all_tag_rows)
tag_csv = OUT_DIR / "feature_tag_metrics_all.csv"
tag_df.to_csv(tag_csv, index=False)

# Summary
print("\n=== Semantic Assignment Summary ===")
print(f"{'Layer':<8} {'CWE':<10} {'N_feats':<9} {'CWE_match%':<12} {'Avg_AUC':<10} {'Cross_CWE%'}")
print("-" * 60)
for layer in LAYERS:
    for cwe in ALL_CWES:
        key = f"CWE-{cwe}"
        subset = tag_df[(tag_df["layer"] == layer) & (tag_df["validated_cwe"] == key)]
        if subset.empty:
            continue
        match_pct  = round(subset["cwe_match"].mean() * 100, 1)
        avg_auc    = round(subset["primary_auc"].mean(), 3)
        cross_pct  = round((subset["n_cross_cwes"] > 0).mean() * 100, 1)
        print(f"{layer:<8} {key:<10} {len(subset):<9} {match_pct:<12} {avg_auc:<10} {cross_pct}")

print("\nStep 4.5 complete.")