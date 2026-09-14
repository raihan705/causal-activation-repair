"""
Phase 4 — Step 4.6
Build feature groups per (layer, CWE).

Three group types per plan:
  Type A — statistical fallback: strong statistical separation, weak semantics
  Type B — semantic: shared CWE tags + activation patterns, AUC >= AUC_MIN
  Type C — mixed robust: strong causal repair + low corruption + cross-CWE

Group record fields:
  group_id, layer_id, cwe_id, group_type, feature_ids,
  repair_rate, corruption_rate, semantic_purity,
  recommended_strength_range, tags
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path.cwd()
PHASE4_DIR   = PROJECT_ROOT / "outputs/phase4"
OUT_DIR      = PHASE4_DIR

# ── Config ─────────────────────────────────────────────────────────────────────
LAYERS      = [16, 19, 23]
CYBER_CWES  = [120, 327, 89, 338]
CVE_CWES    = [79, 125, 787, 190, 476]
ALL_CWES    = CYBER_CWES + CVE_CWES

AUC_MIN          = 0.55   # minimum AUC for Type B semantic group
REPAIR_STRONG    = 0.70   # minimum repair_rate for Type C
CORRUPT_STRONG   = 0.10   # maximum corruption_rate for Type C
GROUP_MAX_FEATS  = 5      # max features per group
GROUP_MIN_FEATS  = 1      # min features per group

# Strength ranges: (low, high) additive alpha multipliers relative to base
STRENGTH_RANGES = {
    "cyber_dev": (0.5, 1.5),   # cyber-detected CWEs: wider range ok
    "cve_pairs": (0.3, 1.0),   # CVE-pair CWEs: more conservative
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def mean_metric(features, key):
    vals = [f.get(key, 0) for f in features if f.get(key) is not None]
    return round(float(np.mean(vals)), 4) if vals else 0.0

def semantic_purity(features):
    """Fraction of features whose primary_cwe matches validated_cwe."""
    if not features:
        return 0.0
    matches = sum(1 for f in features
                  if f.get("primary_cwe") == f.get("validated_cwe"))
    return round(matches / len(features), 4)

def build_groups_for_cwe(layer, cwe, semantics, source):
    """
    Build Type A, B, C groups for one (layer, CWE).
    Returns list of group dicts.
    """
    groups = []
    cwe_str = f"CWE-{cwe}"
    strength_range = STRENGTH_RANGES[source]

    if not semantics:
        return groups

    # ── Type B — Semantic group ────────────────────────────────────────────
    # Features with AUC >= AUC_MIN AND primary_cwe matches validated_cwe
    type_b_feats = [
        f for f in semantics
        if f.get("primary_auc", 0) >= AUC_MIN
        and f.get("primary_cwe") == f.get("validated_cwe")
    ][:GROUP_MAX_FEATS]

    if len(type_b_feats) >= GROUP_MIN_FEATS:
        groups.append({
            "group_id":               f"L{layer}_{cwe_str}_B",
            "layer_id":               layer,
            "cwe_id":                 cwe_str,
            "group_type":             "semantic",
            "feature_ids":            [f["feature_id"] for f in type_b_feats],
            "repair_rate":            mean_metric(type_b_feats, "repair_rate"),
            "corruption_rate":        mean_metric(type_b_feats, "corrupt_rate"),
            "semantic_purity":        semantic_purity(type_b_feats),
            "avg_auc":                mean_metric(type_b_feats, "primary_auc"),
            "recommended_strength_range": strength_range,
            "tags":                   list(set(
                t for f in type_b_feats
                for t in f.get("assigned_tags", [])
            )),
            "source": source,
        })

    # ── Type C — Mixed robust group ────────────────────────────────────────
    # Features with repair >= REPAIR_STRONG AND corruption <= CORRUPT_STRONG
    # regardless of CWE match — includes cross-CWE features
    type_c_feats = [
        f for f in semantics
        if f.get("repair_rate", 0) >= REPAIR_STRONG
        and f.get("corrupt_rate", 1.0) <= CORRUPT_STRONG
    ][:GROUP_MAX_FEATS]

    if len(type_c_feats) >= GROUP_MIN_FEATS:
        groups.append({
            "group_id":               f"L{layer}_{cwe_str}_C",
            "layer_id":               layer,
            "cwe_id":                 cwe_str,
            "group_type":             "mixed_robust",
            "feature_ids":            [f["feature_id"] for f in type_c_feats],
            "repair_rate":            mean_metric(type_c_feats, "repair_rate"),
            "corruption_rate":        mean_metric(type_c_feats, "corrupt_rate"),
            "semantic_purity":        semantic_purity(type_c_feats),
            "avg_auc":                mean_metric(type_c_feats, "primary_auc"),
            "recommended_strength_range": strength_range,
            "tags":                   list(set(
                t for f in type_c_feats
                for t in f.get("assigned_tags", [])
            )),
            "source": source,
        })

    # ── Type A — Statistical fallback ─────────────────────────────────────
    # All validated features ranked by repair_rate (fallback if B and C weak)
    type_a_feats = sorted(
        semantics, key=lambda f: f.get("repair_rate", 0), reverse=True
    )[:GROUP_MAX_FEATS]

    if len(type_a_feats) >= GROUP_MIN_FEATS:
        groups.append({
            "group_id":               f"L{layer}_{cwe_str}_A",
            "layer_id":               layer,
            "cwe_id":                 cwe_str,
            "group_type":             "statistical_fallback",
            "feature_ids":            [f["feature_id"] for f in type_a_feats],
            "repair_rate":            mean_metric(type_a_feats, "repair_rate"),
            "corruption_rate":        mean_metric(type_a_feats, "corrupt_rate"),
            "semantic_purity":        semantic_purity(type_a_feats),
            "avg_auc":                mean_metric(type_a_feats, "primary_auc"),
            "recommended_strength_range": (strength_range[0], strength_range[1] * 0.8),
            "tags":                   list(set(
                t for f in type_a_feats
                for t in f.get("assigned_tags", [])
            )),
            "source": source,
        })

    return groups

# ── Main ──────────────────────────────────────────────────────────────────────
all_groups   = {}
summary_rows = []

for layer in LAYERS:
    print(f"\n=== Layer {layer} ===")
    semantics_data = json.load(
        open(PHASE4_DIR / f"feature_semantics_layer_{layer}.json")
    )
    layer_groups = {}

    for cwe in ALL_CWES:
        key    = f"CWE-{cwe}"
        source = "cyber_dev" if cwe in CYBER_CWES else "cve_pairs"
        feats  = semantics_data.get(key, [])

        if not feats:
            print(f"  [SKIP] {key}: no semantic features")
            continue

        groups = build_groups_for_cwe(layer, cwe, feats, source)
        layer_groups[key] = groups

        for g in groups:
            summary_rows.append({
                "group_id":        g["group_id"],
                "layer":           layer,
                "cwe_id":          key,
                "group_type":      g["group_type"],
                "n_features":      len(g["feature_ids"]),
                "repair_rate":     g["repair_rate"],
                "corruption_rate": g["corruption_rate"],
                "semantic_purity": g["semantic_purity"],
                "avg_auc":         g["avg_auc"],
                "source":          source,
            })
            print(f"  {g['group_id']}: type={g['group_type']} "
                  f"n={len(g['feature_ids'])} "
                  f"repair={g['repair_rate']:.3f} "
                  f"corrupt={g['corruption_rate']:.3f} "
                  f"purity={g['semantic_purity']:.2f}")

    all_groups[layer] = layer_groups

    out_json = OUT_DIR / f"feature_groups_layer_{layer}.json"
    with open(out_json, "w") as f:
        json.dump(layer_groups, f, indent=2)
    print(f"  Saved {out_json.name}")

# Summary
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_DIR / "feature_groups_summary.csv", index=False)

print("\n=== Feature Groups Summary ===")
print(summary_df.to_string(index=False))

# Count deployable-eligible groups (repair > 0, corrupt < 0.20)
eligible = summary_df[
    (summary_df["repair_rate"]     > 0) &
    (summary_df["corruption_rate"] < 0.20)
]
n_cwes_with_groups = eligible["cwe_id"].nunique()
print(f"\nCWEs with at least one eligible group: {n_cwes_with_groups}/9")

# Check cyber CWEs specifically
for cwe in CYBER_CWES:
    key = f"CWE-{cwe}"
    has = eligible[eligible["cwe_id"] == key]
    print(f"  {key}: {len(has)} eligible groups across layers")

print("\nStep 4.6 complete.")