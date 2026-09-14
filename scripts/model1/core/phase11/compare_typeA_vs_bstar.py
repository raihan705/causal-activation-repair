"""
Phase 11B: Compare Type-A statistical groups vs B* (B2_a40)
Computes VRR, corruption, validity for Type-A and compares to B* from Phase 5.
Output: outputs/phase11/phase11_ablation_semantic_vs_statistical_groups.csv
"""
import json, csv, os
from collections import defaultdict

ROOT = os.environ.get("CAR_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
P2    = os.path.join(ROOT, "outputs", "phase2")
P5    = os.path.join(ROOT, "outputs", "phase5")
P11   = os.path.join(ROOT, "outputs", "phase11")

DERIVATION_CWES = {"CWE-120","CWE-125","CWE-787","CWE-190","CWE-476","CWE-89","CWE-79","CWE-327"}
ACTIVE_CWES     = ["CWE-120", "CWE-89", "CWE-327", "CWE-338"]

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def fmt(v, d=4):
    try:    return round(float(v), d)
    except: return None

# ── Load ICD results ──────────────────────────────────────────────────────────
# B0 baseline (phase5)
b0_icd  = {r["prompt_id"]: r for r in load_json(os.path.join(P2, "baseline_dev_icd.json"))}
# B* (B2_a40) ICD (phase5)
bstar_icd_path = os.path.join(P5, "thea_static_dev_icd_a40.json")
if not os.path.exists(bstar_icd_path):
    bstar_icd_path = os.path.join(P5, "thea_static_dev_icd.json")
bstar_icd = {r["prompt_id"]: r for r in load_json(bstar_icd_path)}
# Type-A ICD (phase11)
typeA_icd = {r["prompt_id"]: r for r in load_json(os.path.join(P11, "typeA_dev_icd.json"))}
# Type-A outputs (to check steered flag)
typeA_out = {r["prompt_id"]: r for r in load_json(os.path.join(P11, "typeA_dev_outputs.json"))}

print(f"B0 ICD:    {len(b0_icd)} records")
print(f"B* ICD:    {len(bstar_icd)} records")
print(f"TypeA ICD: {len(typeA_icd)} records")

# ── Identify B0-vulnerable prompt IDs ─────────────────────────────────────────
b0_vuln_ids = {
    pid for pid, r in b0_icd.items()
    if r.get("is_vulnerable") and not r.get("skipped")
}
b0_safe_ids = {
    pid for pid, r in b0_icd.items()
    if not r.get("is_vulnerable") and not r.get("skipped")
}
print(f"\nB0 vulnerable: {len(b0_vuln_ids)}  B0 safe: {len(b0_safe_ids)}")

# ── Compute metrics for a method's ICD dict ───────────────────────────────────
def compute_metrics(icd_dict, b0_vuln_ids, b0_safe_ids, label):
    all_pids   = set(icd_dict.keys())
    n_total    = len(all_pids)
    n_skipped  = sum(1 for r in icd_dict.values() if r.get("skipped"))
    n_scanned  = n_total - n_skipped

    # Vulnerable among B0-vulnerable prompts (for VRR)
    method_vuln = {
        pid for pid in b0_vuln_ids
        if pid in icd_dict and icd_dict[pid].get("is_vulnerable") and not icd_dict[pid].get("skipped")
    }
    # Also count empties among B0-vulnerable as still vulnerable (fallback)
    method_vuln_corrected = set(method_vuln)
    for pid in b0_vuln_ids:
        if pid in typeA_out and not typeA_out[pid].get("generated_code","").strip():
            method_vuln_corrected.add(pid)

    b0_v = len(b0_vuln_ids)
    vrr  = 1 - len(method_vuln_corrected) / b0_v if b0_v > 0 else 0.0

    # Corruption: B0-safe → vulnerable under method
    corrupted = {
        pid for pid in b0_safe_ids
        if pid in icd_dict and icd_dict[pid].get("is_vulnerable") and not icd_dict[pid].get("skipped")
    }
    corruption_rate = len(corrupted) / len(b0_safe_ids) if b0_safe_ids else 0.0

    # Validity
    n_empty   = sum(1 for r in typeA_out.values() if not r.get("generated_code","").strip()) if label == "TypeA" else None
    validity  = 1 - (n_empty / n_total) if n_empty is not None else None

    return {
        "method":          label,
        "n_total":         n_total,
        "n_scanned":       n_scanned,
        "n_skipped":       n_skipped,
        "b0_vuln":         b0_v,
        "method_vuln_corrected": len(method_vuln_corrected),
        "vrr":             fmt(vrr),
        "corrupted":       len(corrupted),
        "corruption_rate": fmt(corruption_rate),
        "n_empty":         n_empty if n_empty is not None else "N/A",
        "validity_rate":   fmt(validity) if validity is not None else "N/A",
    }

# ── Per-CWE VRR ───────────────────────────────────────────────────────────────
def cwe_vrr(icd_dict, b0_icd, cwe):
    b0_cwe_vuln = [
        pid for pid, r in b0_icd.items()
        if cwe in r.get("vulnerable_cwes", []) and not r.get("skipped")
    ]
    if not b0_cwe_vuln:
        return None
    method_still_vuln = [
        pid for pid in b0_cwe_vuln
        if pid in icd_dict and cwe in icd_dict[pid].get("vulnerable_cwes", [])
    ]
    return fmt(1 - len(method_still_vuln) / len(b0_cwe_vuln))

# ── Compute ───────────────────────────────────────────────────────────────────
typeA_metrics = compute_metrics(typeA_icd, b0_vuln_ids, b0_safe_ids, "TypeA")

# B* metrics taken directly from frozen phase5 results (phase5_main_results.csv)
# corrected_vrr=0.55, corrected_vuln=27, corruption_rate=0.0023, validity=0.9806, empty_b0_vuln=0
bstar_metrics = {
    "method":                 "B_STAR",
    "n_total":                1341,
    "n_scanned":              906,
    "n_skipped":              435,
    "b0_vuln":                60,
    "method_vuln_corrected":  27,
    "vrr":                    0.55,
    "corrupted":              3,
    "corruption_rate":        0.0023,
    "n_empty":                26,
    "validity_rate":          0.9806,
}

# B* per-CWE VRR from phase5_per_cwe_results.csv (frozen)
bstar_cwe_vrr = {"CWE-120": 0.6364, "CWE-89": 1.0, "CWE-327": 0.3448, "CWE-338": None}
for cwe in ACTIVE_CWES:
    bstar_metrics[f"vrr_{cwe}"] = bstar_cwe_vrr.get(cwe)
    typeA_metrics[f"vrr_{cwe}"] = cwe_vrr(typeA_icd, b0_icd, cwe)

# ── Print ─────────────────────────────────────────────────────────────────────
S = "-"*90
print(f"\n{'='*90}")
print("PHASE 11B: Type-A Statistical vs B* (B2_a40 = single-feature semantic)")
print(f"{'='*90}")
print(f"\n{'Metric':<30} {'B* (B2_a40)':>15} {'TypeA (stat)':>15} {'Delta':>10}")
print(S)
metrics_to_print = [
    ("vrr",             "Overall VRR"),
    ("corruption_rate", "Corruption rate"),
    ("validity_rate",   "Validity rate"),
] + [(f"vrr_{c}", f"VRR {c}") for c in ACTIVE_CWES]

for key, label in metrics_to_print:
    bv = bstar_metrics.get(key)
    tv = typeA_metrics.get(key)
    try:
        delta = f"{float(tv)-float(bv):+.4f}"
    except:
        delta = "N/A"
    print(f"  {label:<28} {str(bv):>15} {str(tv):>15} {delta:>10}")

print(S)
print(f"  {'B0 vuln count':<28} {bstar_metrics['b0_vuln']:>15} {typeA_metrics['b0_vuln']:>15}")
print(f"  {'Method vuln (corrected)':<28} {bstar_metrics['method_vuln_corrected']:>15} {typeA_metrics['method_vuln_corrected']:>15}")
print(f"  {'Corrupted (safe→vuln)':<28} {bstar_metrics['corrupted']:>15} {typeA_metrics['corrupted']:>15}")
print(f"  {'Empty outputs':<28} {bstar_metrics['n_empty']:>15} {typeA_metrics['n_empty']:>15}")

# ── Write CSV ─────────────────────────────────────────────────────────────────
fields = (
    ["method","n_total","n_scanned","n_skipped","b0_vuln",
     "method_vuln_corrected","vrr","corrupted","corruption_rate",
     "n_empty","validity_rate"]
    + [f"vrr_{c}" for c in ACTIVE_CWES]
)
out_path = os.path.join(P11, "phase11_ablation_semantic_vs_statistical_groups.csv")
with open(out_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    w.writerow(bstar_metrics)
    w.writerow(typeA_metrics)

print(f"\nWritten: {out_path}")
print("\nInterpretation:")
print("  B* uses single best feature per CWE at alpha=40 (semantic selection).")
print("  TypeA uses statistical_fallback groups at alpha=20 (no semantic validation).")
print("  Key difference for CWE-120: B* intervenes (feature 14193, L19),")
print("  TypeA does NOT intervene (no Type-A group exists for CWE-120).")
