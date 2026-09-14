"""
Phase 11C: Fixed Layer (L19) vs Dynamic Layer (B*) Comparison
B*: dynamic layer per CWE (L19 for CWE-120, L23 for CWE-327/89/338), alpha=40
Fixed-L19: single layer L19 for all active CWEs, alpha=40

B* metrics taken directly from frozen phase5 results.
Output: outputs/phase11/phase11_ablation_layer_fixed_vs_dynamic.csv
"""
import json, csv, os

ROOT = os.environ.get("CAR_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
P2   = os.path.join(ROOT, "outputs", "phase2")
P11  = os.path.join(ROOT, "outputs", "phase11")

DERIVATION_CWES = {"CWE-120","CWE-125","CWE-787","CWE-190","CWE-476","CWE-89","CWE-79","CWE-327"}
ACTIVE_CWES     = ["CWE-120", "CWE-89", "CWE-327", "CWE-338"]

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def fmt(v, d=4):
    try:    return round(float(v), d)
    except: return None

# ── Load ICD ──────────────────────────────────────────────────────────────────
b0_icd     = {r["prompt_id"]: r for r in load_json(os.path.join(P2, "baseline_dev_icd.json"))}
l19_icd    = {r["prompt_id"]: r for r in load_json(os.path.join(P11, "fixed_l19_dev_icd.json"))}
l19_out    = {r["prompt_id"]: r for r in load_json(os.path.join(P11, "fixed_l19_dev_outputs.json"))}

print(f"B0 ICD:      {len(b0_icd)} records")
print(f"Fixed-L19:   {len(l19_icd)} records")

b0_vuln_ids = {pid for pid, r in b0_icd.items()
               if r.get("is_vulnerable") and not r.get("skipped")}
b0_safe_ids = {pid for pid, r in b0_icd.items()
               if not r.get("is_vulnerable") and not r.get("skipped")}
print(f"B0 vulnerable: {len(b0_vuln_ids)}  B0 safe: {len(b0_safe_ids)}")

# ── Compute Fixed-L19 metrics ─────────────────────────────────────────────────
method_vuln = set()
for pid in b0_vuln_ids:
    if pid not in l19_icd:
        method_vuln.add(pid); continue
    r = l19_icd[pid]
    if r.get("skipped") or not l19_out.get(pid, {}).get("generated_code","").strip():
        method_vuln.add(pid)
    elif r.get("is_vulnerable"):
        method_vuln.add(pid)

vrr = 1 - len(method_vuln) / len(b0_vuln_ids) if b0_vuln_ids else 0.0

corrupted = {pid for pid in b0_safe_ids
             if pid in l19_icd and l19_icd[pid].get("is_vulnerable")
             and not l19_icd[pid].get("skipped")}
corruption_rate = len(corrupted) / len(b0_safe_ids) if b0_safe_ids else 0.0

n_empty    = sum(1 for r in l19_out.values() if not r.get("generated_code","").strip())
validity   = 1 - n_empty / len(l19_out)

def cwe_vrr(icd_dict, b0_icd, cwe):
    b0_cwe = [pid for pid, r in b0_icd.items()
              if cwe in r.get("vulnerable_cwes",[]) and not r.get("skipped")]
    if not b0_cwe: return None
    still = [pid for pid in b0_cwe
             if pid in icd_dict and cwe in icd_dict[pid].get("vulnerable_cwes",[])]
    return fmt(1 - len(still) / len(b0_cwe))

l19_metrics = {
    "method":                "Fixed-L19",
    "layer_config":          "L19 for all CWEs",
    "b0_vuln":               len(b0_vuln_ids),
    "method_vuln_corrected": len(method_vuln),
    "vrr":                   fmt(vrr),
    "corrupted":             len(corrupted),
    "corruption_rate":       fmt(corruption_rate),
    "n_empty":               n_empty,
    "validity_rate":         fmt(validity),
}
for c in ACTIVE_CWES:
    l19_metrics[f"vrr_{c}"] = cwe_vrr(l19_icd, b0_icd, c)

# ── B* metrics from frozen phase5 results ────────────────────────────────────
bstar_metrics = {
    "method":                "B* (Dynamic layer)",
    "layer_config":          "L19→CWE-120, L23→CWE-327/89/338",
    "b0_vuln":               60,
    "method_vuln_corrected": 27,
    "vrr":                   0.55,
    "corrupted":             3,
    "corruption_rate":       0.0023,
    "n_empty":               26,
    "validity_rate":         0.9806,
    "vrr_CWE-120":           0.6364,
    "vrr_CWE-89":            1.0,
    "vrr_CWE-327":           0.3448,
    "vrr_CWE-338":           None,
}

# ── Print ─────────────────────────────────────────────────────────────────────
S = "-"*90
print(f"\n{'='*90}")
print("PHASE 11C: Fixed Layer (L19) vs Dynamic Layer (B*)")
print(f"{'='*90}")
print(f"\n{'Metric':<30} {'B* (dynamic)':>15} {'Fixed-L19':>15} {'Delta':>10}")
print(S)
for key, label in [
    ("vrr",             "Overall VRR"),
    ("corruption_rate", "Corruption rate"),
    ("validity_rate",   "Validity rate"),
] + [(f"vrr_{c}", f"VRR {c}") for c in ACTIVE_CWES]:
    bv = bstar_metrics.get(key)
    lv = l19_metrics.get(key)
    try:    delta = f"{float(lv)-float(bv):+.4f}"
    except: delta = "N/A"
    print(f"  {label:<28} {str(bv):>15} {str(lv):>15} {delta:>10}")

print(S)
print(f"  {'B0 vuln count':<28} {bstar_metrics['b0_vuln']:>15} {l19_metrics['b0_vuln']:>15}")
print(f"  {'Method vuln (corrected)':<28} {bstar_metrics['method_vuln_corrected']:>15} {l19_metrics['method_vuln_corrected']:>15}")
print(f"  {'Corrupted (safe→vuln)':<28} {bstar_metrics['corrupted']:>15} {l19_metrics['corrupted']:>15}")
print(f"  {'Empty outputs':<28} {bstar_metrics['n_empty']:>15} {l19_metrics['n_empty']:>15}")

print(f"\nLayer configuration:")
print(f"  B*:        L19 for CWE-120 (feature 14193), L23 for CWE-327/89/338")
print(f"  Fixed-L19: L19 for ALL active CWEs")
print(f"             CWE-327: feature 16897, CWE-89: feature 11462, CWE-338: feature 151")

# ── Write CSV ─────────────────────────────────────────────────────────────────
fields = (
    ["method","layer_config","b0_vuln","method_vuln_corrected","vrr",
     "corrupted","corruption_rate","n_empty","validity_rate"]
    + [f"vrr_{c}" for c in ACTIVE_CWES]
)
out_path = os.path.join(P11, "phase11_ablation_layer_fixed_vs_dynamic.csv")
with open(out_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    w.writerow(bstar_metrics)
    w.writerow(l19_metrics)

print(f"\nWritten: {out_path}")
