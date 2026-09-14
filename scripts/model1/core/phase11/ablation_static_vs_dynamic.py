"""
Phase 11A: Static vs Dynamic Comparison Table
- Dev results (phase5): B0, B1, B2_a20, B2_a40(=B*), B3_ungated, B3_PG
- B4 (phase7): offline policy metrics only; not deployed (safety threshold violated)
- Held-out test results (phase9): B0, B1, B2_a40(=BSTAR), B3_PG(=B3)
  B2_a20 and B3_ungated not evaluated on held-out test → marked "—"
"""
import os, csv

ROOT = os.environ.get("CAR_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
P5   = os.path.join(ROOT, "outputs", "phase5")
P7   = os.path.join(ROOT, "outputs", "phase7")
P9   = os.path.join(ROOT, "outputs", "phase9")
OUT  = os.path.join(ROOT, "outputs", "phase11")
os.makedirs(OUT, exist_ok=True)

def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def idx(rows, *keys):
    return {tuple(r[k] for k in keys): r for r in rows}

def pct(v):
    try:    return f"{float(v)*100:.2f}%"
    except: return "—"

def fmt(v, d=4):
    try:    return f"{float(v):.{d}f}"
    except: return "—"

# Load sources
p5_main = idx(load_csv(os.path.join(P5, "phase5_main_results.csv")),       "method")
p5_cwe  = idx(load_csv(os.path.join(P5, "phase5_per_cwe_results.csv")),    "method", "cwe_id")
p9_main = idx(load_csv(os.path.join(P9, "phase9_overall_results.csv")),    "method")
p9_cwe  = idx(load_csv(os.path.join(P9, "phase9_per_cwe_results.csv")),    "method", "cwe_id")
p7_ckpt = load_csv(os.path.join(P7, "checkpoint_validation_metrics.csv"))

# Best B4 checkpoint = linear_epoch_01 (highest RCR before collapse)
best_b4 = next((r for r in p7_ckpt if r["checkpoint"] == "linear_bandit_epoch_01.ckpt"), None)

CWES = ["CWE-120", "CWE-89", "CWE-327", "CWE-338"]

# (key, label, ctrl, p5_key, p9_key, note)
# p9_key=None means not evaluated on held-out test
METHODS = [
    ("B0",         "B0 (Raw baseline)",              "none",    "B0",         "B0",    ""),
    ("B1",         "B1 (Secure prompting)",           "static",  "B1",         "B1",    ""),
    ("B2_a20",     "B2 (alpha=20)",                   "static",  "B2_a20",     None,    "not eval. on test"),
    ("B2_a40",     "B2_a40 / B* (final method)",      "static",  "B2_a40",     "BSTAR", ""),
    ("B3_ungated", "B3-ungated (semantic, no gate)",   "static",  "B3_ungated", None,    "not eval. on test"),
    ("B3_PG",      "B3-PG (plausibility gate)",        "static",  "B3_PG",      "B3",    ""),
    ("B4",         "B4 (contextual bandit)",            "dynamic", None,         None,    "not deployed; offline eval only"),
]

rows_out = []
for key, label, ctrl, p5k, p9k, note in METHODS:
    r = {"method_key": key, "method_label": label, "control_type": ctrl, "note": note}

    # ── Dev (phase5) ──────────────────────────────────────────────────────────
    if p5k and (p5k,) in p5_main:
        m = p5_main[(p5k,)]
        r["dev_vuln_count"]   = m["corrected_vuln"]
        r["dev_vrr"]          = fmt(m["corrected_vrr"])
        r["dev_corruption"]   = pct(m["corruption_rate"])
        r["dev_validity"]     = pct(m["validity_rate"])
        r["dev_safe_to_safe"] = pct(m["safe_to_safe"])
        for c in CWES:
            v = p5_cwe.get((p5k, c), {})
            r[f"dev_vrr_{c}"] = fmt(v.get("cwe_vrr","")) if v and v.get("cwe_vrr") else "—"
    elif key == "B4":
        # B4: show offline policy metrics instead of generation metrics
        r["dev_vuln_count"]   = "—"
        r["dev_vrr"]          = "—"
        r["dev_corruption"]   = "—"
        r["dev_validity"]     = "—"
        r["dev_safe_to_safe"] = "—"
        for c in CWES:
            r[f"dev_vrr_{c}"] = "—"
    else:
        r["dev_vuln_count"] = "—"; r["dev_vrr"] = "—"
        r["dev_corruption"] = "—"; r["dev_validity"] = "—"; r["dev_safe_to_safe"] = "—"
        for c in CWES: r[f"dev_vrr_{c}"] = "—"

    # ── B4 offline policy metrics ─────────────────────────────────────────────
    if key == "B4" and best_b4:
        r["b4_best_rcr"]         = fmt(best_b4["rcr"], 4)
        r["b4_noop_rate"]        = pct(best_b4["noop_rate"])
        r["b4_unsafe_rate"]      = pct(best_b4["unsafe_rate"])
        r["b4_n_actionable"]     = best_b4["n_actionable_evaluated"]
        r["b4_reject_reason"]    = "unsafe_rate=1.86% > 1% threshold; all other ckpts noop_rate=100%"
    else:
        r["b4_best_rcr"] = "—"; r["b4_noop_rate"] = "—"
        r["b4_unsafe_rate"] = "—"; r["b4_n_actionable"] = "—"; r["b4_reject_reason"] = "—"

    # ── Held-out test (phase9) ────────────────────────────────────────────────
    if p9k and (p9k,) in p9_main:
        m9 = p9_main[(p9k,)]
        r["test_vuln_count"]   = m9["vuln_count"]
        r["test_vrr"]          = fmt(m9["VRR"])
        r["test_corruption"]   = pct(m9["corruption_rate"])
        r["test_validity"]     = pct(m9["validity_rate"])
        r["test_safe_to_safe"] = pct(m9["safe_to_safe_rate"])
        for c in CWES:
            v = p9_cwe.get((p9k, c), {})
            r[f"test_vrr_{c}"] = fmt(v.get("VRR_cwe","")) if v and v.get("VRR_cwe") else "—"
    else:
        r["test_vuln_count"] = "—"; r["test_vrr"] = "—"
        r["test_corruption"] = "—"; r["test_validity"] = "—"; r["test_safe_to_safe"] = "—"
        for c in CWES: r[f"test_vrr_{c}"] = "—"

    rows_out.append(r)

# ── Write CSV ─────────────────────────────────────────────────────────────────
fields = (
    ["method_key","method_label","control_type","note"]
    + ["dev_vuln_count","dev_vrr","dev_corruption","dev_validity","dev_safe_to_safe"]
    + [f"dev_vrr_{c}" for c in CWES]
    + ["b4_best_rcr","b4_noop_rate","b4_unsafe_rate","b4_n_actionable","b4_reject_reason"]
    + ["test_vuln_count","test_vrr","test_corruption","test_validity","test_safe_to_safe"]
    + [f"test_vrr_{c}" for c in CWES]
)
out_path = os.path.join(OUT, "phase11_ablation_static_vs_dynamic.csv")
with open(out_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader(); w.writerows(rows_out)
print(f"Written: {out_path}\n")

# ── Console print ─────────────────────────────────────────────────────────────
S = "-"*132
for split, vc, vrr, corr, val, sts, pfx, n, b0v in [
    ("DEV (N=1341, B0-vuln=60)",  "dev_vuln_count",  "dev_vrr",  "dev_corruption",  "dev_validity",  "dev_safe_to_safe",  "dev_vrr_",  1341, 60),
    ("HELD-OUT TEST (N=575, B0-vuln=28)", "test_vuln_count","test_vrr","test_corruption","test_validity","test_safe_to_safe","test_vrr_", 575,  28),
]:
    print(split); print(S)
    print(f"  {'Method':<38} {'Type':>7} {'Vuln':>5} {'VRR':>7} {'Corr%':>7} {'Valid%':>7}  "
          + "  ".join(f"{c:>8}" for c in CWES) + "  Note")
    print(S)
    for r in rows_out:
        cwe_s = "  ".join(f"{r[pfx+c]:>8}" for c in CWES)
        print(f"  {r['method_label']:<38} {r['control_type']:>7} "
              f"{r[vc]:>5} {r[vrr]:>7} {r[corr]:>7} {r[val]:>7}  {cwe_s}  {r['note']}")
    print()

print("B4 OFFLINE POLICY METRICS (best checkpoint: linear_epoch_01)")
print(S)
b4r = next(r for r in rows_out if r["method_key"]=="B4")
print(f"  RCR={b4r['b4_best_rcr']}  noop_rate={b4r['b4_noop_rate']}  "
      f"unsafe_rate={b4r['b4_unsafe_rate']}  n_actionable={b4r['b4_n_actionable']}")
print(f"  Rejection reason: {b4r['b4_reject_reason']}")
print(f"  Structural finding: reward sparsity (3.9% actionable prompts) prevented policy learning.")
