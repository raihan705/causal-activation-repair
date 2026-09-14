"""
Phase 5 Step 5.8 - Method comparison and B* selection
Reads phase5_main_results.csv and phase5_per_cwe_results.csv
Applies plan-defined selection rule for B*
Output: outputs/phase5/phase5_comparison_report.txt
"""

import csv
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path.cwd()
P5_DIR       = PROJECT_ROOT / "outputs/phase5"

KEY_CWES = {"CWE-327", "CWE-89", "CWE-120", "CWE-338"}


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    main_results  = read_csv(P5_DIR / "phase5_main_results.csv")
    per_cwe       = read_csv(P5_DIR / "phase5_per_cwe_results.csv")
    plausibility  = read_csv(P5_DIR / "phase5_plausibility_effect.csv")

    # Index by method
    by_method = {r["method"]: r for r in main_results}

    # Per-CWE index: method -> cwe -> row
    cwe_by_method = {}
    for r in per_cwe:
        m = r["method"]
        if m not in cwe_by_method:
            cwe_by_method[m] = {}
        cwe_by_method[m][r["cwe_id"]] = r

    plaus = {r["metric"]: r["value"] for r in plausibility}

    lines = []
    lines.append("=" * 70)
    lines.append("PHASE 5 METHOD COMPARISON REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 70)

    lines.append("\n--- OVERALL RESULTS ---")
    lines.append(f"{'Method':<15} {'VRR':>8} {'CorrVRR':>9} {'Corrupt':>8} {'Validity':>9}")
    lines.append("-" * 55)
    for r in main_results:
        lines.append(
            f"{r['method']:<15} {float(r['vrr']):>8.1%} "
            f"{float(r['corrected_vrr']):>9.1%} "
            f"{float(r['corruption_rate']):>8.1%} "
            f"{float(r['validity_rate']):>9.1%}"
        )

    lines.append("\n--- PER KEY-CWE RESULTS (CWE-327, 89, 120, 338) ---")
    header = f"{'Method':<15}" + "".join(f"{c:>12}" for c in sorted(KEY_CWES))
    lines.append(header)
    lines.append("-" * (15 + 12 * len(KEY_CWES)))
    for method in ["B0","B1","B2_a20","B2_a40","B3_ungated","B3_PG"]:
        row_str = f"{method:<15}"
        for cwe in sorted(KEY_CWES):
            cwe_row = cwe_by_method.get(method, {}).get(cwe, {})
            count = cwe_row.get("method_count", "N/A")
            row_str += f"{str(count):>12}"
        lines.append(row_str)

    lines.append("\n--- PLAUSIBILITY GATE ANALYSIS ---")
    lines.append(f"Gate activation rate:    {float(plaus.get('gate_activation_rate',0)):.1%}")
    lines.append(f"Total steered prompts:   {plaus.get('total_steered_prompts','N/A')}")
    lines.append(f"Total steps:             {plaus.get('total_steps','N/A')}")
    lines.append(f"Total gated steps:       {plaus.get('total_gated_steps','N/A')}")
    lines.append(f"B3_ungated VRR:          {float(plaus.get('B3_ungated_vrr',0)):.1%}")
    lines.append(f"B3_PG VRR:               {float(plaus.get('B3_PG_vrr',0)):.1%}")
    lines.append(f"B3_ungated corrected VRR:{float(plaus.get('B3_ungated_corrected_vrr',0)):.1%}")
    lines.append(f"B3_PG corrected VRR:     {float(plaus.get('B3_PG_corrected_vrr',0)):.1%}")

    lines.append("\n--- B* SELECTION RULE (from experimental plan) ---")
    lines.append("Select B3-PG if:")
    lines.append("  (1) VRR > 0")
    lines.append("  (2) corruption <= B2")
    lines.append("  (3) improvement on >=2 key CWEs (338, 327, 120, 89)")
    lines.append("  (4) validity >= 95%")
    lines.append("Otherwise select B2 (best alpha).")

    lines.append("\n--- EVALUATION OF B3-PG ---")
    b3pg = by_method["B3_PG"]
    b2a40 = by_method["B2_a40"]
    cond1 = float(b3pg["corrected_vrr"]) > 0
    cond2 = float(b3pg["corruption_rate"]) <= float(b2a40["corruption_rate"])
    # Count key CWE improvements vs B0
    b0_key = {cwe: int(cwe_by_method["B0"].get(cwe, {}).get("method_count", 0)) for cwe in KEY_CWES}
    pg_key = {cwe: int(cwe_by_method["B3_PG"].get(cwe, {}).get("method_count", 0)) for cwe in KEY_CWES}
    improved_cwes = [cwe for cwe in KEY_CWES if pg_key[cwe] < b0_key[cwe]]
    cond3 = len(improved_cwes) >= 2
    cond4 = float(b3pg["validity_rate"]) >= 0.95

    lines.append(f"  (1) VRR > 0:               {cond1} (corrected_vrr={float(b3pg['corrected_vrr']):.1%})")
    lines.append(f"  (2) corruption <= B2_a40:  {cond2} ({float(b3pg['corruption_rate']):.1%} vs {float(b2a40['corruption_rate']):.1%})")
    lines.append(f"  (3) >=2 key CWE improved:  {cond3} (improved={improved_cwes})")
    lines.append(f"  (4) validity >= 95%:        {cond4} ({float(b3pg['validity_rate']):.1%})")
    b3pg_eligible = cond1 and cond2 and cond3 and cond4
    lines.append(f"  B3-PG eligible:            {b3pg_eligible}")

    lines.append("\n--- B* SELECTION DECISION ---")
    if b3pg_eligible:
        bstar = "B3_PG"
        rationale = "B3-PG passes all four selection conditions."
    else:
        bstar = "B2_a40"
        failed = []
        if not cond1: failed.append("VRR <= 0")
        if not cond2: failed.append("corruption > B2_a40")
        if not cond3: failed.append(f"only {len(improved_cwes)}/2 key CWEs improved")
        if not cond4: failed.append("validity < 95%")
        rationale = (
            f"B3-PG fails conditions: {', '.join(failed)}. "
            f"B2-a40 selected as B* with corrected VRR={float(b2a40['corrected_vrr']):.1%}, "
            f"corruption={float(b2a40['corruption_rate']):.1%}, "
            f"validity={float(b2a40['validity_rate']):.1%}."
        )

    lines.append(f"  B* = {bstar}")
    lines.append(f"  Rationale: {rationale}")

    lines.append("\n--- DEVIATIONS DOCUMENTED ---")
    lines.append("1. Semgrep skipped on Windows (process hang on steered outputs).")
    lines.append("   Rescanned on Linux (Google Colab) with semgrep+regex.")
    lines.append("   B0 vulnerable count restored to 60, consistent with Phase 2.")
    lines.append("2. B3-ungated had 68 empty outputs in target CWEs (9 were B0-vulnerable).")
    lines.append("   Corrected VRR computed. True B3-ungated VRR = 40.0%.")
    lines.append("3. B3-PG shows negative VRR (-1.7%). Plausibility gate counterproductive.")
    lines.append("   Documented as mechanistic finding: KL gating incompatible with")
    lines.append("   SAE steering at token level for CWE-327 and CWE-89.")
    lines.append("4. B2-a20 and B2-a40 both achieve 55% raw VRR. B2-a40 selected")
    lines.append("   as superior due to zero empty B0-vulnerable outputs.")

    lines.append("\n--- CHECKPOINT STATUS ---")
    bstar_m = by_method[bstar]
    checkpoint_vrr     = float(bstar_m["corrected_vrr"]) > 0
    checkpoint_corrupt = float(bstar_m["corruption_rate"]) <= 0.05
    checkpoint_valid   = float(bstar_m["validity_rate"]) >= 0.95
    checkpoint_cwe     = any(
        int(cwe_by_method[bstar].get(cwe, {}).get("avoided", 0)) > 0
        for cwe in ["CWE-338","CWE-327","CWE-120","CWE-89"]
    )
    lines.append(f"  VRR > 0:               {checkpoint_vrr}")
    lines.append(f"  Corruption <= 5%:      {checkpoint_corrupt} ({float(bstar_m['corruption_rate']):.1%})")
    lines.append(f"  Validity >= 95%:       {checkpoint_valid} ({float(bstar_m['validity_rate']):.1%})")
    lines.append(f"  Key CWE improvement:   {checkpoint_cwe}")
    all_pass = checkpoint_vrr and checkpoint_corrupt and checkpoint_valid and checkpoint_cwe
    lines.append(f"  PHASE 5 CHECKPOINT:    {'PASS' if all_pass else 'FAIL'}")

    report = "\n".join(lines)
    out_path = P5_DIR / "phase5_comparison_report.txt"
    out_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved → {out_path}")

    # Save B* selection as JSON for Phase 6
    bstar_json = {
        "bstar": bstar,
        "rationale": rationale,
        "corrected_vrr": float(bstar_m["corrected_vrr"]),
        "corruption_rate": float(bstar_m["corruption_rate"]),
        "validity_rate": float(bstar_m["validity_rate"]),
        "alpha": 40.0,
        "layer": 19,
        "feature_map": {
            "CWE-120": {"layer": 19, "feature": 14193},
            "CWE-787": {"layer": 19, "feature": 1515},
            "CWE-190": {"layer": 19, "feature": 16897},
            "CWE-327": {"layer": 23, "feature": 14449},
            "CWE-89":  {"layer": 23, "feature": 1652},
            "CWE-338": {"layer": 23, "feature": 7533},
            "CWE-79":  {"layer": 16, "feature": 9816},
            "CWE-125": {"layer": 23, "feature": 16655},
            "CWE-476": {"layer": 23, "feature": 18397},
        }
    }
    bstar_path = PROJECT_ROOT / "configs/bstar_config.json"
    bstar_path.write_text(json.dumps(bstar_json, indent=2), encoding="utf-8")
    print(f"Saved B* config → {bstar_path}")


if __name__ == "__main__":
    main()