"""
Phase 5 Step 5.6 - Compute evaluation metrics for all methods
Outputs: outputs/phase5/phase5_main_results.csv
         outputs/phase5/phase5_per_cwe_results.csv
         outputs/phase5/phase5_corruption_results.csv
         outputs/phase5/phase5_output_validity.csv
         outputs/phase5/phase5_plausibility_effect.csv
"""

import json
import csv
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path.cwd()
P5_DIR       = PROJECT_ROOT / "outputs/phase5"
P2_DIR       = PROJECT_ROOT / "outputs/phase2"

DERIVATION_CWES = {
    "CWE-120","CWE-125","CWE-787","CWE-190",
    "CWE-476","CWE-89","CWE-79","CWE-327"
}
EXPLORATORY_CWES = {"CWE-338"}
TARGET_LANGUAGES = {"c","cpp","c++","python","java","javascript","js"}

# --- File map ---
# B0 uses semgrep ICD from Colab for consistency
METHODS = {
    "B0": {
        "outputs": P2_DIR / "baseline_dev_outputs.json",
        "icd":     P5_DIR / "baseline_dev_icd_semgrep.json",
    },
    "B1": {
        "outputs": P5_DIR / "zeroshot_dev_outputs.json",
        "icd":     P5_DIR / "zeroshot_dev_icd.json",
    },
    "B2_a20": {
        "outputs": P5_DIR / "thea_static_dev_outputs_a20.json",
        "icd":     P5_DIR / "thea_static_dev_icd_a20.json",
    },
    "B2_a40": {
        "outputs": P5_DIR / "thea_static_dev_outputs_a40.json",
        "icd":     P5_DIR / "thea_static_dev_icd_a40.json",
    },
    "B3_ungated": {
        "outputs": P5_DIR / "semantic_static_dev_outputs.json",
        "icd":     P5_DIR / "semantic_static_dev_icd.json",
    },
    "B3_PG": {
        "outputs": P5_DIR / "semantic_static_pg_dev_outputs.json",
        "icd":     P5_DIR / "semantic_static_pg_dev_icd.json",
    },
}


def load_icd(path):
    """Returns dict: prompt_id -> icd record."""
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    return {r["prompt_id"]: r for r in records}


def load_outputs(path):
    """Returns dict: prompt_id -> output record."""
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    return {r["prompt_id"]: r for r in records}


def is_valid_output(rec):
    """Non-empty generated code."""
    return bool(rec.get("generated_code", "").strip())


def compute_method_metrics(method_name, icd_map, out_map, b0_icd_map):
    """Compute overall metrics for one method."""
    total        = len(icd_map)
    n_vuln       = sum(1 for r in icd_map.values() if r.get("is_vulnerable"))
    n_skipped    = sum(1 for r in icd_map.values() if r.get("skipped"))
    n_scanned    = total - n_skipped
    n_valid      = sum(1 for pid, r in out_map.items() if is_valid_output(r))
    validity     = n_valid / total if total > 0 else 0

    # VRR vs B0
    b0_vuln = sum(1 for r in b0_icd_map.values() if r.get("is_vulnerable"))
    vrr = 1 - (n_vuln / b0_vuln) if b0_vuln > 0 else 0

    # Corruption: prompts safe under B0 that become vulnerable under method
    b0_safe_ids = {pid for pid, r in b0_icd_map.items() if not r.get("is_vulnerable")}
    corrupted   = sum(
        1 for pid in b0_safe_ids
        if pid in icd_map and icd_map[pid].get("is_vulnerable")
    )
    corruption_rate = corrupted / len(b0_safe_ids) if b0_safe_ids else 0

    # Safe-to-safe preservation
    safe_to_safe = 1 - corruption_rate

    # Empty outputs in target CWEs
    target_cwe_ids = {
        pid for pid, r in b0_icd_map.items()
        if r.get("cwe_id","") in DERIVATION_CWES | EXPLORATORY_CWES
    }
    empty_target = sum(
        1 for pid in target_cwe_ids
        if pid in out_map and not is_valid_output(out_map[pid])
    )
    # How many of those empty were vulnerable in B0
    empty_were_b0_vuln = sum(
        1 for pid in target_cwe_ids
        if pid in out_map and not is_valid_output(out_map[pid])
        and b0_icd_map.get(pid, {}).get("is_vulnerable")
    )

    # Corrected vuln count (add back empty outputs that were B0-vulnerable)
    corrected_vuln = n_vuln + empty_were_b0_vuln
    corrected_vrr  = 1 - (corrected_vuln / b0_vuln) if b0_vuln > 0 else 0

    return {
        "method":            method_name,
        "total_prompts":     total,
        "n_scanned":         n_scanned,
        "n_skipped":         n_skipped,
        "n_vulnerable":      n_vuln,
        "corrected_vuln":    corrected_vuln,
        "b0_vulnerable":     b0_vuln,
        "vrr":               round(vrr, 4),
        "corrected_vrr":     round(corrected_vrr, 4),
        "corrupted":         corrupted,
        "corruption_rate":   round(corruption_rate, 4),
        "safe_to_safe":      round(safe_to_safe, 4),
        "n_valid":           n_valid,
        "validity_rate":     round(validity, 4),
        "empty_target_cwe":  empty_target,
        "empty_were_b0_vuln":empty_were_b0_vuln,
    }


def compute_per_cwe(method_name, icd_map, b0_icd_map):
    """Per-CWE vulnerability counts and VRR."""
    all_cwes = DERIVATION_CWES | EXPLORATORY_CWES

    rows = []
    for cwe in sorted(all_cwes):
        # B0 count for this CWE
        b0_count = sum(
            1 for r in b0_icd_map.values()
            if r.get("is_vulnerable") and cwe in r.get("vulnerable_cwes", [])
        )
        # Method count
        m_count = sum(
            1 for r in icd_map.values()
            if r.get("is_vulnerable") and cwe in r.get("vulnerable_cwes", [])
        )
        avoided = b0_count - m_count
        cwe_vrr = round(1 - (m_count / b0_count), 4) if b0_count > 0 else None
        cwe_type = "exploratory" if cwe in EXPLORATORY_CWES else "derivation"

        rows.append({
            "method":      method_name,
            "cwe_id":      cwe,
            "cwe_type":    cwe_type,
            "b0_count":    b0_count,
            "method_count":m_count,
            "avoided":     avoided,
            "cwe_vrr":     cwe_vrr,
        })
    return rows


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved → {path}")


def main():
    # Load B0 as reference
    b0_icd_map = load_icd(METHODS["B0"]["icd"])
    b0_out_map = load_outputs(METHODS["B0"]["outputs"])

    main_results  = []
    per_cwe_rows  = []
    corruption_rows = []
    validity_rows = []

    for method_name, paths in METHODS.items():
        print(f"Computing {method_name}...")
        icd_map = load_icd(paths["icd"])
        out_map = load_outputs(paths["outputs"])

        metrics = compute_method_metrics(
            method_name, icd_map, out_map, b0_icd_map
        )
        main_results.append(metrics)

        cwe_rows = compute_per_cwe(method_name, icd_map, b0_icd_map)
        per_cwe_rows.extend(cwe_rows)

        corruption_rows.append({
            "method":          method_name,
            "b0_safe_prompts": len([r for r in b0_icd_map.values() if not r.get("is_vulnerable")]),
            "corrupted":       metrics["corrupted"],
            "corruption_rate": metrics["corruption_rate"],
            "safe_to_safe":    metrics["safe_to_safe"],
        })

        validity_rows.append({
            "method":       method_name,
            "total":        metrics["total_prompts"],
            "n_valid":      metrics["n_valid"],
            "validity_rate":metrics["validity_rate"],
            "empty_target": metrics["empty_target_cwe"],
            "empty_b0_vuln":metrics["empty_were_b0_vuln"],
        })

    # Write outputs
    write_csv(
        P5_DIR / "phase5_main_results.csv",
        main_results,
        ["method","total_prompts","n_scanned","n_skipped","n_vulnerable",
         "corrected_vuln","b0_vulnerable","vrr","corrected_vrr",
         "corrupted","corruption_rate","safe_to_safe","n_valid","validity_rate",
         "empty_target_cwe","empty_were_b0_vuln"]
    )

    write_csv(
        P5_DIR / "phase5_per_cwe_results.csv",
        per_cwe_rows,
        ["method","cwe_id","cwe_type","b0_count","method_count","avoided","cwe_vrr"]
    )

    write_csv(
        P5_DIR / "phase5_corruption_results.csv",
        corruption_rows,
        ["method","b0_safe_prompts","corrupted","corruption_rate","safe_to_safe"]
    )

    write_csv(
        P5_DIR / "phase5_output_validity.csv",
        validity_rows,
        ["method","total","n_valid","validity_rate","empty_target","empty_b0_vuln"]
    )

    # Plausibility effect (B3-ungated vs B3-PG)
    b3_metrics   = next(r for r in main_results if r["method"] == "B3_ungated")
    b3pg_metrics = next(r for r in main_results if r["method"] == "B3_PG")
    plausibility_log = json.loads(
        (P5_DIR / "semantic_static_pg_dev_plausibility_log.json").read_text(encoding="utf-8")
    )
    steered = [r for r in plausibility_log if r["n_steps"] > 0]
    total_steps  = sum(r["n_steps"] for r in steered)
    total_gated  = sum(r["n_gated"] for r in steered)
    gate_rate    = round(total_gated / total_steps, 4) if total_steps > 0 else 0

    plausibility_rows = [{
        "metric": "B3_ungated_vrr",        "value": b3_metrics["vrr"]},
        {"metric": "B3_PG_vrr",            "value": b3pg_metrics["vrr"]},
        {"metric": "B3_ungated_corrected_vrr", "value": b3_metrics["corrected_vrr"]},
        {"metric": "B3_PG_corrected_vrr",  "value": b3pg_metrics["corrected_vrr"]},
        {"metric": "B3_ungated_corruption","value": b3_metrics["corruption_rate"]},
        {"metric": "B3_PG_corruption",     "value": b3pg_metrics["corruption_rate"]},
        {"metric": "gate_activation_rate", "value": gate_rate},
        {"metric": "total_steered_prompts","value": len(steered)},
        {"metric": "total_steps",          "value": total_steps},
        {"metric": "total_gated_steps",    "value": total_gated},
    ]
    write_csv(
        P5_DIR / "phase5_plausibility_effect.csv",
        plausibility_rows,
        ["metric", "value"]
    )

    # Print summary
    print("\n=== PHASE 5 SUMMARY ===")
    print(f"{'Method':<15} {'Vuln':>6} {'Corr':>6} {'VRR':>8} {'CorrVRR':>9} {'Corrupt':>8} {'Validity':>9}")
    print("-" * 70)
    for r in main_results:
        print(f"{r['method']:<15} {r['n_vulnerable']:>6} {r['corrected_vuln']:>6} "
              f"{r['vrr']:>8.1%} {r['corrected_vrr']:>9.1%} "
              f"{r['corruption_rate']:>8.1%} {r['validity_rate']:>9.1%}")


if __name__ == "__main__":
    main()