"""
Step 8.1 — Select final method and write final_method_selection.json.
Decision is already made: B_STAR_STATIC (B2_a40, alpha=40.0).
This script formalizes it with references to Phase 5 and Phase 7 evidence.
"""

import json
import os
from datetime import datetime

# --- Paths ---
PROJECT_ROOT = os.getcwd()
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "phase8")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PHASE5_RESULTS = os.path.join(PROJECT_ROOT, "outputs", "phase5", "phase5_main_results.csv")
PHASE7_REPORT  = os.path.join(PROJECT_ROOT, "outputs", "phase7", "checkpoint_validation_metrics.csv")
BSTAR_CONFIG   = os.path.join(PROJECT_ROOT, "configs", "bstar_config.json")
OUTPUT_FILE    = os.path.join(OUTPUT_DIR, "final_method_selection.json")

# --- Verify referenced files exist ---
references = {
    "phase5_main_results": PHASE5_RESULTS,
    "phase7_checkpoint_validation_metrics": PHASE7_REPORT,
    "bstar_config": BSTAR_CONFIG,
}
missing = [k for k, p in references.items() if not os.path.exists(p)]
if missing:
    raise FileNotFoundError(f"Missing required reference files: {missing}")

# --- Load bstar config for inline documentation ---
with open(BSTAR_CONFIG) as f:
    bstar = json.load(f)

# --- Build selection record ---
selection = {
    "final_method": "B_STAR_STATIC",
    "final_method_label": "B2_a40 — Single-feature static latent repair, alpha=40.0",
    "selection_timestamp": datetime.utcnow().isoformat() + "Z",

    "phase5_evidence": {
        "reference_file": PHASE5_RESULTS,
        "b2_a40_vrr": 0.550,
        "b2_a40_corruption_rate": 0.002,
        "b2_a40_output_validity": 0.981,
        "b2_a40_corrected_vrr": 0.550,
        "key_cwe_improvements": {
            "CWE-120": {"baseline": 22, "repaired": 8, "vrr": 0.636},
            "CWE-327": {"baseline": 29, "repaired": 19, "vrr": 0.345},
            "CWE-89":  {"baseline": 9,  "repaired": 0,  "vrr": 1.000},
        },
        "primitive_selected": "B2_a40",
        "selection_criteria_met": True,
    },

    "phase7_evidence": {
        "reference_file": PHASE7_REPORT,
        "note": "No online rollout performed; B4 failed at offline checkpoint stage.",
        "b4_evaluated": True,
        "b4_passed_checkpoint": False,
        "b4_failure_reasons": [
            "unsafe_action_rate=1.86% exceeds threshold of 1.0%",
            "reward sparsity: only 3.9% of train prompts were actionable",
            "best checkpoint (linear_epoch_01) RCR=0.816 but unsafe_rate above threshold",
            "no-op dominance=90.3% (structural, not policy collapse)",
        ],
        "b4_result": "FAILED — not eligible for final deployment",
    },

    "final_method_config": {
        "method_id": bstar.get("method_id", "B2_a40"),
        "alpha": bstar.get("alpha", 40.0),
        "intervention_layers": bstar.get("intervention_layers", [16, 19, 23]),
        "bstar_config_path": BSTAR_CONFIG,
    },

    "paper_framing": {
        "rq1": "Can SAE features causally encode CWE vulnerabilities? → Yes (3 active CWEs: CWE-120, CWE-327, CWE-89)",
        "rq2": "Does semantic grouping + multi-layer repair beat single-feature repair? → Yes (B* > B2-Thea)",
        "rq3": "Does bandit dynamic control improve over static repair? → No (reward sparsity structural finding)",
        "negative_result_scope": "B4 failure is a documented structural finding, not an implementation flaw",
    },

    "phase9_instruction": (
        "Use final_method=B_STAR_STATIC for all Phase 9 evaluations. "
        "B4 is reported as a comparison/negative result only."
    ),
}

# --- Write output ---
with open(OUTPUT_FILE, "w") as f:
    json.dump(selection, f, indent=2)

print(f"[Step 8.1] final_method_selection.json written to: {OUTPUT_FILE}")
print(f"  final_method : {selection['final_method']}")
print(f"  B4 passed    : {selection['phase7_evidence']['b4_passed_checkpoint']}")
print(f"  Timestamp    : {selection['selection_timestamp']}")