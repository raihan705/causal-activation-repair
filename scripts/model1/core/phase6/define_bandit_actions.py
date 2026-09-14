"""
Phase 6 Step 6.2 - Define Bandit Action Space
Action 0: no-op
Actions 1-18: 9 groups x 2 strengths (Low=0.5x, Med=1.0x)
High masked: B* alpha=40.0, High=60.0 > max validated high_alpha=30.0
Total: 19 actions <= 25 ✓
"""
import json
import os

ROOT = "."
OUT_DIR = os.path.join(ROOT, "outputs/phase6")
os.makedirs(OUT_DIR, exist_ok=True)

BSTAR_ALPHA = 40.0
ACTIVE_CWES = ["CWE-120", "CWE-327", "CWE-89"]

lib = json.load(open(os.path.join(ROOT, "configs/intervention_library.json")))
all_groups = lib["groups"]
selected_groups = [g for g in all_groups if g["cwe_id"] in ACTIVE_CWES]
selected_groups = sorted(selected_groups, key=lambda g: (g["cwe_id"], g["group_id"]))

actions = []

# Action 0: no-op
actions.append({
    "action_id": 0,
    "type": "no_op",
    "group_id": None,
    "cwe_id": None,
    "layer_id": None,
    "feature_ids": None,
    "strength_label": "none",
    "alpha": 0.0,
    "strength_level": 0,
    "masked": False,
    "mask_reason": None
})

action_id = 1
for group in selected_groups:
    group_id = group["group_id"]
    cwe_id = group["cwe_id"]
    layer_id = group["layer_id"]
    feature_ids = group["feature_ids"]
    low_alpha = group["low_alpha"]    # Phase 4 validated
    base_alpha = group["base_alpha"]  # Phase 4 base

    # High alpha from B* scaling
    high_alpha_bstar = BSTAR_ALPHA * 1.5  # 60.0
    max_validated = group["high_alpha"]   # Phase 4 max (<=30.0)

    for strength_label, alpha in [("Low", BSTAR_ALPHA * 0.5), ("Med", BSTAR_ALPHA * 1.0)]:
        # Med=40.0 matches B* validated in Phase 5 — always safe
        # Low=20.0 is below B* — always safe
        masked = False
        mask_reason = None
        strength_level = 1 if strength_label == "Low" else 2

        actions.append({
            "action_id": action_id,
            "type": "intervention",
            "group_id": group_id,
            "cwe_id": cwe_id,
            "layer_id": layer_id,
            "feature_ids": feature_ids,
            "strength_label": strength_label,
            "alpha": alpha,
            "strength_level": strength_level,
            "masked": masked,
            "mask_reason": mask_reason
        })
        action_id += 1

    # High: always masked (exceeds validated range)
    actions.append({
        "action_id": action_id,
        "type": "intervention",
        "group_id": group_id,
        "cwe_id": cwe_id,
        "layer_id": layer_id,
        "feature_ids": feature_ids,
        "strength_label": "High",
        "alpha": high_alpha_bstar,
        "strength_level": 3,
        "masked": True,
        "mask_reason": f"alpha={high_alpha_bstar:.1f} > validated high_alpha={max_validated:.1f}"
    })
    action_id += 1

total = len(actions)
active = [a for a in actions if not a["masked"]]
masked = [a for a in actions if a["masked"]]

print(f"Total actions defined: {total}")
print(f"Active (unmasked): {len(active)}")
print(f"Masked: {len(masked)}")
print(f"\nActive action breakdown:")
for a in active:
    if a["type"] == "no_op":
        print(f"  {a['action_id']:2d}: no-op")
    else:
        print(f"  {a['action_id']:2d}: {a['group_id']:<22} {a['strength_label']:<4} alpha={a['alpha']:.1f}")

print(f"\nMasked actions:")
for a in masked:
    print(f"  {a['action_id']:2d}: {a['group_id']:<22} {a['strength_label']:<4} ({a['mask_reason']})")

assert len(active) <= 25, f"Active actions {len(active)} > 25"
assert active[0]["type"] == "no_op"

# Save full action map
action_map_path = os.path.join(OUT_DIR, "bandit_action_map.json")
json.dump({
    "total_actions": total,
    "n_active": len(active),
    "n_masked": len(masked),
    "bstar_alpha": BSTAR_ALPHA,
    "active_cwes": ACTIVE_CWES,
    "actions": actions
}, open(action_map_path, "w"), indent=2)
print(f"\nSaved: {action_map_path}")
print(f"Active actions: {len(active)} <= 25 ✓")