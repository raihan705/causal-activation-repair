"""
Phase 7 — Step 7.3: Offline checkpoint validation on offline_bandit_val.jsonl.

Inputs:
  outputs/phase7/offline_bandit_val.jsonl
  outputs/phase7/checkpoints/
  outputs/phase6/bandit_action_map.json

Outputs:
  outputs/phase7/checkpoint_validation_metrics.csv
"""

import json
import os
import csv
from collections import defaultdict

import torch
import torch.nn as nn

# ── Config ───────────────────────────────────────────────────────────────────
VAL_JSONL   = "outputs/phase7/offline_bandit_val.jsonl"
ACTION_MAP  = "outputs/phase6/bandit_action_map.json"
CKPT_DIR    = "outputs/phase7/checkpoints"
OUT_CSV     = "outputs/phase7/checkpoint_validation_metrics.csv"

STATE_DIM   = 102

# ── Load action map ───────────────────────────────────────────────────────────
with open(ACTION_MAP) as f:
    am = json.load(f)
actions   = am["actions"]
N_ACTIONS = len(actions)

# Masked action IDs
masked_ids = {a["action_id"] for a in actions if a.get("masked", False)}
high_ids   = {a["action_id"] for a in actions if a.get("strength_label") == "High"}

print(f"N_ACTIONS={N_ACTIONS} | masked={len(masked_ids)} | high={len(high_ids)}")

# ── Load val records grouped by prompt ───────────────────────────────────────
prompt_records = defaultdict(list)
with open(VAL_JSONL) as f:
    for line in f:
        rec = json.loads(line.strip())
        prompt_records[rec["prompt_id"]].append(rec)

print(f"Val prompts: {len(prompt_records)}")

# Actionable prompts (r_star > 0)
actionable_ids = {pid for pid, recs in prompt_records.items()
                  if max(r["base_reward"] for r in recs) > 0}
print(f"Actionable val prompts (r_star>0): {len(actionable_ids)}")

# Build ordered prompt list and state tensor
prompt_ids = list(prompt_records.keys())
states     = torch.tensor(
    [prompt_records[pid][0]["state"] for pid in prompt_ids],
    dtype=torch.float32
)

# ── Model definitions (must match training) ───────────────────────────────────
class LinearPolicy(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.fc = nn.Linear(state_dim, n_actions)
    def forward(self, x):
        return self.fc(x)

class MLPPolicy(nn.Module):
    def __init__(self, state_dim, n_actions, hidden=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),   nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_actions),
        )
    def forward(self, x):
        return self.net(x)

# ── Evaluation function ───────────────────────────────────────────────────────
def evaluate_checkpoint(model, prompt_ids, prompt_records, states,
                         actionable_ids, masked_ids, high_ids):
    model.eval()
    with torch.no_grad():
        preds = model(states).argmax(dim=1).tolist()

    noop_count     = 0
    unsafe_count   = 0
    high_count     = 0
    intervention_count = 0
    top1_correct   = 0

    chosen_rewards = []
    best_rewards   = []

    for i, pid in enumerate(prompt_ids):
        recs        = prompt_records[pid]
        pred_action = preds[i]

        # top-1 accuracy vs is_best_action label
        best_recs = [r for r in recs if r.get("is_best_action")]
        if not best_recs:
            best_recs = [max(recs, key=lambda r: r["base_reward"])]
        label = best_recs[0]["action_id"]
        if pred_action == label:
            top1_correct += 1

        # action profile
        if pred_action == 0:
            noop_count += 1
        else:
            intervention_count += 1
            if pred_action in masked_ids:
                unsafe_count += 1
            if pred_action in high_ids:
                high_count += 1

        # RCR — actionable prompts only
        if pid not in actionable_ids:
            continue
        r_star   = max(r["base_reward"] for r in recs)
        pred_rec = next((r for r in recs if r["action_id"] == pred_action), None)
        if pred_rec is None:
            continue
        chosen_rewards.append(pred_rec["base_reward"])
        best_rewards.append(r_star)

    n = len(prompt_ids)
    top1_acc    = top1_correct / n
    noop_rate   = noop_count / n
    unsafe_rate = unsafe_count / n
    high_rate   = (high_count / intervention_count) if intervention_count > 0 else 0.0

    if best_rewards:
        mean_best   = sum(best_rewards)   / len(best_rewards)
        mean_chosen = sum(chosen_rewards) / len(chosen_rewards)
        rcr = mean_chosen / mean_best if mean_best != 0 else 0.0
    else:
        rcr = 0.0

    return {
        "top1_acc":    round(top1_acc,    4),
        "rcr":         round(rcr,         4),
        "noop_rate":   round(noop_rate,   4),
        "unsafe_rate": round(unsafe_rate, 4),
        "high_rate":   round(high_rate,   4),
        "n_actionable_evaluated": len(best_rewards),
    }

# ── Evaluate all checkpoints ──────────────────────────────────────────────────
ckpt_files = sorted(os.listdir(CKPT_DIR))
rows = []

print(f"\n{'Checkpoint':<45} {'RCR':>6} {'top1':>6} {'noop':>6} {'unsafe':>7} {'high':>6} {'n_act':>6}")
print("-" * 90)

for fname in ckpt_files:
    if not fname.endswith(".ckpt"):
        continue
    path = os.path.join(CKPT_DIR, fname)
    ckpt = torch.load(path, map_location="cpu")

    # Infer model type from filename
    if fname.startswith("linear"):
        model = LinearPolicy(STATE_DIM, N_ACTIONS)
    else:
        model = MLPPolicy(STATE_DIM, N_ACTIONS)

    model.load_state_dict(ckpt["model_state"])

    metrics = evaluate_checkpoint(
        model, prompt_ids, prompt_records, states,
        actionable_ids, masked_ids, high_ids
    )

    epoch  = ckpt.get("epoch", "?")
    policy = "linear" if fname.startswith("linear") else "mlp"

    row = {"checkpoint": fname, "policy": policy, "epoch": epoch, **metrics}
    rows.append(row)

    print(f"  {fname:<43} {metrics['rcr']:>6.4f} {metrics['top1_acc']:>6.4f} "
          f"{metrics['noop_rate']:>6.4f} {metrics['unsafe_rate']:>7.4f} "
          f"{metrics['high_rate']:>6.4f} {metrics['n_actionable_evaluated']:>6}")

# ── Save CSV ──────────────────────────────────────────────────────────────────
fieldnames = ["checkpoint", "policy", "epoch", "rcr", "top1_acc",
              "noop_rate", "unsafe_rate", "high_rate", "n_actionable_evaluated"]

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved: {OUT_CSV}")

# ── Summary ───────────────────────────────────────────────────────────────────
passing = [r for r in rows if r["rcr"] >= 0.70 and r["unsafe_rate"] <= 0.01
           and r["noop_rate"] <= 0.90]

print(f"\nCheckpoints passing RCR>=0.70, unsafe<=1%, noop<=90%: {len(passing)}")
for r in passing:
    print(f"  {r['checkpoint']}  RCR={r['rcr']}  noop={r['noop_rate']}  unsafe={r['unsafe_rate']}")

if not passing:
    print("\nNo checkpoint passes all offline thresholds.")
    print("Phase 7 offline checkpoint condition FAILED — B* remains final method.")
else:
    print("\nPassing checkpoints found — proceed to Step 7.4 selection.")