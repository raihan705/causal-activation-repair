"""
Phase 7 — Step 7.2: Train linear and MLP bandit policies on offline_bandit_train.jsonl.

Inputs:
  outputs/phase7/offline_bandit_train.jsonl
  outputs/phase6/bandit_action_map.json

Outputs:
  outputs/phase7/checkpoints/linear_bandit_epoch_{k}.ckpt
  outputs/phase7/checkpoints/mlp_bandit_epoch_{k}.ckpt
  outputs/phase7/bandit_train_log.csv
"""

import json
import os
import csv
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ── Config ───────────────────────────────────────────────────────────────────
TRAIN_JSONL    = "outputs/phase7/offline_bandit_train.jsonl"
ACTION_MAP     = "outputs/phase6/bandit_action_map.json"
OUT_DIR        = "outputs/phase7"
CKPT_DIR       = os.path.join(OUT_DIR, "checkpoints")
TRAIN_LOG      = os.path.join(OUT_DIR, "bandit_train_log.csv")

STATE_DIM      = 102
EPOCHS         = 30
BATCH_SIZE     = 64
LR             = 1e-3
WEIGHT_DECAY   = 1e-4
PATIENCE       = 5          # early stopping
HIDDEN_SIZE    = 64
DROPOUT        = 0.1
SEED           = 42
HARD_CWE_BONUS = 0.25       # extra weight for CWE-327, CWE-89

HARD_CWES      = {"CWE-327", "CWE-89"}

os.makedirs(CKPT_DIR, exist_ok=True)
torch.manual_seed(SEED)

# ── 1. Load action map → number of actions ──────────────────────────────────
with open(ACTION_MAP) as f:
    action_map = json.load(f)

# action_map is a list or dict; derive N_ACTIONS
if isinstance(action_map, list):
    N_ACTIONS = len(action_map)
elif isinstance(action_map, dict):
    actions = action_map.get("actions", action_map)
    N_ACTIONS = len(actions) if isinstance(actions, list) else max(int(k) for k in actions.keys()) + 1
print(f"N_ACTIONS: {N_ACTIONS}")

# ── 2. Load and group training records by prompt ─────────────────────────────
prompt_records = defaultdict(list)
with open(TRAIN_JSONL) as f:
    for line in f:
        rec = json.loads(line.strip())
        prompt_records[rec["prompt_id"]].append(rec)

print(f"Train prompts: {len(prompt_records)}")

# ── 3. Build per-prompt training examples ────────────────────────────────────
#  - label     = action_id where is_best_action == True
#  - weight    = 1 + max(0, r_star - r_bar_i) + hard_cwe_bonus
#  - state     = from any record for that prompt (all identical)

examples = []   # (state_tensor, label, weight)

for pid, recs in prompt_records.items():
    # state (same for all records of this prompt)
    state = torch.tensor(recs[0]["state"], dtype=torch.float32)

    # best action label
    best_recs = [r for r in recs if r.get("is_best_action")]
    if not best_recs:
        # fallback: pick highest base_reward
        best_recs = [max(recs, key=lambda r: r["base_reward"])]
    label = best_recs[0]["action_id"]

    # r_star and r_bar
    r_star  = best_recs[0]["base_reward"]
    r_bar   = sum(r["base_reward"] for r in recs) / len(recs)
    weight  = 1.0 + max(0.0, r_star - r_bar)

    # hard CWE bonus
    prompt_cwe = recs[0].get("prompt_cwe", "")
    if prompt_cwe in HARD_CWES:
        weight += HARD_CWE_BONUS

    examples.append((state, label, weight))

print(f"Training examples (one per prompt): {len(examples)}")

# ── 4. Dataset ───────────────────────────────────────────────────────────────
class BanditDataset(Dataset):
    def __init__(self, examples):
        self.states  = torch.stack([e[0] for e in examples])
        self.labels  = torch.tensor([e[1] for e in examples], dtype=torch.long)
        self.weights = torch.tensor([e[2] for e in examples], dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.states[idx], self.labels[idx], self.weights[idx]

dataset    = BanditDataset(examples)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED))

# ── 5. Model definitions ─────────────────────────────────────────────────────
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
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)

# ── 6. Training loop ─────────────────────────────────────────────────────────
def compute_rcr(model, dataset, prompt_records_map):
    """
    Reward Capture Ratio over actionable prompts only
    (prompts where r_star > 0, i.e. at least one intervention helps).
    Prompts where the predicted action was not rolled out are skipped.
    RCR = mean(chosen_reward) / mean(r_star).
    """
    model.eval()
    with torch.no_grad():
        logits = model(dataset.states)
        preds  = logits.argmax(dim=1).tolist()

    prompt_ids = list(prompt_records_map.keys())
    chosen_rewards, best_rewards = [], []

    for i, pid in enumerate(prompt_ids):
        recs   = prompt_records_map[pid]
        r_star = max(r["base_reward"] for r in recs)

        if r_star <= 0:
            continue  # skip non-actionable prompts

        pred_action = preds[i]
        pred_rec    = next((r for r in recs if r["action_id"] == pred_action), None)

        if pred_rec is None:
            continue  # predicted action not in rollout; skip

        chosen_rewards.append(pred_rec["base_reward"])
        best_rewards.append(r_star)

    if not best_rewards:
        return 0.0

    mean_best   = sum(best_rewards)   / len(best_rewards)
    mean_chosen = sum(chosen_rewards) / len(chosen_rewards)
    denom = mean_best if mean_best != 0 else 1.0
    return mean_chosen / denom

def train_policy(model, name, dataloader, dataset, prompt_records_map):
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(reduction="none")

    log_rows = []
    best_score = -1e9
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        correct_top1 = 0
        total = 0

        for states, labels, weights in dataloader:
            optimizer.zero_grad()
            logits = model(states)
            loss   = (criterion(logits, labels) * weights).mean()
            loss.backward()
            optimizer.step()

            total_loss   += loss.item() * len(labels)
            correct_top1 += (logits.argmax(1) == labels).sum().item()
            total        += len(labels)

        avg_loss = total_loss / total
        top1_acc = correct_top1 / total
        rcr      = compute_rcr(model, dataset, prompt_records_map)

        log_rows.append({
            "policy": name,
            "epoch": epoch,
            "loss": round(avg_loss, 4),
            "top1_acc": round(top1_acc, 4),
            "rcr": round(rcr, 4),
        })

        # Save checkpoint
        ckpt_path = os.path.join(CKPT_DIR, f"{name}_epoch_{epoch:02d}.ckpt")
        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "rcr": rcr, "top1_acc": top1_acc, "loss": avg_loss}, ckpt_path)

        print(f"  [{name}] epoch {epoch:2d} | loss={avg_loss:.4f} | top1={top1_acc:.4f} | RCR={rcr:.4f}")

        # Early stopping on RCR
        if rcr > best_score:
            best_score   = rcr
            best_epoch   = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  [{name}] Early stop at epoch {epoch} (best epoch={best_epoch}, RCR={best_score:.4f})")
                break

    return log_rows

# ── 7. Train both policies ───────────────────────────────────────────────────
linear_model = LinearPolicy(STATE_DIM, N_ACTIONS)
mlp_model    = MLPPolicy(STATE_DIM, N_ACTIONS, HIDDEN_SIZE, DROPOUT)

print(f"\n=== Training Linear Policy ===")
log_linear = train_policy(linear_model, "linear_bandit", dataloader, dataset, prompt_records)

print(f"\n=== Training MLP Policy ===")
log_mlp = train_policy(mlp_model, "mlp_bandit", dataloader, dataset, prompt_records)

# ── 8. Save training log ─────────────────────────────────────────────────────
all_logs = log_linear + log_mlp
fieldnames = ["policy", "epoch", "loss", "top1_acc", "rcr"]

with open(TRAIN_LOG, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_logs)

print(f"\nTraining log saved to {TRAIN_LOG}")
print("Step 7.2 COMPLETE.")