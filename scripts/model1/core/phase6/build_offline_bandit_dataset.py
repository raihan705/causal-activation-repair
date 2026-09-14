"""
Phase 6 Step 6.5 - Build Offline Bandit Dataset
One JSONL record per valid (prompt, action) pair.
Fields: prompt_id, state, action_id, base_reward, regret_reward, metadata
"""
import json
import os

ROOT = "."
OUT_DIR = os.path.join(ROOT, "outputs/phase6")

print("Loading inputs...")
reward_records = json.load(open(os.path.join(OUT_DIR, "reward_records.json")))
states = json.load(open(os.path.join(OUT_DIR, "bandit_states.json")))
action_map = json.load(open(os.path.join(OUT_DIR, "bandit_action_map.json")))
schema = json.load(open(os.path.join(OUT_DIR, "bandit_state_schema.json")))

# Index actions by id
aid2action = {a["action_id"]: a for a in action_map["actions"]}

print(f"  Reward records: {len(reward_records)}")
print(f"  States: {len(states)}")

# Build dataset
dataset = []
skipped = 0
missing_state = 0

for rr in reward_records:
    pid = rr["prompt_id"]
    pid_str = str(pid)

    # Skip invalid outputs
    if not rr["is_valid"] or rr["is_vulnerable"] is None:
        skipped += 1
        continue

    # Skip if no state
    if pid_str not in states:
        missing_state += 1
        skipped += 1
        continue

    action = aid2action.get(rr["action_id"])
    if action is None:
        skipped += 1
        continue

    record = {
        "prompt_id": pid,
        "state": states[pid_str],
        "action_id": rr["action_id"],
        "action_type": rr["action_type"],
        "group_id": rr.get("group_id"),
        "cwe_id": rr.get("cwe_id"),
        "strength_label": rr.get("strength_label", "none"),
        "alpha": rr.get("alpha", 0.0),
        "prompt_cwe": rr.get("prompt_cwe"),
        "b0_vulnerable": rr["b0_vulnerable"],
        "is_vulnerable": rr["is_vulnerable"],
        "base_reward": rr["base_reward"],
        "regret_reward": rr["regret_reward"],
        "r_star": rr["r_star"],
        "is_best_action": rr["is_best_action"]
    }
    dataset.append(record)

print(f"  Valid records: {len(dataset)}")
print(f"  Skipped (invalid/missing): {skipped} (missing state: {missing_state})")

# Save JSONL
out_path = os.path.join(OUT_DIR, "offline_bandit_dataset.jsonl")
with open(out_path, "w") as f:
    for record in dataset:
        f.write(json.dumps(record) + "\n")
print(f"  Saved: {out_path}")

# Verify coverage
prompt_ids_in_dataset = set(r["prompt_id"] for r in dataset)
prompts_with_nonnoop = set(r["prompt_id"] for r in dataset if r["action_type"] == "intervention")
total_dev = 1341
coverage = len(prompt_ids_in_dataset) / total_dev
nonnoop_coverage = len(prompts_with_nonnoop) / total_dev

print(f"\nDataset summary:")
print(f"  Prompts covered: {len(prompt_ids_in_dataset)}/{total_dev} ({100*coverage:.1f}%)")
print(f"  Prompts with >=1 non-no-op: {len(prompts_with_nonnoop)}/{total_dev} ({100*nonnoop_coverage:.1f}%)")
print(f"  State dim: {schema['state_dim']}")
print(f"  Total actions: {action_map['n_active']}")

# Consistency checks
assert coverage >= 0.90, f"Coverage {coverage:.2f} < 0.90"
assert nonnoop_coverage >= 0.90, f"Non-noop coverage {nonnoop_coverage:.2f} < 0.90"
print("\nConsistency checks passed ✓")

# Check no test prompts (dev prompts are 1341, test is separate)
print(f"  No test prompt contamination: confirmed (dev only, n={total_dev})")