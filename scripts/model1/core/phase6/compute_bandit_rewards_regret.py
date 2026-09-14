"""
Phase 6 Step 6.4 - Compute Bandit Rewards and Regret Transformation
Base reward per (prompt, action):
  vuln->safe:    +1.0
  unchanged:      0.0
  safe->vuln:    -1.5
  invalid:       -1.0
  cost:          -0.1 * strength_level
  crypto bonus:  +0.25 for CWE-327/338 repair
Regret: r'(i,a) = r(i,a) - r*(i) where r*(i) = max_a r(i,a)
"""
import json
import os
import numpy as np
from collections import defaultdict

ROOT = "."
OUT_DIR = os.path.join(ROOT, "outputs/phase6")

CRYPTO_CWES = {"CWE-327", "CWE-338"}
CRYPTO_BONUS = 0.25
COST_PER_LEVEL = 0.05

print("Loading rollout results...")
results = json.load(open(os.path.join(OUT_DIR, "candidate_rollout_results.json")))
print(f"  {len(results)} (prompt, action) pairs")

def compute_base_reward(r):
    """Compute base reward for a single (prompt, action) record."""
    # Invalid output
    if not r["is_valid"] or r["is_vulnerable"] is None:
        return -1.0

    b0_vuln = r["b0_vulnerable"]
    is_vuln = r["is_vulnerable"]
    action_type = r["action_type"]
    strength_level = 0
    if action_type == "intervention":
        sl = r.get("strength_label", "Low")
        strength_level = {"Low": 1, "Med": 2, "High": 3}.get(sl, 1)

    reward = 0.0

    if b0_vuln and not is_vuln:
        reward += 1.0  # repair
        # Crypto bonus
        if r.get("prompt_cwe") in CRYPTO_CWES:
            reward += CRYPTO_BONUS
    elif not b0_vuln and is_vuln:
        reward -= 1.5  # corruption
    elif b0_vuln and is_vuln:
        reward += 0.0  # unchanged vulnerable
    else:
        reward += 0.0  # unchanged safe

    # Intervention cost
    reward -= COST_PER_LEVEL * strength_level

    return reward

# Group by prompt
prompt_records = defaultdict(list)
for r in results:
    prompt_records[r["prompt_id"]].append(r)

print(f"  Prompts: {len(prompt_records)}")

# Compute rewards
reward_records = []
for pid, records in prompt_records.items():
    base_rewards = []
    for r in records:
        br = compute_base_reward(r)
        base_rewards.append(br)

    r_star = max(base_rewards)

    for r, br in zip(records, base_rewards):
        regret_reward = br - r_star
        reward_records.append({
            "prompt_id": pid,
            "action_id": r["action_id"],
            "action_type": r["action_type"],
            "group_id": r.get("group_id"),
            "cwe_id": r.get("cwe_id"),
            "strength_label": r.get("strength_label", "none"),
            "alpha": r.get("alpha", 0.0),
            "prompt_cwe": r.get("prompt_cwe"),
            "b0_vulnerable": r["b0_vulnerable"],
            "is_vulnerable": r["is_vulnerable"],
            "is_valid": r["is_valid"],
            "base_reward": round(br, 4),
            "regret_reward": round(regret_reward, 4),
            "r_star": round(r_star, 4),
            "is_best_action": abs(br - r_star) < 1e-6
        })

print(f"  Total reward records: {len(reward_records)}")

# --- Reward distributions by action ---
action_stats = defaultdict(lambda: {"base_rewards": [], "regret_rewards": [], "n_best": 0})
for rr in reward_records:
    key = (rr["action_id"], rr["action_type"], rr.get("group_id"), rr.get("strength_label", "none"))
    action_stats[key]["base_rewards"].append(rr["base_reward"])
    action_stats[key]["regret_rewards"].append(rr["regret_reward"])
    if rr["is_best_action"]:
        action_stats[key]["n_best"] += 1

print("\nReward distribution by action:")
print(f"{'Action':30} {'N':>5} {'Base Mean':>10} {'Base Std':>9} {'Regret Mean':>12} {'N Best':>7}")
dist_rows = []
for key, stats in sorted(action_stats.items()):
    aid, atype, gid, sl = key
    label = "no-op" if atype == "no_op" else f"{gid}_{sl}"
    n = len(stats["base_rewards"])
    bm = np.mean(stats["base_rewards"])
    bs = np.std(stats["base_rewards"])
    rm = np.mean(stats["regret_rewards"])
    nb = stats["n_best"]
    print(f"  {label:28} {n:5d} {bm:10.4f} {bs:9.4f} {rm:12.4f} {nb:7d}")
    dist_rows.append({
        "action_id": aid, "label": label,
        "n": n, "base_mean": round(float(bm), 4), "base_std": round(float(bs), 4),
        "regret_mean": round(float(rm), 4), "n_best": nb
    })

# --- Best action dominance check ---
best_action_counts = defaultdict(int)
for rr in reward_records:
    if rr["is_best_action"]:
        best_action_counts[rr["action_id"]] += 1

total_prompts = len(prompt_records)
print(f"\nBest action dominance (total prompts: {total_prompts}):")
for aid, cnt in sorted(best_action_counts.items(), key=lambda x: -x[1]):
    label = next((r["label"] for r in dist_rows if r["action_id"] == aid), str(aid))
    pct = 100 * cnt / total_prompts
    flag = " *** DOMINANCE WARNING" if pct > 85 else ""
    print(f"  Action {aid:2d} ({label:28}): {cnt:4d} ({pct:.1f}%){flag}")

# --- Coverage check ---
prompts_with_nonnoop = set()
for rr in reward_records:
    if rr["action_type"] == "intervention":
        prompts_with_nonnoop.add(rr["prompt_id"])
coverage = len(prompts_with_nonnoop) / total_prompts
print(f"\nCoverage: {len(prompts_with_nonnoop)}/{total_prompts} prompts have >=1 non-no-op action ({100*coverage:.1f}%)")
if coverage < 0.90:
    print("  WARNING: Coverage < 90%")

# --- Sanity checks ---
noop_rewards = [rr["base_reward"] for rr in reward_records if rr["action_type"] == "no_op"]
bstar_mimic_rewards = [rr["base_reward"] for rr in reward_records
                        if rr.get("alpha", 0) == 40.0 and rr["action_type"] == "intervention"]
bstar_mimic_regret = [rr["regret_reward"] for rr in reward_records
                       if rr.get("alpha", 0) == 40.0 and rr["action_type"] == "intervention"]

print(f"\nSanity checks:")
print(f"  No-op mean base reward:      {np.mean(noop_rewards):.4f} (expected ~0)")
print(f"  B*-mimic mean base reward:   {np.mean(bstar_mimic_rewards):.4f} (expected > 0)")
print(f"  B*-mimic mean regret reward: {np.mean(bstar_mimic_regret):.4f} (expected ~0)")

# --- Reward sanity report ---
sanity = {
    "n_prompts": total_prompts,
    "n_pairs": len(reward_records),
    "noop_mean_base": round(float(np.mean(noop_rewards)), 4),
    "bstar_mimic_mean_base": round(float(np.mean(bstar_mimic_rewards)), 4),
    "bstar_mimic_mean_regret": round(float(np.mean(bstar_mimic_regret)), 4),
    "coverage_pct": round(100 * coverage, 2),
    "dominance_warning": any(v / total_prompts > 0.85 for v in best_action_counts.values()),
    "best_action_counts": {str(k): v for k, v in best_action_counts.items()}
}
sanity_path = os.path.join(OUT_DIR, "reward_sanity_report.json")
json.dump(sanity, open(sanity_path, "w"), indent=2)
print(f"\nSaved: {sanity_path}")

# --- Save distributions ---
import csv
dist_path = os.path.join(OUT_DIR, "reward_distribution_by_action.csv")
with open(dist_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["action_id", "label", "n", "base_mean", "base_std", "regret_mean", "n_best"])
    w.writeheader()
    w.writerows(dist_rows)
print(f"Saved: {dist_path}")

regret_path = os.path.join(OUT_DIR, "regret_distribution_by_action.csv")
regret_rows = []
for key, stats in sorted(action_stats.items()):
    aid, atype, gid, sl = key
    label = "no-op" if atype == "no_op" else f"{gid}_{sl}"
    regret_rows.append({
        "action_id": aid, "label": label,
        "regret_mean": round(float(np.mean(stats["regret_rewards"])), 4),
        "regret_std": round(float(np.std(stats["regret_rewards"])), 4),
        "regret_min": round(float(np.min(stats["regret_rewards"])), 4),
        "regret_max": round(float(np.max(stats["regret_rewards"])), 4)
    })
with open(regret_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["action_id", "label", "regret_mean", "regret_std", "regret_min", "regret_max"])
    w.writeheader()
    w.writerows(regret_rows)
print(f"Saved: {regret_path}")

# Save full reward records
rewards_path = os.path.join(OUT_DIR, "reward_records.json")
json.dump(reward_records, open(rewards_path, "w"), indent=2)
print(f"Saved: {rewards_path}")

print("\nDone.")