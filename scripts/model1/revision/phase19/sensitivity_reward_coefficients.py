#!/usr/bin/env python3
"""Run the approved OFAT reward sensitivity from saved Phase 6 rollouts."""

from __future__ import annotations

import collections
import json
import statistics

from phase19_common import OUT, ROOT, atomic_csv, atomic_json, hash_record, read_json, require


FINAL = {"corruption_penalty": 1.5, "intervention_cost": 0.05, "crypto_bonus": 0.25}
SETTINGS = [
    ("corruption_penalty", 1.0), ("corruption_penalty", 1.5), ("corruption_penalty", 2.0),
    ("intervention_cost", 0.0), ("intervention_cost", 0.05), ("intervention_cost", 0.1),
    ("crypto_bonus", 0.0), ("crypto_bonus", 0.25), ("crypto_bonus", 0.5),
]
CRYPTO_CWES = {"CWE-327", "CWE-338"}


def coefficients(factor, value):
    result = dict(FINAL)
    result[factor] = value
    return result


def base_reward(row, values):
    if not row["is_valid"] or row["is_vulnerable"] is None:
        return -1.0
    strength = 0
    if row["action_type"] == "intervention":
        strength = {"Low": 1, "Med": 2, "High": 3}.get(row.get("strength_label", "Low"), 1)
    reward = 0.0
    if row["b0_vulnerable"] and not row["is_vulnerable"]:
        reward += 1.0
        if row.get("prompt_cwe") in CRYPTO_CWES:
            reward += values["crypto_bonus"]
    elif not row["b0_vulnerable"] and row["is_vulnerable"]:
        reward -= values["corruption_penalty"]
    reward -= values["intervention_cost"] * strength
    return round(reward, 4)


def evaluate(rollouts, values, labels):
    grouped = collections.defaultdict(list)
    for row in rollouts:
        item = dict(row)
        item["base_reward"] = base_reward(row, values)
        grouped[int(row["prompt_id"])].append(item)
    prompt_results = {}
    reward_records = []
    for prompt_id in sorted(grouped):
        rows = grouped[prompt_id]
        r_star = max(row["base_reward"] for row in rows)
        best = sorted(int(row["action_id"]) for row in rows if abs(row["base_reward"] - r_star) < 1e-9)
        representative = best[0]
        prompt_results[prompt_id] = {
            "r_star": r_star,
            "best_action_ids": best,
            "representative_best_action_id": representative,
            "representative_best_action_label": labels[representative],
            "actionable": r_star > 0,
            "noop_best": 0 in best,
            "nonnoop_best": any(action != 0 for action in best),
        }
        for row in rows:
            reward_records.append({**row, "regret_reward": round(row["base_reward"] - r_star, 4), "is_best_action": int(row["action_id"]) in best})
    return prompt_results, reward_records


def main() -> None:
    rollout_path = ROOT / "outputs/phase6/candidate_rollout_results.json"
    reward_path = ROOT / "outputs/phase6/reward_records.json"
    action_path = ROOT / "outputs/phase6/bandit_action_map.json"
    script_path = ROOT / "phases/phase6/compute_bandit_rewards_regret.py"
    rollouts = read_json(rollout_path)
    historical = read_json(reward_path)
    action_map = read_json(action_path)
    require(len(rollouts) == len(historical) == 5364, "saved Phase 6 rollout/reward count changed")
    labels = {int(action["action_id"]): f'{action["type"]}:{action.get("group_id") or "none"}:{action.get("strength_label", "none")}' for action in action_map["actions"]}

    final_prompts, final_records = evaluate(rollouts, FINAL, labels)
    historical_index = {(int(row["prompt_id"]), int(row["action_id"])): row for row in historical}
    require(len(historical_index) == len(historical), "historical reward keys are not unique")
    for row in final_records:
        saved = historical_index[(int(row["prompt_id"]), int(row["action_id"]))]
        require(float(saved["base_reward"]) == row["base_reward"], "final base reward does not reproduce")
        require(float(saved["regret_reward"]) == row["regret_reward"], "final regret reward does not reproduce")
        require(bool(saved["is_best_action"]) == row["is_best_action"], "final best-action flag does not reproduce")

    coefficient_rows = []
    actionability_rows = []
    change_rows = []
    summary_by_setting = {}
    for factor, value in SETTINGS:
        setting_id = f"{factor}={value:g}"
        values = coefficients(factor, value)
        prompts, records = evaluate(rollouts, values, labels)
        changed = [prompt_id for prompt_id in sorted(prompts) if prompts[prompt_id]["best_action_ids"] != final_prompts[prompt_id]["best_action_ids"]]
        representative_changed = [prompt_id for prompt_id in sorted(prompts) if prompts[prompt_id]["representative_best_action_id"] != final_prompts[prompt_id]["representative_best_action_id"]]
        actionable = sum(row["actionable"] for row in prompts.values())
        noop_best = sum(row["noop_best"] for row in prompts.values())
        nonnoop_best = sum(row["nonnoop_best"] for row in prompts.values())
        strictly_nonnoop = sum(row["nonnoop_best"] and not row["noop_best"] for row in prompts.values())
        bstar = [row for row in records if row["action_type"] == "intervention" and float(row.get("alpha", 0)) == 40.0]
        best_action_counts = collections.Counter(action for row in prompts.values() for action in row["best_action_ids"])
        representative_counts = collections.Counter(row["representative_best_action_id"] for row in prompts.values())
        coefficient_rows.append({
            "setting_id": setting_id,
            "varied_coefficient": factor,
            "varied_value": format(value, ".12g"),
            "corruption_penalty": format(values["corruption_penalty"], ".12g"),
            "intervention_cost": format(values["intervention_cost"], ".12g"),
            "crypto_bonus": format(values["crypto_bonus"], ".12g"),
            "is_final_setting": str(values == FINAL).lower(),
            "prompt_count": len(prompts),
            "reward_record_count": len(records),
            "mean_base_reward": format(statistics.fmean(row["base_reward"] for row in records), ".12g"),
            "mean_regret_reward": format(statistics.fmean(row["regret_reward"] for row in records), ".12g"),
            "bstar_mimic_mean_base_reward": format(statistics.fmean(row["base_reward"] for row in bstar), ".12g"),
            "bstar_mimic_mean_regret_reward": format(statistics.fmean(row["regret_reward"] for row in bstar), ".12g"),
            "best_action_set_changes_vs_final": len(changed),
            "representative_best_action_changes_vs_final": len(representative_changed),
            "best_action_id_counts": json.dumps(dict(sorted(best_action_counts.items())), separators=(",", ":")),
            "representative_best_action_id_counts": json.dumps(dict(sorted(representative_counts.items())), separators=(",", ":")),
        })
        actionability_rows.append({
            "setting_id": setting_id,
            "varied_coefficient": factor,
            "varied_value": format(value, ".12g"),
            "prompt_count": len(prompts),
            "actionable_prompt_count_reference_rstar_gt_zero": actionable,
            "actionable_prompt_ratio": format(actionable / len(prompts), ".12g"),
            "noop_best_count_including_ties": noop_best,
            "noop_best_rate": format(noop_best / len(prompts), ".12g"),
            "best_nonnoop_count_including_ties": nonnoop_best,
            "best_nonnoop_rate": format(nonnoop_best / len(prompts), ".12g"),
            "strict_nonnoop_best_count": strictly_nonnoop,
            "strict_nonnoop_best_rate": format(strictly_nonnoop / len(prompts), ".12g"),
            "best_action_set_changes_vs_final": len(changed),
        })
        for prompt_id in sorted(prompts):
            current = prompts[prompt_id]
            baseline = final_prompts[prompt_id]
            change_rows.append({
                "setting_id": setting_id,
                "prompt_id": prompt_id,
                "best_action_ids": json.dumps(current["best_action_ids"], separators=(",", ":")),
                "representative_best_action_id": current["representative_best_action_id"],
                "representative_best_action_label": current["representative_best_action_label"],
                "r_star": format(current["r_star"], ".12g"),
                "final_best_action_ids": json.dumps(baseline["best_action_ids"], separators=(",", ":")),
                "best_action_set_changed": str(current["best_action_ids"] != baseline["best_action_ids"]).lower(),
            })
        summary_by_setting[setting_id] = {"actionable": actionable, "noop_best": noop_best, "nonnoop_best": nonnoop_best, "changed": len(changed)}

    coefficient_path = OUT / "reward_coefficient_sensitivity.csv"
    actionability_path = OUT / "reward_actionability_sensitivity.csv"
    changes_path = OUT / "reward_best_action_changes.csv"
    atomic_csv(coefficient_path, list(coefficient_rows[0]), coefficient_rows)
    atomic_csv(actionability_path, list(actionability_rows[0]), actionability_rows)
    atomic_csv(changes_path, list(change_rows[0]), change_rows)
    atomic_json(OUT / "reward_sensitivity_summary.json", {
        "schema_version": "phase19_reward_sensitivity_v1",
        "status": "PASS",
        "final_coefficients": FINAL,
        "settings": summary_by_setting,
        "final_reward_records_exactly_reproduced": True,
        "actionability_semantics": "reference max observed valid-candidate base_reward > 0",
        "tie_handling": "all best actions retained; lowest action_id is the deterministic representative only",
        "unsupported_action_outcomes": "not imputed; only saved per-prompt rollout actions evaluated",
        "inputs": {name: hash_record(path) for name, path in {
            "rollouts": rollout_path, "reference_rewards": reward_path,
            "action_map": action_path, "reference_reward_script": script_path,
        }.items()},
        "outputs": {name: hash_record(path) for name, path in {
            "coefficient_sensitivity": coefficient_path,
            "actionability_sensitivity": actionability_path,
            "best_action_changes": changes_path,
        }.items()},
        "policy_retrained": False,
        "b4_reselected": False,
        "b4_deployed": False,
        "generation_run": False,
        "scanner_run": False,
    })
    print(json.dumps({"status": "COMPLETE", "settings": len(SETTINGS), "prompts_per_setting": len(final_prompts), "reference_rewards_reproduced": True}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
