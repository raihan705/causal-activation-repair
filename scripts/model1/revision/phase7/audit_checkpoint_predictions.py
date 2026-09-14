#!/usr/bin/env python3
"""Read-only Phase 7 checkpoint loadability and validation-prediction audit.

This revision-local script never trains, updates, masks, or selects a policy. It
loads the frozen Phase 7 checkpoints on CPU and evaluates raw 28-action argmax
predictions against the frozen Phase 7 validation records.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


STATE_DIM = 102
N_ACTIONS = 28
N_VALIDATION_PROMPTS = 269
N_ACTIONABLE_PROMPTS = 18
N_CHECKPOINTS = 12
N_EXPECTED_PREDICTIONS = 3228

EXPECTED_SOURCE_HASHES = {
    "outputs/phase7/offline_bandit_val.jsonl": "06f69c7c2d2f4d98988ee1fe94be9e4a0c6d5d6fc84fa2a90211b06770351496",
    "outputs/phase6/bandit_action_map.json": "67ed4f4d81d8edfe5096a50587937fc0eaeaa4c2d576949e7ec9ef21035a9193",
    "outputs/phase7/checkpoint_validation_metrics.csv": "8602e42adc2b3cf6905b2788caa857e8aa9530f2f0a9edb97ea970b3ba232a0d",
    "outputs/phase6/offline_bandit_dataset.jsonl": "d251e5d816691041819ed85b770e63845d240c372fa2ade680906421e3cdd1eb",
}

EXPECTED_CHECKPOINT_HASHES = {
    "linear_bandit_epoch_01.ckpt": "c4c38bc404223cc6e37632ff781487ded42a4c21c4ba0fe7983838a9cd33fa89",
    "linear_bandit_epoch_02.ckpt": "77cd7ec1eacfed4db5afbb12be45cfec5712228f2ba8488e9adcdcc6ec7d1a75",
    "linear_bandit_epoch_03.ckpt": "b0663e97296b96b3c49cbde03fb0d447dd7b9ddf96b379a9273fad96826a48f6",
    "linear_bandit_epoch_04.ckpt": "6cbf3911dafcef8c60b21dff89189ae65530a879dd8e5b9348f6af2ad54388ab",
    "linear_bandit_epoch_05.ckpt": "f314dccfdf6f02cde6b5656658332e44ca9c74d03f073e325feb395d94d5ac6c",
    "linear_bandit_epoch_06.ckpt": "41e9b18d32791b5ace500019c5b8215a245a8f76a83b11d54dbbc2a66fc81995",
    "mlp_bandit_epoch_01.ckpt": "eb00cf14e4928bea214a2e88011c9502bc756c020e1fca2aa5c2a636b0efccf7",
    "mlp_bandit_epoch_02.ckpt": "d628481d39473a462bdd798d597e6291d7c8e077a3ee8fadaec2c6f071e624e7",
    "mlp_bandit_epoch_03.ckpt": "1c68d1c063b30846bebdc702d39a56016d197fa3d965ecbb4e0805a14138bd3e",
    "mlp_bandit_epoch_04.ckpt": "5f53b91643e36fabbaa286d50041046c38f6d9ab367d44a6a803f2f90a256f69",
    "mlp_bandit_epoch_05.ckpt": "d1665a9a6fcb4d2fa1a87090da9ceca76d4686ef54f1f23ed754ba2874e190d2",
    "mlp_bandit_epoch_06.ckpt": "d006be3e04ed15108983287fb83fae6d32853d49e721462c4e21a602ab393155",
}


class LinearPolicy(nn.Module):
    def __init__(self, state_dim: int, n_actions: int) -> None:
        super().__init__()
        self.fc = nn.Linear(state_dim, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MLPPolicy(nn.Module):
    def __init__(
        self, state_dim: int, n_actions: int, hidden: int = 64, dropout: float = 0.1
    ) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def safe_torch_load(path: Path) -> dict[str, Any]:
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        loaded = torch.load(path, map_location="cpu")
    if not isinstance(loaded, dict) or "model_state" not in loaded:
        raise ValueError(f"Checkpoint has no model_state dictionary: {path}")
    return loaded


def make_model(policy: str) -> nn.Module:
    if policy == "linear":
        model: nn.Module = LinearPolicy(STATE_DIM, N_ACTIONS)
    elif policy == "mlp":
        model = MLPPolicy(STATE_DIM, N_ACTIONS)
    else:
        raise ValueError(f"Unknown policy architecture: {policy}")
    model.to("cpu")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def strict_load_model(model: nn.Module, checkpoint: dict[str, Any]) -> dict[str, Any]:
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=False)
    result = {
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }
    if result["missing_keys"] or result["unexpected_keys"]:
        raise ValueError(f"State-dict incompatibility: {result}")
    return result


def group_validation(
    rows: list[dict[str, Any]],
) -> tuple[list[Any], dict[Any, list[dict[str, Any]]]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["prompt_id"]].append(row)
    return list(grouped.keys()), dict(grouped)


def verified_sources(repo: Path) -> dict[str, dict[str, Any]]:
    paths = list(EXPECTED_SOURCE_HASHES) + [
        "phases/phase7/split_dev_for_bandit.py",
        "phases/phase7/train_bandit_policy.py",
        "phases/phase7/validate_bandit_policy_offline.py",
        "revision/model1/phase7/scripts/audit_checkpoint_predictions.py",
    ]
    result: dict[str, dict[str, Any]] = {}
    for relative in paths:
        path = repo / relative
        actual = sha256(path)
        expected = EXPECTED_SOURCE_HASHES.get(relative)
        result[relative] = {
            "sha256": actual,
            "expected_sha256": expected,
            "hash_verified": expected is None or actual == expected,
        }
        if expected is not None and actual != expected:
            raise ValueError(f"Frozen source hash mismatch for {relative}: {actual}")
    return result


def checkpoint_manifest(repo: Path) -> list[dict[str, Any]]:
    ckpt_dir = repo / "outputs/phase7/checkpoints"
    actual_names = sorted(path.name for path in ckpt_dir.glob("*.ckpt"))
    expected_names = sorted(EXPECTED_CHECKPOINT_HASHES)
    if actual_names != expected_names:
        raise ValueError(
            f"Checkpoint-set mismatch: expected={expected_names}, actual={actual_names}"
        )
    manifest = []
    for name in expected_names:
        path = ckpt_dir / name
        actual_hash = sha256(path)
        expected_hash = EXPECTED_CHECKPOINT_HASHES[name]
        if actual_hash != expected_hash:
            raise ValueError(f"Checkpoint hash mismatch for {name}: {actual_hash}")
        match = re.fullmatch(r"(linear|mlp)_bandit_epoch_(\d{2})\.ckpt", name)
        if match is None:
            raise ValueError(f"Unexpected checkpoint filename: {name}")
        manifest.append(
            {
                "checkpoint": name,
                "path": f"outputs/phase7/checkpoints/{name}",
                "sha256": actual_hash,
                "policy": match.group(1),
                "epoch": int(match.group(2)),
                "size_bytes": path.stat().st_size,
            }
        )
    if len(manifest) != N_CHECKPOINTS:
        raise ValueError(f"Expected {N_CHECKPOINTS} checkpoints; found {len(manifest)}")
    return manifest


def environment_record() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": "cpu",
        "dtype": "float32",
    }


def architecture_record() -> dict[str, Any]:
    return {
        "state_dim": STATE_DIM,
        "output_dim": N_ACTIONS,
        "linear": "LinearPolicy: Linear(102, 28)",
        "mlp": (
            "MLPPolicy: Linear(102,64), ReLU, Dropout(0.1), "
            "Linear(64,64), ReLU, Dropout(0.1), Linear(64,28)"
        ),
        "inference": "model.eval(); torch.no_grad(); raw argmax over 28 actions",
        "prediction_mask_applied": False,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False, allow_nan=False)
        handle.write("\n")


def base_payload(mode: str, repo: Path) -> dict[str, Any]:
    return {
        "schema_version": "phase7-checkpoint-prediction-audit-v1",
        "mode": mode,
        "status": "IN_PROGRESS",
        "environment": environment_record(),
        "architecture": architecture_record(),
        "heldout_used": False,
        "policy_updated": False,
        "optimizer_created": False,
        "gradients_computed": False,
        "source_files": verified_sources(repo),
    }


def loadability_audit(repo: Path) -> dict[str, Any]:
    payload = base_payload("loadability", repo)
    checkpoint_name = "linear_bandit_epoch_01.ckpt"
    checkpoint_path = repo / "outputs/phase7/checkpoints" / checkpoint_name
    actual_hash = sha256(checkpoint_path)
    expected_hash = EXPECTED_CHECKPOINT_HASHES[checkpoint_name]
    if actual_hash != expected_hash:
        raise ValueError(f"Checkpoint hash mismatch: {actual_hash}")

    validation_rows = read_jsonl(repo / "outputs/phase7/offline_bandit_val.jsonl")
    prompt_ids, grouped = group_validation(validation_rows)
    if not prompt_ids:
        raise ValueError("No saved validation state is available")
    prompt_id = prompt_ids[0]
    state_values = grouped[prompt_id][0]["state"]
    if len(state_values) != STATE_DIM:
        raise ValueError(f"Expected state dimension {STATE_DIM}; found {len(state_values)}")

    checkpoint = safe_torch_load(checkpoint_path)
    model = make_model("linear")
    compatibility = strict_load_model(model, checkpoint)
    state = torch.tensor([state_values], dtype=torch.float32, device="cpu")
    with torch.no_grad():
        logits = model(state)
    if tuple(logits.shape) != (1, N_ACTIONS):
        raise ValueError(f"Expected output shape (1, {N_ACTIONS}); found {tuple(logits.shape)}")
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError("Non-finite loadability output")
    prediction = int(logits.argmax(dim=1).item())
    action_map = json.loads(
        (repo / "outputs/phase6/bandit_action_map.json").read_text(encoding="utf-8")
    )
    action_by_id = {
        int(action["action_id"]): action for action in action_map["actions"]
    }
    global_rows = read_jsonl(repo / "outputs/phase6/offline_bandit_dataset.jsonl")
    globally_observed_actions = {int(row["action_id"]) for row in global_rows}
    outcome_available = any(
        int(record["action_id"]) == prediction for record in grouped[prompt_id]
    )
    predicted_action = action_by_id[prediction]
    payload.update(
        {
            "status": "VERIFIED",
            "checkpoint": {
                "name": checkpoint_name,
                "path": f"outputs/phase7/checkpoints/{checkpoint_name}",
                "sha256": actual_hash,
                "expected_sha256": expected_hash,
                "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
                "state_dict_compatibility": compatibility,
            },
            "saved_validation_state": {
                "source": "outputs/phase7/offline_bandit_val.jsonl",
                "source_sha256": EXPECTED_SOURCE_HASHES[
                    "outputs/phase7/offline_bandit_val.jsonl"
                ],
                "prompt_id": prompt_id,
                "state_dim": len(state_values),
                "state_shape": [1, len(state_values)],
                "state_dtype": "float32",
            },
            "inference": {
                "output_shape": list(logits.shape),
                "all_finite": True,
                "raw_argmax_action_id": prediction,
                "no_op": prediction == 0,
                "strength_label": predicted_action.get("strength_label"),
                "action_support": {
                    "global_masked": bool(predicted_action.get("masked", False)),
                    "rollout_supported_globally": prediction
                    in globally_observed_actions,
                    "outcome_available_for_this_prompt": outcome_available,
                },
            },
            "validation_checks": {
                "checkpoint_hash_exact": True,
                "cpu_only": logits.device.type == "cpu",
                "model_eval": not model.training,
                "missing_keys_zero": not compatibility["missing_keys"],
                "unexpected_keys_zero": not compatibility["unexpected_keys"],
                "output_dim_28": logits.shape[1] == N_ACTIONS,
                "all_outputs_finite": True,
                "no_gradients": all(parameter.grad is None for parameter in model.parameters()),
                "no_policy_update": True,
            },
        }
    )
    return payload


def load_saved_metrics(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["checkpoint"]: row for row in csv.DictReader(handle)}


def validation_audit(repo: Path) -> dict[str, Any]:
    payload = base_payload("validation-audit", repo)
    manifest = checkpoint_manifest(repo)
    validation_rows = read_jsonl(repo / "outputs/phase7/offline_bandit_val.jsonl")
    prompt_ids, grouped = group_validation(validation_rows)
    if len(prompt_ids) != N_VALIDATION_PROMPTS:
        raise ValueError(
            f"Expected {N_VALIDATION_PROMPTS} validation prompts; found {len(prompt_ids)}"
        )
    duplicate_prompt_action_rows = len(validation_rows) - len(
        {(row["prompt_id"], int(row["action_id"])) for row in validation_rows}
    )
    if duplicate_prompt_action_rows:
        raise ValueError(
            f"Validation data has {duplicate_prompt_action_rows} duplicate prompt/action keys"
        )

    states = torch.tensor(
        [grouped[prompt_id][0]["state"] for prompt_id in prompt_ids],
        dtype=torch.float32,
        device="cpu",
    )
    if tuple(states.shape) != (N_VALIDATION_PROMPTS, STATE_DIM):
        raise ValueError(f"Unexpected validation state shape: {tuple(states.shape)}")

    action_map = json.loads(
        (repo / "outputs/phase6/bandit_action_map.json").read_text(encoding="utf-8")
    )
    actions = action_map["actions"]
    if len(actions) != N_ACTIONS:
        raise ValueError(f"Expected {N_ACTIONS} actions; found {len(actions)}")
    action_by_id = {int(action["action_id"]): action for action in actions}
    if set(action_by_id) != set(range(N_ACTIONS)):
        raise ValueError("Action map is not exactly the raw action space 0..27")
    masked_ids = {
        action_id for action_id, action in action_by_id.items() if action.get("masked", False)
    }
    high_ids = {
        action_id
        for action_id, action in action_by_id.items()
        if action.get("strength_label") == "High"
    }

    global_rows = read_jsonl(repo / "outputs/phase6/offline_bandit_dataset.jsonl")
    globally_observed_actions = sorted({int(row["action_id"]) for row in global_rows})
    globally_observed_set = set(globally_observed_actions)
    actionable_ids = {
        prompt_id
        for prompt_id, records in grouped.items()
        if max(float(record["base_reward"]) for record in records) > 0.0
    }
    if len(actionable_ids) != N_ACTIONABLE_PROMPTS:
        raise ValueError(
            f"Expected {N_ACTIONABLE_PROMPTS} actionable prompts; found {len(actionable_ids)}"
        )

    saved_metrics = load_saved_metrics(
        repo / "outputs/phase7/checkpoint_validation_metrics.csv"
    )
    if sorted(saved_metrics) != sorted(EXPECTED_CHECKPOINT_HASHES):
        raise ValueError("Saved metric checkpoint set differs from frozen checkpoint set")

    predictions: list[dict[str, Any]] = []
    checkpoint_summaries: list[dict[str, Any]] = []
    all_outputs_finite = True
    output_dimensions: set[int] = set()

    for checkpoint_info in manifest:
        name = checkpoint_info["checkpoint"]
        checkpoint = safe_torch_load(repo / checkpoint_info["path"])
        model = make_model(checkpoint_info["policy"])
        compatibility = strict_load_model(model, checkpoint)
        if int(checkpoint.get("epoch", -1)) != checkpoint_info["epoch"]:
            raise ValueError(f"Checkpoint epoch metadata mismatch for {name}")

        with torch.no_grad():
            logits = model(states)
        output_dimensions.add(int(logits.shape[1]))
        finite = bool(torch.isfinite(logits).all().item())
        all_outputs_finite = all_outputs_finite and finite
        if tuple(logits.shape) != (N_VALIDATION_PROMPTS, N_ACTIONS) or not finite:
            raise ValueError(f"Invalid model output for {name}: {tuple(logits.shape)}, finite={finite}")
        predicted_actions = logits.argmax(dim=1).tolist()

        noop_count = 0
        masked_count = 0
        unmasked_globally_unobserved_count = 0
        globally_observed_prompt_unavailable_count = 0
        exact_available_count = 0
        high_count = 0
        intervention_count = 0
        top1_correct = 0
        actionable_available_count = 0
        actionable_chosen_rewards: list[float] = []
        actionable_best_rewards: list[float] = []

        for index, prompt_id in enumerate(prompt_ids):
            records = grouped[prompt_id]
            pred_action = int(predicted_actions[index])
            action = action_by_id[pred_action]
            exact_record = next(
                (record for record in records if int(record["action_id"]) == pred_action),
                None,
            )
            global_masked = pred_action in masked_ids
            rollout_supported_globally = pred_action in globally_observed_set
            outcome_available = exact_record is not None
            is_noop = pred_action == 0
            is_actionable = prompt_id in actionable_ids
            best_reward = max(float(record["base_reward"]) for record in records)
            best_records = [record for record in records if record.get("is_best_action")]
            if not best_records:
                best_records = [max(records, key=lambda record: float(record["base_reward"]))]
            first_best_label = int(best_records[0]["action_id"])
            tied_best_ids = sorted(
                int(record["action_id"])
                for record in records
                if math.isclose(
                    float(record["base_reward"]), best_reward, rel_tol=0.0, abs_tol=1e-12
                )
            )

            noop_count += int(is_noop)
            intervention_count += int(not is_noop)
            masked_count += int(global_masked)
            unmasked_globally_unobserved_count += int(
                not global_masked and not rollout_supported_globally
            )
            globally_observed_prompt_unavailable_count += int(
                rollout_supported_globally and not outcome_available
            )
            exact_available_count += int(outcome_available)
            high_count += int(pred_action in high_ids)
            top1_correct += int(pred_action == first_best_label)

            outcome_status = (
                "EXACT_PROMPT_ACTION_OUTCOME_AVAILABLE"
                if outcome_available
                else "NOT_REPRODUCIBLE_MISSING_ACTION_OUTCOME"
            )
            chosen_reward = (
                float(exact_record["base_reward"]) if exact_record is not None else None
            )
            if is_actionable and outcome_available:
                actionable_available_count += 1
                actionable_chosen_rewards.append(float(chosen_reward))
                actionable_best_rewards.append(best_reward)

            predictions.append(
                {
                    "checkpoint": name,
                    "checkpoint_sha256": checkpoint_info["sha256"],
                    "policy": checkpoint_info["policy"],
                    "epoch": checkpoint_info["epoch"],
                    "prompt_id": prompt_id,
                    "actionable": is_actionable,
                    "raw_argmax_action_id": pred_action,
                    "no_op": is_noop,
                    "action_type": action.get("type"),
                    "strength_label": action.get("strength_label"),
                    "global_masked": global_masked,
                    "rollout_supported_globally": rollout_supported_globally,
                    "outcome_available_for_this_prompt": outcome_available,
                    "outcome_status": outcome_status,
                    "chosen_base_reward": chosen_reward,
                    "best_available_base_reward": best_reward,
                    "first_best_action_label": first_best_label,
                    "tied_best_action_ids": tied_best_ids,
                    "matches_first_best_label": pred_action == first_best_label,
                    "matches_any_tied_best_action": pred_action in tied_best_ids,
                }
            )

        historical_rcr = (
            (sum(actionable_chosen_rewards) / len(actionable_chosen_rewards))
            / (sum(actionable_best_rewards) / len(actionable_best_rewards))
            if actionable_best_rewards and sum(actionable_best_rewards) != 0.0
            else 0.0
        )
        actionable_missing_count = N_ACTIONABLE_PROMPTS - actionable_available_count
        if actionable_missing_count == 0:
            complete_rcr_value: float | None = historical_rcr
            complete_rcr_status = "ESTIMABLE_ALL_ACTIONABLE_OUTCOMES_AVAILABLE"
        else:
            complete_rcr_value = None
            complete_rcr_status = "NOT_ESTIMABLE_MISSING_ACTION_OUTCOMES"

        n_prompts = len(prompt_ids)
        noop_rate = noop_count / n_prompts
        unsafe_rate = masked_count / n_prompts
        unsupported_rate = (n_prompts - exact_available_count) / n_prompts
        high_rate = high_count / intervention_count if intervention_count else 0.0
        top1_accuracy = top1_correct / n_prompts
        recomputed_rounded = {
            "rcr": round(historical_rcr, 4),
            "top1_acc": round(top1_accuracy, 4),
            "noop_rate": round(noop_rate, 4),
            "unsafe_rate": round(unsafe_rate, 4),
            "high_rate": round(high_rate, 4),
            "n_actionable_evaluated": actionable_available_count,
        }
        saved = saved_metrics[name]
        saved_values: dict[str, float | int] = {
            "rcr": float(saved["rcr"]),
            "top1_acc": float(saved["top1_acc"]),
            "noop_rate": float(saved["noop_rate"]),
            "unsafe_rate": float(saved["unsafe_rate"]),
            "high_rate": float(saved["high_rate"]),
            "n_actionable_evaluated": int(saved["n_actionable_evaluated"]),
        }
        comparison = {
            key: {
                "saved": saved_values[key],
                "recomputed_rounded": recomputed_rounded[key],
                "exact_match": saved_values[key] == recomputed_rounded[key],
            }
            for key in recomputed_rounded
        }
        exact_saved_agreement = all(item["exact_match"] for item in comparison.values())
        gates = {
            "historical_conditional_rcr_at_least_0_70": historical_rcr >= 0.70,
            "submitted_unsafe_rate_at_most_0_01": unsafe_rate <= 0.01,
            "noop_rate_at_most_0_90": noop_rate <= 0.90,
        }
        gates["all_submitted_deployment_gates_pass"] = all(gates.values())

        checkpoint_summaries.append(
            {
                **checkpoint_info,
                "checkpoint_epoch_metadata": int(checkpoint["epoch"]),
                "state_dict_compatibility": compatibility,
                "total_validation_prompts": n_prompts,
                "total_actionable_prompts": N_ACTIONABLE_PROMPTS,
                "actionable_outcome_available_count": actionable_available_count,
                "actionable_outcome_missing_count": actionable_missing_count,
                "prediction_category_counts": {
                    "no_op": noop_count,
                    "globally_masked": masked_count,
                    "unmasked_but_globally_unobserved": unmasked_globally_unobserved_count,
                    "globally_observed_but_unavailable_for_this_prompt": globally_observed_prompt_unavailable_count,
                    "exact_outcome_available": exact_available_count,
                    "missing_exact_prompt_action_outcome": n_prompts - exact_available_count,
                },
                "rates": {
                    "no_op_rate": noop_rate,
                    "submitted_unsafe_rate": unsafe_rate,
                    "unsupported_rate": unsupported_rate,
                    "high_rate_submitted_definition": high_rate,
                },
                "historical_conditional_rcr": {
                    "value": historical_rcr,
                    "n_actionable_evaluated": actionable_available_count,
                    "definition": "actionable prompts with exact predicted-action outcomes only",
                },
                "complete_actionable_rcr": {
                    "value": complete_rcr_value,
                    "status": complete_rcr_status,
                    "n_actionable_required": N_ACTIONABLE_PROMPTS,
                },
                "top1_accuracy_first_best_label": top1_accuracy,
                "submitted_gate_results": gates,
                "saved_metrics_comparison": {
                    "all_metrics_exact_after_four_decimal_rounding": exact_saved_agreement,
                    "metrics": comparison,
                },
                "output_shape": [n_prompts, N_ACTIONS],
                "all_outputs_finite": finite,
                "no_gradients": all(parameter.grad is None for parameter in model.parameters()),
                "policy_updated": False,
            }
        )

    duplicate_prediction_keys = len(predictions) - len(
        {(row["checkpoint"], row["prompt_id"]) for row in predictions}
    )
    if len(predictions) != N_EXPECTED_PREDICTIONS or duplicate_prediction_keys:
        raise ValueError(
            f"Prediction cardinality invalid: rows={len(predictions)}, "
            f"duplicate_keys={duplicate_prediction_keys}"
        )
    saved_metrics_all_exact = all(
        summary["saved_metrics_comparison"][
            "all_metrics_exact_after_four_decimal_rounding"
        ]
        for summary in checkpoint_summaries
    )
    any_checkpoint_passes = any(
        summary["submitted_gate_results"]["all_submitted_deployment_gates_pass"]
        for summary in checkpoint_summaries
    )
    payload.update(
        {
            "status": "VERIFIED",
            "checkpoint_manifest": manifest,
            "validation_population": {
                "path": "outputs/phase7/offline_bandit_val.jsonl",
                "sha256": EXPECTED_SOURCE_HASHES[
                    "outputs/phase7/offline_bandit_val.jsonl"
                ],
                "validation_prompt_count": len(prompt_ids),
                "actionable_prompt_count": len(actionable_ids),
                "validation_record_count": len(validation_rows),
                "duplicate_prompt_action_rows": duplicate_prompt_action_rows,
            },
            "support_provenance": {
                "path": "outputs/phase6/offline_bandit_dataset.jsonl",
                "sha256": EXPECTED_SOURCE_HASHES[
                    "outputs/phase6/offline_bandit_dataset.jsonl"
                ],
                "globally_observed_action_ids": globally_observed_actions,
                "classification_fields_are_independent": [
                    "global_masked",
                    "rollout_supported_globally",
                    "outcome_available_for_this_prompt",
                ],
            },
            "checkpoint_summaries": checkpoint_summaries,
            "per_prompt_predictions": predictions,
            "audit_invariants": {
                "checkpoint_count": len(manifest),
                "unique_checkpoint_count": len({item["checkpoint"] for item in manifest}),
                "validation_prompts_per_checkpoint": len(prompt_ids),
                "prediction_row_count": len(predictions),
                "expected_prediction_row_count": N_EXPECTED_PREDICTIONS,
                "duplicate_checkpoint_prompt_keys": duplicate_prediction_keys,
                "all_outputs_finite": all_outputs_finite,
                "output_dimensions_observed": sorted(output_dimensions),
                "raw_action_space_size": N_ACTIONS,
                "prediction_mask_applied": False,
                "optimizer_created": False,
                "gradients_computed": False,
                "policy_updated": False,
                "heldout_used": False,
                "saved_metrics_all_exact_after_four_decimal_rounding": saved_metrics_all_exact,
            },
            "frozen_gate_conclusion": {
                "any_checkpoint_passes_all_submitted_gates": any_checkpoint_passes,
                "b4_deployment_decision": "NOT_DEPLOYED",
                "checkpoint_reselection_performed": False,
                "contradicts_original_phase7_conclusion": any_checkpoint_passes,
            },
        }
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("loadability", "validation-audit"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path(__file__).resolve().parents[4]
    output = args.output if args.output.is_absolute() else repo / args.output
    try:
        if args.mode == "loadability":
            payload = loadability_audit(repo)
        else:
            payload = validation_audit(repo)
    except Exception as exc:
        failure = {
            "schema_version": "phase7-checkpoint-prediction-audit-v1",
            "mode": args.mode,
            "status": "FAILED_CLOSED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "heldout_used": False,
            "policy_updated": False,
            "optimizer_created": False,
            "gradients_computed": False,
        }
        write_json(output, failure)
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 1
    write_json(output, payload)
    print(json.dumps({"status": payload["status"], "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
