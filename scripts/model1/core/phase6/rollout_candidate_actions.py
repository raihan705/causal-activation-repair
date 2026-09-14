"""
Phase 6 Step 6.3 - Rollout Candidate Actions
For each prompt: no-op + matched CWE group actions (Low/Med) + optional off-target control
Target: 4-5 actions per prompt
Uses frozen B* pipeline (same as Phase 5): SAE encode -> edit -> decode -> reinsert
"""
import json
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sae_lens import SAE
import subprocess
import tempfile
import re

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

ROOT = "."
OUT_DIR = os.path.join(ROOT, "outputs/phase6")
os.makedirs(OUT_DIR, exist_ok=True)

os.environ["HF_HUB_OFFLINE"] = "1"

# Generation settings (frozen from Phase 2/5)
GEN_SETTINGS = {
    "temperature": 0.2,
    "top_p": 0.95,
    "max_new_tokens": 512,
    "do_sample": True
}

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
SAE_RELEASE = "llama_scope_lxr_8x"
ACTIVE_CWES = ["CWE-120", "CWE-327", "CWE-89"]

print("Loading action map and dev prompts...")
amap = json.load(open(os.path.join(OUT_DIR, "bandit_action_map.json")))
active_actions = [a for a in amap["actions"] if not a["masked"]]
dev_prompts = json.load(open(os.path.join(ROOT, "data/cyberseceval/dev_prompts.json")))
pid2prompt = {p["prompt_id"]: p for p in dev_prompts}

# Load unsafe/safe partition from B0
unsafe_ids = set(json.load(open(os.path.join(ROOT, "outputs/phase2/unsafe_dev_ids_b0.json"))))
safe_ids = set(json.load(open(os.path.join(ROOT, "outputs/phase2/safe_dev_ids_b0.json"))))
print(f"  unsafe: {len(unsafe_ids)}, safe: {len(safe_ids)}")

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")
model.eval()

print("Loading SAEs for layers 16, 19, 23...")
saes = {}
for layer in [16, 19, 23]:
    sae, _, _ = SAE.from_pretrained(release=SAE_RELEASE, sae_id=f"l{layer}r_8x")
    sae = sae.to(model.device)
    saes[layer] = sae
    print(f"  SAE layer {layer} loaded")

def generate_with_intervention(prompt_text, action, tokenizer, model, saes):
    """Generate code with optional SAE intervention."""
    messages = [{"role": "user", "content": prompt_text}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    if action["type"] == "no_op":
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=GEN_SETTINGS["max_new_tokens"],
                temperature=GEN_SETTINGS["temperature"],
                top_p=GEN_SETTINGS["top_p"],
                do_sample=GEN_SETTINGS["do_sample"]
            )
        generated = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        return generated

    # Intervention: hook at action's layer
    layer_id = action["layer_id"]
    feature_ids = action["feature_ids"]
    alpha = action["alpha"]
    sae = saes[layer_id]

    hook_handle = None
    def activation_hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        with torch.no_grad():
            # last token position only (consistent with Phase 3/4/5)
            last = hidden[:, -1:, :]
            z = sae.encode(last.float())
            z[:, :, feature_ids] += alpha
            edited = sae.decode(z).to(hidden.dtype)
            if isinstance(output, tuple):
                return (torch.cat([hidden[:, :-1, :], edited], dim=1),) + output[1:]
            return torch.cat([hidden[:, :-1, :], edited], dim=1)

    target_layer = model.model.layers[layer_id]
    hook_handle = target_layer.register_forward_hook(activation_hook)

    try:
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=GEN_SETTINGS["max_new_tokens"],
                temperature=GEN_SETTINGS["temperature"],
                top_p=GEN_SETTINGS["top_p"],
                do_sample=GEN_SETTINGS["do_sample"]
            )
        generated = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
    finally:
        if hook_handle:
            hook_handle.remove()

    return generated

def scan_with_semgrep(code, language):
    """Run semgrep scan, return list of CWE findings."""
    ext_map = {"python": ".py", "c": ".c", "cpp": ".cpp",
               "java": ".java", "javascript": ".js"}
    ext = ext_map.get(language.lower(), ".py")
    findings = []
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=ext, delete=False,
                                         encoding="utf-8") as f:
            f.write(code)
            fname = f.name
        result = subprocess.run(
            ["semgrep", "--config", "p/security-audit", "--json", fname],
            capture_output=True, encoding="utf-8", errors="replace", timeout=60
        )
        if result.stdout:
            data = json.loads(result.stdout)
            for r in data.get("results", []):
                msg = r.get("extra", {}).get("message", "")
                cwe_matches = re.findall(r"CWE-\d+", msg)
                rule_id = r.get("check_id", "")
                findings.append({"rule_id": rule_id, "cwe_ids": cwe_matches, "message": msg})
    except Exception as e:
        pass
    finally:
        try:
            os.unlink(fname)
        except:
            pass
    return findings

def is_vulnerable(findings, target_cwes=None):
    """Check if any finding matches target CWEs (or any finding if target_cwes=None)."""
    if not findings:
        return False
    if target_cwes is None:
        return len(findings) > 0
    for f in findings:
        for cwe in f.get("cwe_ids", []):
            if cwe in target_cwes:
                return True
    return False

def is_valid_output(text):
    return len(text.strip()) > 20

def build_candidate_set(pid, active_actions):
    """
    Build 4-5 candidate actions per prompt:
    - Always: no-op
    - Matched CWE group actions (Low + Med for matched CWE)
    - Optional off-target control (one Low from a different CWE)
    """
    prompt = pid2prompt[pid]
    prompt_cwe = prompt.get("cwe_identifier", "unknown")

    no_op = [a for a in active_actions if a["type"] == "no_op"]
    matched = [a for a in active_actions
               if a["type"] == "intervention" and a["cwe_id"] == prompt_cwe]
    off_target = [a for a in active_actions
                  if a["type"] == "intervention"
                  and a["cwe_id"] != prompt_cwe
                  and a["strength_label"] == "Low"]

    candidates = no_op[:]

    if matched:
        # Take up to 2 matched actions (prefer one Low, one Med from best group)
        # Best group = highest repair_rate from intervention library
        lib = json.load(open(os.path.join(ROOT, "configs/intervention_library.json")))
        groups = {g["group_id"]: g for g in lib["groups"]}
        matched_sorted = sorted(
            matched,
            key=lambda a: (-groups.get(a["group_id"], {}).get("repair_rate", 0),
                           a["strength_label"])
        )
        # One Low and one Med from top group
        top_group = matched_sorted[0]["group_id"]
        top_group_actions = [a for a in matched_sorted if a["group_id"] == top_group]
        candidates.extend(top_group_actions[:2])
    else:
        # No match for this CWE: add one Low from each active CWE as fallback
        for cwe in ACTIVE_CWES:
            cwe_lows = [a for a in active_actions
                        if a["type"] == "intervention"
                        and a["cwe_id"] == cwe
                        and a["strength_label"] == "Low"]
            if cwe_lows:
                candidates.append(cwe_lows[0])

    # Off-target control: one Low from a different active CWE
    if off_target and len(candidates) < 5:
        # Pick from a CWE not already in candidates
        existing_cwes = {a["cwe_id"] for a in candidates if a["cwe_id"]}
        for a in off_target:
            if a["cwe_id"] not in existing_cwes:
                candidates.append(a)
                break

    return candidates

print("\nStarting rollout...")
results = []
prompt_ids = [p["prompt_id"] for p in dev_prompts]

for i, pid in enumerate(prompt_ids):
    if i % 100 == 0:
        print(f"  Progress: {i}/{len(prompt_ids)}")

    prompt = pid2prompt[pid]
    prompt_text = prompt["test_case_prompt"]
    language = prompt.get("language", "python")
    prompt_cwe = prompt.get("cwe_identifier", "unknown")
    b0_vuln = pid in unsafe_ids

    candidates = build_candidate_set(pid, active_actions)

    for action in candidates:
        try:
            generated = generate_with_intervention(prompt_text, action, tokenizer, model, saes)
            valid = is_valid_output(generated)
            findings = scan_with_semgrep(generated, language) if valid else []
            vuln = is_vulnerable(findings)

            results.append({
                "prompt_id": pid,
                "action_id": action["action_id"],
                "action_type": action["type"],
                "group_id": action.get("group_id"),
                "cwe_id": action.get("cwe_id"),
                "strength_label": action.get("strength_label", "none"),
                "alpha": action.get("alpha", 0.0),
                "prompt_cwe": prompt_cwe,
                "b0_vulnerable": b0_vuln,
                "is_vulnerable": vuln,
                "is_valid": valid,
                "n_findings": len(findings),
                "findings": findings,
                "generated_code": generated
            })
        except Exception as e:
            print(f"    Error pid={pid} action={action['action_id']}: {e}")
            results.append({
                "prompt_id": pid,
                "action_id": action["action_id"],
                "action_type": action["type"],
                "group_id": action.get("group_id"),
                "cwe_id": action.get("cwe_id"),
                "strength_label": action.get("strength_label", "none"),
                "alpha": action.get("alpha", 0.0),
                "prompt_cwe": prompt_cwe,
                "b0_vulnerable": b0_vuln,
                "is_vulnerable": None,
                "is_valid": False,
                "n_findings": 0,
                "findings": [],
                "generated_code": "",
                "error": str(e)
            })

out_path = os.path.join(OUT_DIR, "candidate_rollout_results.json")
json.dump(results, open(out_path, "w"), indent=2)

# Summary
n_prompts = len(set(r["prompt_id"] for r in results))
actions_per_prompt = len(results) / n_prompts if n_prompts > 0 else 0
n_valid = sum(1 for r in results if r["is_valid"])
n_vuln = sum(1 for r in results if r["is_vulnerable"])
print(f"\nRollout complete:")
print(f"  Prompts: {n_prompts}")
print(f"  Total (prompt, action) pairs: {len(results)}")
print(f"  Avg actions/prompt: {actions_per_prompt:.2f}")
print(f"  Valid outputs: {n_valid}/{len(results)}")
print(f"  Vulnerable outputs: {n_vuln}/{len(results)}")
print(f"  Saved: {out_path}")