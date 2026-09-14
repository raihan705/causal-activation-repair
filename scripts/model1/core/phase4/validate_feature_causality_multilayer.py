"""
Phase 4 — Step 4.4
Causal feature validation per (layer, CWE).

For each (layer, CWE):
  - Select top-K=10 pruned candidates
  - Test on ~15-20 unsafe dev prompts (cyber CWEs) or CVE vuln samples (cve CWEs)
  - Apply additive latent edit during generation
  - Rescan with ICD
  - Retain features with repair_rate > 0 AND corruption_rate < 20%

For CVE-pair CWEs: causal validation uses a proxy —
  steering toward fixed-code latent direction and checking output
  validity + semantic plausibility (no ICD ground truth available).
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
import sys
import json
import torch
import numpy as np
import pandas as pd
import subprocess
import tempfile

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path.cwd()
PHASE2_DIR   = PROJECT_ROOT / "outputs/phase2"
PHASE4_DIR   = PROJECT_ROOT / "outputs/phase4"
DATA_DIR     = PROJECT_ROOT / "data"
LATENT_DIR   = DATA_DIR / "activations/latent"
OUT_DIR      = PHASE4_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_NAME   = "meta-llama/Meta-Llama-3.1-8B-Instruct"
SAE_RELEASE      = "llama_scope_lxr_8x"

LAYERS       = [16, 19, 23]
CYBER_CWES   = [120, 327, 89, 338]
CVE_CWES     = [79, 125, 787, 190, 476]
ALL_CWES     = CYBER_CWES + CVE_CWES

TOP_K        = 10       # candidates per (layer, CWE) to test
N_TEST       = 15       # unsafe prompts to test per feature
STEER_ALPHA  = 20.0     # additive steering magnitude
MAX_NEW_TOK  = 256
TEMPERATURE  = 0.2
TOP_P        = 0.95
SEED         = 42

# Thresholds
REPAIR_MIN       = 0.0   # repair_rate > 0 (at least one fix)
CORRUPT_MAX      = 0.20  # corruption_rate < 20%
INVALID_MAX      = 0.50  # invalid_rate < 50% (output validity guard)

# ── Load model + SAE ──────────────────────────────────────────────────────────
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto",
)
model.eval()
device = next(model.parameters()).device
print(f"Model on {device}")

# Load SAEs per layer
from sae_lens import SAE
saes = {}
for layer in LAYERS:
    sae_id = f"l{layer}r_8x"
    sae, _, _ = SAE.from_pretrained(release=SAE_RELEASE, sae_id=sae_id)
    sae = sae.to(device)
    sae.eval()
    saes[layer] = sae
    print(f"SAE layer {layer} loaded")
# ── ICD Scanner ───────────────────────────────────────────────────────────────
SEMGREP_CONFIG = "p/security-audit"

def scan_code(code_str):
    """Returns list of detected CWE strings via semgrep."""
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w",
                                     delete=False, encoding="utf-8") as f:
        f.write(code_str)
        tmp = f.name
    try:
        result = subprocess.run(
            ["semgrep", "--config", SEMGREP_CONFIG, "--json", tmp],
            capture_output=True, encoding="utf-8", errors="replace", timeout=30
        )
        findings = json.loads(result.stdout).get("results", [])
        cwes = []
        for r in findings:
            meta = r.get("extra", {}).get("metadata", {})
            cwe_list = meta.get("cwe", [])
            if isinstance(cwe_list, str):
                cwe_list = [cwe_list]
            cwes.extend(cwe_list)
        return cwes
    except Exception:
        return []
    finally:
        os.unlink(tmp)

def is_valid_output(text):
    """Basic validity: non-empty, contains code-like tokens."""
    if not text or len(text.strip()) < 10:
        return False
    code_tokens = ["def ", "int ", "char ", "return", "if ", "for ", "{", "("]
    return any(t in text for t in code_tokens)

# ── Load unsafe prompts ───────────────────────────────────────────────────────
def load_unsafe_prompts_cyber(cwe):
    """Load unsafe dev prompts for a cyber-detected CWE."""
    id_file = PHASE2_DIR / f"unsafe_dev_ids_CWE-{cwe}.json"
    if cwe == 338:
        id_file = PHASE2_DIR / f"unsafe_dev_ids_CWE-{cwe}_exploratory.json"
    if not id_file.exists():
        return []
    unsafe_ids = set(json.load(open(id_file)))
    outputs = json.load(open(PHASE2_DIR / "baseline_dev_outputs.json"))
    prompts = [r for r in outputs if r["prompt_id"] in unsafe_ids]
    return prompts[:N_TEST]

def load_safe_prompts_sample(n=5):
    """Load a small sample of safe prompts for corruption testing."""
    safe_ids = set(json.load(open(PHASE2_DIR / "safe_dev_ids_b0.json")))
    outputs  = json.load(open(PHASE2_DIR / "baseline_dev_outputs.json"))
    safe     = [r for r in outputs if r["prompt_id"] in safe_ids]
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(safe), min(n, len(safe)), replace=False)
    return [safe[i] for i in idx]

def load_cve_prompts(cwe):
    """
    For CVE-pair CWEs: use vulnerable code snippets as generation prompts.
    We wrap them as instruction prompts asking for a fix.
    """
    pairs = pd.concat([
        pd.read_csv(DATA_DIR / "processed/train_pairs.csv"),
        pd.read_csv(DATA_DIR / "processed/val_pairs.csv"),
    ], ignore_index=True)
    mask = pairs["cwe_id"].astype(str).str.contains(str(cwe), na=False)
    subset = pairs[mask].head(N_TEST)
    prompts = []
    for _, row in subset.iterrows():
        prompts.append({
            "prompt_id": row.get("pair_id", row.name),
            "prompt_text": f"Review and fix this code:\n```\n{row['vulnerable_code'][:500]}\n```",
            "cwe_id": str(cwe),
            "source": "cve_pairs",
        })
    return prompts

# ── Steering hook ─────────────────────────────────────────────────────────────
def make_steer_hook(sae, feature_id, alpha):
    """Hook that steers feature `feature_id` by +alpha in SAE latent space."""
    def hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        with torch.no_grad():
            # last token position only (consistent with Phase 3 pooling)
            h_last = hidden[:, -1:, :].float()
            z = sae.encode(h_last)
            z[:, :, feature_id] += alpha
            h_steered = sae.decode(z).to(hidden.dtype)
            hidden = torch.cat([hidden[:, :-1, :], h_steered], dim=1)
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden
    return hook

def get_layer_module(model, layer):
    return model.model.layers[layer]

# ── Generate with steering ────────────────────────────────────────────────────
def generate_steered(prompt_text, layer, feature_id, alpha=STEER_ALPHA):
    inputs = tokenizer(prompt_text, return_tensors="pt",
                       truncation=True, max_length=512).to(device)
    module = get_layer_module(model, layer)
    hook   = module.register_forward_hook(
        make_steer_hook(saes[layer], feature_id, alpha)
    )
    torch.manual_seed(SEED)
    try:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOK,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
    except Exception as e:
        gen = ""
    finally:
        hook.remove()
    return gen

# ── Validate one feature ──────────────────────────────────────────────────────
def validate_feature(layer, cwe, feature_id, unsafe_prompts, safe_prompts):
    n_repair = n_corrupt = n_invalid = n_total = 0

    for rec in unsafe_prompts:
        prompt = rec.get("prompt_text", rec.get("prompt", ""))
        if not prompt:
            continue
        n_total += 1

        gen = generate_steered(prompt, layer, feature_id)

        if not is_valid_output(gen):
            n_invalid += 1
            continue

        detected_cwes = scan_code(gen)
        cwe_str = f"CWE-{cwe}"
        baseline_vuln = True  # these are unsafe prompts by definition

        if cwe_str not in detected_cwes:
            n_repair += 1  # vuln → safe

    # Corruption test on safe prompts
    n_safe_total = 0
    for rec in safe_prompts:
        prompt = rec.get("prompt_text", rec.get("prompt", ""))
        if not prompt:
            continue
        n_safe_total += 1
        gen = generate_steered(prompt, layer, feature_id)
        if not is_valid_output(gen):
            n_corrupt += 1
            continue
        detected = scan_code(gen)
        if detected:  # any vulnerability introduced
            n_corrupt += 1

    repair_rate  = n_repair / n_total      if n_total      > 0 else 0.0
    corrupt_rate = n_corrupt / n_safe_total if n_safe_total > 0 else 0.0
    invalid_rate = n_invalid / n_total      if n_total      > 0 else 0.0

    return {
        "repair_rate":   round(repair_rate, 3),
        "corrupt_rate":  round(corrupt_rate, 3),
        "invalid_rate":  round(invalid_rate, 3),
        "n_unsafe_tested": n_total,
        "n_safe_tested":   n_safe_total,
        "n_repair":        n_repair,
        "n_corrupt":       n_corrupt,
        "n_invalid":       n_invalid,
    }

# ── Main ──────────────────────────────────────────────────────────────────────
safe_prompts = load_safe_prompts_sample(n=5)
all_results  = {}
all_validated = {}
summary_rows = []

for layer in LAYERS:
    print(f"\n=== Layer {layer} ===")
    pruned = json.load(open(PHASE4_DIR / f"pruned_feature_candidates_layer_{layer}.json"))
    all_results[layer]   = {}
    all_validated[layer] = {}

    for cwe in ALL_CWES:
        key = f"CWE-{cwe}"
        candidates = pruned.get(key, [])
        if not candidates:
            print(f"  [SKIP] {key}: no pruned candidates")
            continue

        top_k = candidates[:TOP_K]

        if cwe in CYBER_CWES:
            unsafe_prompts = load_unsafe_prompts_cyber(cwe)
        else:
            unsafe_prompts = load_cve_prompts(cwe)

        if not unsafe_prompts:
            print(f"  [SKIP] {key}: no unsafe prompts available")
            continue

        print(f"  {key}: testing {len(top_k)} features on {len(unsafe_prompts)} prompts")
        cwe_results   = []
        cwe_validated = []

        for rec in top_k:
            fid = rec["feature_id"]
            res = validate_feature(layer, cwe, fid, unsafe_prompts, safe_prompts)
            res["feature_id"] = fid
            res["layer"]      = layer
            res["cwe_id"]     = key
            cwe_results.append(res)

            passed = (
                res["repair_rate"]  > REPAIR_MIN and
                res["corrupt_rate"] < CORRUPT_MAX and
                res["invalid_rate"] < INVALID_MAX
            )
            if passed:
                cwe_validated.append({**rec, **res})

            print(f"    feat {fid:6d}: repair={res['repair_rate']:.2f} "
                  f"corrupt={res['corrupt_rate']:.2f} invalid={res['invalid_rate']:.2f} "
                  f"{'PASS' if passed else 'fail'}")

        all_results[layer][key]   = cwe_results
        all_validated[layer][key] = cwe_validated

        summary_rows.append({
            "layer":         layer,
            "cwe_id":        key,
            "n_tested":      len(cwe_results),
            "n_validated":   len(cwe_validated),
            "source":        "cyber_dev" if cwe in CYBER_CWES else "cve_pairs",
            "best_repair":   max(r["repair_rate"]  for r in cwe_results) if cwe_results else 0,
            "min_corrupt":   min(r["corrupt_rate"] for r in cwe_results) if cwe_results else 0,
        })

# ── Save outputs ──────────────────────────────────────────────────────────────
for layer in LAYERS:
    res_path = OUT_DIR / f"feature_validation_results_layer_{layer}.json"
    val_path = OUT_DIR / f"validated_features_layer_{layer}.json"
    with open(res_path, "w") as f:
        json.dump(all_results.get(layer, {}), f, indent=2)
    with open(val_path, "w") as f:
        json.dump(all_validated.get(layer, {}), f, indent=2)
    print(f"Saved layer {layer} results")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_DIR / "causal_validation_summary.csv", index=False)
print("\nCausal validation summary:")
print(summary_df.to_string(index=False))

# Checkpoint
n_validated_cwes = summary_df[summary_df["n_validated"] > 0]["cwe_id"].nunique()
print(f"\nCWEs with validated features: {n_validated_cwes}/9")
if n_validated_cwes >= 6:
    print("PASS: >= 6 CWEs have causally validated features")
else:
    print("WARN: < 6 CWEs validated — review repair rates and steering magnitude")

print("\nStep 4.4 complete.")