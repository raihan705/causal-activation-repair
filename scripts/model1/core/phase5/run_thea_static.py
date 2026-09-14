"""
Phase 5 Step 5.2 - Thea-style static single-feature repair (B2)
Runs alpha=20.0 and alpha=40.0, saves both outputs.
Output: outputs/phase5/thea_static_dev_outputs_a20.json
        outputs/phase5/thea_static_dev_outputs_a40.json
"""

import json
import os
import time
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from sae_lens import SAE

os.environ["HF_HUB_OFFLINE"] = "1"

# --- Paths ---
PROJECT_ROOT = Path.cwd()
DEV_PROMPTS  = PROJECT_ROOT / "data/cyberseceval/dev_prompts.json"
OUTPUT_DIR   = PROJECT_ROOT / "outputs/phase5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Config ---
MODEL_ID    = "meta-llama/Meta-Llama-3.1-8B-Instruct"
SAE_RELEASE = "llama_scope_lxr_8x"
SEED        = 42
TEMPERATURE = 0.2
TOP_P       = 0.95
MAX_NEW_TOK = 512

# B2: top single feature per CWE and its layer
# Layer 19 preferred; fallback to best available layer
CWE_FEATURE_MAP = {
    "CWE-120": {"layer": 19, "feature": 14193},
    "CWE-787": {"layer": 19, "feature": 1515},
    "CWE-190": {"layer": 19, "feature": 16897},
    "CWE-327": {"layer": 23, "feature": 14449},
    "CWE-89":  {"layer": 23, "feature": 1652},
    "CWE-338": {"layer": 23, "feature": 7533},
    "CWE-79":  {"layer": 16, "feature": 9816},
    "CWE-125": {"layer": 23, "feature": 16655},
    "CWE-476": {"layer": 23, "feature": 18397},
}

ALPHAS = [20.0, 40.0]


def load_sae(layer):
    sae_id = f"l{layer}r_8x"
    sae, _, _ = SAE.from_pretrained(release=SAE_RELEASE, sae_id=sae_id)
    sae = sae.to("cuda").eval()
    return sae


def make_hook(sae, feature_id, alpha, activation_store):
    """Hook: encode activation, add alpha to feature, decode back."""
    def hook_fn(module, input, output):
        # output may be tuple
        hidden = output[0] if isinstance(output, tuple) else output
        # last token pooling for edit, but edit full sequence
        z = sae.encode(hidden)
        # steering direction: unit vector along feature dimension
        direction = torch.zeros_like(z)
        direction[..., feature_id] = 1.0
        z_steered = z + alpha * direction
        h_steered = sae.decode(z_steered).to(hidden.dtype)
        activation_store['fired'] = activation_store.get('fired', 0) + 1
        if isinstance(output, tuple):
            return (h_steered,) + output[1:]
        return h_steered
    return hook_fn


def run_b2(prompts, model, tokenizer, saes_by_layer, alpha, output_path):
    results = []
    t0 = time.time()

    for i, item in enumerate(prompts):
        prompt_id  = item["prompt_id"]
        prompt_txt = item["test_case_prompt"]
        language   = item.get("language", "")
        cwe_id     = item.get("cwe_identifier", "")

        cwe_cfg = CWE_FEATURE_MAP.get(cwe_id)

        if cwe_cfg is None:
            # No intervention for unknown/held-out CWEs — use raw generation
            inputs = tokenizer(
                prompt_txt, return_tensors="pt", truncation=True, max_length=1024
            ).to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOK,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            new_tokens = gen[0][inputs["input_ids"].shape[1]:]
            generated_code = tokenizer.decode(new_tokens, skip_special_tokens=True)
            results.append({
                "prompt_id": prompt_id, "prompt_text": prompt_txt,
                "generated_code": generated_code, "language": language,
                "cwe_id": cwe_id, "steered": False, "alpha": None,
            })
        else:
            layer      = cwe_cfg["layer"]
            feature_id = cwe_cfg["feature"]
            sae        = saes_by_layer[layer]

            store = {}
            # Register hook at the correct transformer layer
            layer_module = model.model.layers[layer]
            hook = layer_module.register_forward_hook(
                make_hook(sae, feature_id, alpha, store)
            )

            inputs = tokenizer(
                prompt_txt, return_tensors="pt", truncation=True, max_length=1024
            ).to(model.device)

            try:
                with torch.no_grad():
                    gen = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOK,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                new_tokens = gen[0][inputs["input_ids"].shape[1]:]
                generated_code = tokenizer.decode(new_tokens, skip_special_tokens=True)
                steered = True
            except Exception as e:
                print(f"  ERROR prompt {prompt_id}: {e}")
                generated_code = ""
                steered = False
            finally:
                hook.remove()

            results.append({
                "prompt_id": prompt_id, "prompt_text": prompt_txt,
                "generated_code": generated_code, "language": language,
                "cwe_id": cwe_id, "steered": steered,
                "alpha": alpha, "layer": layer, "feature": feature_id,
            })

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(prompts)} — {time.time()-t0:.1f}s")

    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved {len(results)} → {output_path}")


def main():
    torch.manual_seed(SEED)

    prompts = json.loads(DEV_PROMPTS.read_text(encoding="utf-8"))
    print(f"Loaded {len(prompts)} dev prompts")

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load SAEs for required layers only
    required_layers = set(v["layer"] for v in CWE_FEATURE_MAP.values())
    print(f"Loading SAEs for layers: {sorted(required_layers)}")
    saes_by_layer = {}
    for layer in sorted(required_layers):
        print(f"  Loading SAE layer {layer}...")
        saes_by_layer[layer] = load_sae(layer)

    # Run for each alpha
    for alpha in ALPHAS:
        print(f"\n=== Running B2 alpha={alpha} ===")
        suffix = f"a{int(alpha)}"
        output_path = OUTPUT_DIR / f"thea_static_dev_outputs_{suffix}.json"
        if output_path.exists():
            print(f"  Already exists, skipping: {output_path}")
            continue
        run_b2(prompts, model, tokenizer, saes_by_layer, alpha, output_path)

    print("\nB2 complete.")


if __name__ == "__main__":
    main()