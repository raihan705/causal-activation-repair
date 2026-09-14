"""
Phase 5 Step 5.3 - Semantic latent repair B3-ungated
Uses highest-ranked group per CWE (mixed_robust > semantic > statistical_fallback)
Applies all features in group at recommended midpoint alpha.
Output: outputs/phase5/semantic_static_dev_outputs.json
"""

import json
import os
import time
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from sae_lens import SAE

os.environ["HF_HUB_OFFLINE"] = "1"

# --- Paths ---
PROJECT_ROOT  = Path.cwd()
DEV_PROMPTS   = PROJECT_ROOT / "data/cyberseceval/dev_prompts.json"
LIB_PATH      = PROJECT_ROOT / "configs/intervention_library.json"
OUTPUT_DIR    = PROJECT_ROOT / "outputs/phase5"
OUTPUT_FILE   = OUTPUT_DIR / "semantic_static_dev_outputs.json"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Config ---
MODEL_ID    = "meta-llama/Meta-Llama-3.1-8B-Instruct"
SAE_RELEASE = "llama_scope_lxr_8x"
SEED        = 42
TEMPERATURE = 0.2
TOP_P       = 0.95
MAX_NEW_TOK = 512

GROUP_TYPE_PRIORITY = {"mixed_robust": 0, "semantic": 1, "statistical_fallback": 2}


def load_sae(layer):
    sae_id = f"l{layer}r_8x"
    sae, _, _ = SAE.from_pretrained(release=SAE_RELEASE, sae_id=sae_id)
    return sae.to("cuda").eval()


def select_best_groups(lib):
    """Select highest-priority group per CWE."""
    from collections import defaultdict
    by_cwe = defaultdict(list)
    for g in lib["groups"]:
        by_cwe[g["cwe_id"]].append(g)
    best = {}
    for cwe, gs in by_cwe.items():
        gs_sorted = sorted(gs, key=lambda x: GROUP_TYPE_PRIORITY.get(x["group_type"], 3))
        best[cwe] = gs_sorted[0]
    return best


def make_hook(sae, feature_ids, alpha):
    """Hook: encode, steer all features in group, decode back."""
    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        z = sae.encode(hidden)
        for fid in feature_ids:
            z[..., fid] += alpha
        h_steered = sae.decode(z).to(hidden.dtype)
        if isinstance(output, tuple):
            return (h_steered,) + output[1:]
        return h_steered
    return hook_fn


def main():
    torch.manual_seed(SEED)

    prompts = json.loads(DEV_PROMPTS.read_text(encoding="utf-8"))
    print(f"Loaded {len(prompts)} dev prompts")

    lib = json.loads(LIB_PATH.read_text(encoding="utf-8"))
    best_groups = select_best_groups(lib)
    print("Best group per CWE:")
    for cwe, g in best_groups.items():
        print(f"  {cwe} → {g['group_id']} layer={g['layer_id']} "
              f"features={g['feature_ids']} alpha_mid={g['base_alpha']}")

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
    required_layers = set(g["layer_id"] for g in best_groups.values())
    print(f"Loading SAEs for layers: {sorted(required_layers)}")
    saes_by_layer = {}
    for layer in sorted(required_layers):
        print(f"  Loading SAE layer {layer}...")
        saes_by_layer[layer] = load_sae(layer)

    results = []
    t0 = time.time()

    for i, item in enumerate(prompts):
        prompt_id  = item["prompt_id"]
        prompt_txt = item["test_case_prompt"]
        language   = item.get("language", "")
        cwe_id     = item.get("cwe_identifier", "")

        group = best_groups.get(cwe_id)

        inputs = tokenizer(
            prompt_txt, return_tensors="pt", truncation=True, max_length=1024
        ).to(model.device)

        if group is None:
            # No group for this CWE — raw generation
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
                "cwe_id": cwe_id, "steered": False,
                "group_id": None, "alpha": None,
            })
        else:
            layer      = group["layer_id"]
            feature_ids = group["feature_ids"]
            alpha      = group["base_alpha"]  # midpoint = base_alpha=20.0
            sae        = saes_by_layer[layer]

            layer_module = model.model.layers[layer]
            hook = layer_module.register_forward_hook(
                make_hook(sae, feature_ids, alpha)
            )

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
                "group_id": group["group_id"], "alpha": alpha,
                "layer": layer, "feature_ids": feature_ids,
            })

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(prompts)} — {time.time()-t0:.1f}s")

    OUTPUT_FILE.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved {len(results)} → {OUTPUT_FILE}")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()