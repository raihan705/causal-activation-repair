"""
Phase 5 Step 5.4 - Semantic latent repair with plausibility gate (B3-PG)
Same groups as B3-ungated. At each token step:
  - if KL(p_raw || p_steered) > KL_THRESH: scale alpha by DECAY
  - if still > KL_THRESH after decay: disable steering for that step
Output: outputs/phase5/semantic_static_pg_dev_outputs.json
        outputs/phase5/semantic_static_pg_dev_plausibility_log.json
"""

import json
import os
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from sae_lens import SAE

os.environ["HF_HUB_OFFLINE"] = "1"

# --- Paths ---
PROJECT_ROOT  = Path.cwd()
DEV_PROMPTS   = PROJECT_ROOT / "data/cyberseceval/dev_prompts.json"
LIB_PATH      = PROJECT_ROOT / "configs/intervention_library.json"
OUTPUT_DIR    = PROJECT_ROOT / "outputs/phase5"
OUTPUT_FILE   = OUTPUT_DIR / "semantic_static_pg_dev_outputs.json"
LOG_FILE      = OUTPUT_DIR / "semantic_static_pg_dev_plausibility_log.json"
CKPT_FILE     = OUTPUT_DIR / "semantic_static_pg_dev_outputs_ckpt.json"
CKPT_LOG      = OUTPUT_DIR / "semantic_static_pg_dev_plausibility_log_ckpt.json"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Config ---
MODEL_ID    = "meta-llama/Meta-Llama-3.1-8B-Instruct"
SAE_RELEASE = "llama_scope_lxr_8x"
SEED        = 42
TEMPERATURE = 0.2
TOP_P       = 0.95
MAX_NEW_TOK = 512
KL_THRESH   = 0.5
DECAY       = 0.5

GROUP_TYPE_PRIORITY = {"mixed_robust": 0, "semantic": 1, "statistical_fallback": 2}


def load_sae(layer):
    sae_id = f"l{layer}r_8x"
    sae, _, _ = SAE.from_pretrained(release=SAE_RELEASE, sae_id=sae_id)
    return sae.to("cuda").eval()


def select_best_groups(lib):
    from collections import defaultdict
    by_cwe = defaultdict(list)
    for g in lib["groups"]:
        by_cwe[g["cwe_id"]].append(g)
    best = {}
    for cwe, gs in by_cwe.items():
        gs_sorted = sorted(gs, key=lambda x: GROUP_TYPE_PRIORITY.get(x["group_type"], 3))
        best[cwe] = gs_sorted[0]
    return best


def kl_div(p, q):
    """KL(p || q) — both are logit tensors, converted to probs."""
    p_prob = F.softmax(p, dim=-1).clamp(min=1e-9)
    q_prob = F.softmax(q, dim=-1).clamp(min=1e-9)
    return (p_prob * (p_prob.log() - q_prob.log())).sum().item()


def generate_with_pg(model, tokenizer, input_ids, sae, layer_id,
                     feature_ids, base_alpha, kl_thresh, decay, max_new_tokens):
    """
    Token-by-token generation with plausibility-gated steering.
    At each step: compute raw logits, steered logits, check KL, decide alpha.
    """
    gate_log = []
    generated = input_ids.clone()

    for step in range(max_new_tokens):
        # --- Raw forward pass (no hook) ---
        with torch.no_grad():
            raw_out = model(generated)
        torch.cuda.synchronize()
        raw_logits = raw_out.logits[:, -1, :]  # (1, vocab)

        # --- Steered forward pass ---
        alpha_used = base_alpha
        steered_logits = None
        kl = 0.0

        for attempt in range(2):  # try base_alpha, then decayed

            def make_hook_pg(sae_, fids, alp):
                def hook_fn(module, input, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    z = sae_.encode(hidden)
                    for fid in fids:
                        z[..., fid] += alp
                    h_st = sae_.decode(z).to(hidden.dtype)
                    if isinstance(output, tuple):
                        return (h_st,) + output[1:]
                    return h_st
                return hook_fn

            layer_module = model.model.layers[layer_id]
            hook = layer_module.register_forward_hook(
                make_hook_pg(sae, feature_ids, alpha_used)
            )
            with torch.no_grad():
                st_out = model(generated)
            torch.cuda.synchronize()
            hook.remove()
            steered_logits = st_out.logits[:, -1, :]

            kl = kl_div(raw_logits[0], steered_logits[0])

            if kl <= kl_thresh:
                break  # accept this alpha
            elif attempt == 0:
                alpha_used = base_alpha * decay  # try decayed
            else:
                # still too high — disable steering
                steered_logits = raw_logits
                alpha_used = 0.0

        gate_log.append({"step": step, "kl": round(kl, 4), "alpha_used": alpha_used})

        # Sample next token from steered distribution
        logits_for_sample = steered_logits[0] / TEMPERATURE
        sorted_logits, sorted_idx = torch.sort(logits_for_sample, descending=True)
        cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cumprobs - F.softmax(sorted_logits, dim=-1) > TOP_P
        sorted_logits[remove] = float('-inf')
        probs = F.softmax(sorted_logits, dim=-1)
        next_token_idx = torch.multinomial(probs, num_samples=1)
        next_token = sorted_idx[next_token_idx].unsqueeze(0)

        generated = torch.cat([generated, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    new_tokens = generated[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True), gate_log


def main():
    torch.manual_seed(SEED)

    prompts = json.loads(DEV_PROMPTS.read_text(encoding="utf-8"))
    print(f"Loaded {len(prompts)} dev prompts")

    lib = json.loads(LIB_PATH.read_text(encoding="utf-8"))
    best_groups = select_best_groups(lib)

    # --- Resume from checkpoint if exists ---
    if CKPT_FILE.exists():
        results = json.loads(CKPT_FILE.read_text(encoding="utf-8"))
        plausibility_logs = json.loads(CKPT_LOG.read_text(encoding="utf-8"))
        done_ids = {r["prompt_id"] for r in results}
        prompts = [p for p in prompts if p["prompt_id"] not in done_ids]
        print(f"Resuming from checkpoint: {len(done_ids)} done, {len(prompts)} remaining")
    else:
        results = []
        plausibility_logs = []
        print("Starting fresh")

    if len(prompts) == 0:
        print("All prompts already completed.")
        OUTPUT_FILE.write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        LOG_FILE.write_text(
            json.dumps(plausibility_logs, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map="auto"
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    required_layers = set(g["layer_id"] for g in best_groups.values())
    print(f"Loading SAEs for layers: {sorted(required_layers)}")
    saes_by_layer = {}
    for layer in sorted(required_layers):
        print(f"  Loading SAE layer {layer}...")
        saes_by_layer[layer] = load_sae(layer)

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
            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOK,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            torch.cuda.synchronize()
            new_tokens = gen[0][inputs["input_ids"].shape[1]:]
            generated_code = tokenizer.decode(new_tokens, skip_special_tokens=True)
            gate_log = []
            steered = False
        else:
            try:
                generated_code, gate_log = generate_with_pg(
                    model, tokenizer,
                    inputs["input_ids"],
                    saes_by_layer[group["layer_id"]],
                    group["layer_id"],
                    group["feature_ids"],
                    group["base_alpha"],
                    KL_THRESH, DECAY,
                    MAX_NEW_TOK,
                )
                steered = True
            except Exception as e:
                print(f"  ERROR prompt {prompt_id}: {e}")
                generated_code = ""
                gate_log = []
                steered = False

        n_gated = sum(1 for s in gate_log if s["alpha_used"] < group["base_alpha"]) if (gate_log and group) else 0

        results.append({
            "prompt_id": prompt_id, "prompt_text": prompt_txt,
            "generated_code": generated_code, "language": language,
            "cwe_id": cwe_id, "steered": steered,
            "group_id": group["group_id"] if group else None,
            "n_steps": len(gate_log), "n_gated": n_gated,
        })
        plausibility_logs.append({
            "prompt_id": prompt_id, "cwe_id": cwe_id,
            "n_steps": len(gate_log), "n_gated": n_gated,
            "gate_log": gate_log,
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(prompts)} — {elapsed:.1f}s")
            CKPT_FILE.write_text(
                json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            CKPT_LOG.write_text(
                json.dumps(plausibility_logs, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    # Final save
    OUTPUT_FILE.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    LOG_FILE.write_text(
        json.dumps(plausibility_logs, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved {len(results)} outputs → {OUTPUT_FILE}")
    print(f"Saved plausibility log → {LOG_FILE}")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()