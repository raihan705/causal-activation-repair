"""
Phase 9 Step 9.2 — Generate outputs for all methods on held-out test prompts.

Methods:
    B0    : raw baseline (no steering, no prefix)
    B1    : secure zero-shot prefix
    B2    : Thea-style single-feature steering (alpha=40.0)
    B3    : semantic latent repair with plausibility gate
    BSTAR : B2_a40 (final method, identical to B2 at alpha=40)

Usage:
    python run_phase9_generation.py --method B0
    python run_phase9_generation.py --method B1
    python run_phase9_generation.py --method B2
    python run_phase9_generation.py --method B3
    python run_phase9_generation.py --method BSTAR

Run one method at a time. Each method supports resume from checkpoint.
Run ONLY after test_split_verification.txt shows PASS.
"""

import argparse
import json
import os
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from sae_lens import SAE

os.environ["HF_HUB_OFFLINE"] = "1"

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path.cwd()
TEST_PROMPTS = PROJECT_ROOT / "data/cyberseceval/test_prompts.json"
CONFIG_DIR   = PROJECT_ROOT / "configs"
OUTPUT_DIR   = PROJECT_ROOT / "outputs/phase9"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Frozen generation settings ─────────────────────────────────────────────────
MODEL_ID    = "meta-llama/Meta-Llama-3.1-8B-Instruct"
SAE_RELEASE = "llama_scope_lxr_8x"
SEED        = 42
TEMPERATURE = 0.2
TOP_P       = 0.95
MAX_NEW_TOK = 512
MAX_CTX_LEN = 1024

# ── B1 prefix (frozen from Phase 5) ───────────────────────────────────────────
SECURITY_PREFIX = (
    "You are a secure coding assistant. "
    "Generate code that avoids known vulnerabilities such as buffer overflows, "
    "SQL injection, and insecure cryptographic practices.\n\n"
)

# ── B3-PG settings (frozen from Phase 5) ──────────────────────────────────────
KL_THRESH = 0.5
DECAY     = 0.5

GROUP_TYPE_PRIORITY = {"mixed_robust": 0, "semantic": 1, "statistical_fallback": 2}

# ── Output file map ────────────────────────────────────────────────────────────
OUTPUT_FILES = {
    "B0":    "baseline_test_outputs.json",
    "B1":    "zeroshot_test_outputs.json",
    "B2":    "thea_static_test_outputs.json",
    "B3":    "semantic_static_test_outputs.json",
    "BSTAR": "bstar_test_outputs.json",
}
CKPT_FILES = {
    m: OUTPUT_DIR / v.replace(".json", "_ckpt.json")
    for m, v in OUTPUT_FILES.items()
}

# ── Verify split check passed ──────────────────────────────────────────────────
verify_path = OUTPUT_DIR / "test_split_verification.txt"
assert verify_path.exists(), "Run verify_heldout_split.py first."
assert "PASS — held-out split is clean" in verify_path.read_text(), \
    "Split verification did not pass. Aborting."


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_sae(layer):
    sae, _, _ = SAE.from_pretrained(
        release=SAE_RELEASE, sae_id=f"l{layer}r_8x"
    )
    return sae.to("cuda").eval()


def kl_div(p_logits, q_logits):
    p = F.softmax(p_logits, dim=-1).clamp(min=1e-9)
    q = F.softmax(q_logits, dim=-1).clamp(min=1e-9)
    return (p * (p.log() - q.log())).sum().item()


def make_b2_hook(sae, feature_id, alpha):
    """Full-sequence steering hook (B2 / BSTAR)."""
    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        z = sae.encode(hidden)
        z[..., feature_id] += alpha
        h_steered = sae.decode(z).to(hidden.dtype)
        if isinstance(output, tuple):
            return (h_steered,) + output[1:]
        return h_steered
    return hook_fn


def select_best_groups(lib):
    """Return {cwe_id: best_group} using same priority as Phase 5."""
    from collections import defaultdict
    by_cwe = defaultdict(list)
    for g in lib["groups"]:
        by_cwe[g["cwe_id"]].append(g)
    best = {}
    for cwe, gs in by_cwe.items():
        gs_sorted = sorted(
            gs, key=lambda x: GROUP_TYPE_PRIORITY.get(x["group_type"], 3)
        )
        best[cwe] = gs_sorted[0]
    return best


def generate_with_pg(model, tokenizer, input_ids, sae, layer_id,
                     feature_ids, base_alpha):
    """Token-by-token generation with plausibility gate (exact Phase 5 logic)."""
    generated = input_ids.clone()
    gate_log  = []

    for step in range(MAX_NEW_TOK):
        # Raw forward pass
        with torch.no_grad():
            raw_out = model(generated)
        raw_logits = raw_out.logits[:, -1, :]

        # Steered forward pass — try base_alpha, then decayed, then disable
        alpha_used     = base_alpha
        steered_logits = None
        kl             = 0.0

        for attempt in range(2):
            def make_pg_hook(sae_, fids, alp):
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

            hook = model.model.layers[layer_id].register_forward_hook(
                make_pg_hook(sae, feature_ids, alpha_used)
            )
            with torch.no_grad():
                st_out = model(generated)
            hook.remove()
            steered_logits = st_out.logits[:, -1, :]
            kl = kl_div(raw_logits[0], steered_logits[0])

            if kl <= KL_THRESH:
                break
            elif attempt == 0:
                alpha_used = base_alpha * DECAY
            else:
                steered_logits = raw_logits
                alpha_used     = 0.0

        gate_log.append({
            "step": step, "kl": round(kl, 4), "alpha_used": alpha_used
        })

        # Sample with top-p (exact Phase 5 logic)
        logits_s = steered_logits[0] / TEMPERATURE
        sorted_logits, sorted_idx = torch.sort(logits_s, descending=True)
        cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cumprobs - F.softmax(sorted_logits, dim=-1) > TOP_P
        sorted_logits[remove] = float("-inf")
        probs = F.softmax(sorted_logits, dim=-1)
        next_tok_idx = torch.multinomial(probs, num_samples=1)
        next_tok     = sorted_idx[next_tok_idx].unsqueeze(0)

        generated = torch.cat([generated, next_tok], dim=1)
        if next_tok.item() == tokenizer.eos_token_id:
            break

    new_tokens = generated[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True), gate_log


# ══════════════════════════════════════════════════════════════════════════════
# Per-method runners
# ══════════════════════════════════════════════════════════════════════════════

def run_b0(prompts, model, tokenizer):
    results = []
    for i, item in enumerate(prompts):
        inputs = tokenizer(
            item["test_case_prompt"],
            return_tensors="pt", truncation=True, max_length=MAX_CTX_LEN
        ).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOK,
                temperature=TEMPERATURE, top_p=TOP_P, do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = gen[0][inputs["input_ids"].shape[1]:]
        results.append({
            "prompt_id":      item["prompt_id"],
            "prompt_text":    item["test_case_prompt"],
            "generated_code": tokenizer.decode(new_tokens, skip_special_tokens=True),
            "language":       item.get("language", ""),
            "cwe_id":         item.get("cwe_identifier", ""),
            "method":         "B0",
        })
        if (i + 1) % 50 == 0:
            print(f"  [B0] {i+1}/{len(prompts)}")
    return results


def run_b1(prompts, model, tokenizer):
    results = []
    for i, item in enumerate(prompts):
        full_prompt = SECURITY_PREFIX + item["test_case_prompt"]
        inputs = tokenizer(
            full_prompt,
            return_tensors="pt", truncation=True, max_length=MAX_CTX_LEN
        ).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOK,
                temperature=TEMPERATURE, top_p=TOP_P, do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = gen[0][inputs["input_ids"].shape[1]:]
        results.append({
            "prompt_id":      item["prompt_id"],
            "prompt_text":    item["test_case_prompt"],
            "generated_code": tokenizer.decode(new_tokens, skip_special_tokens=True),
            "language":       item.get("language", ""),
            "cwe_id":         item.get("cwe_identifier", ""),
            "method":         "B1",
        })
        if (i + 1) % 50 == 0:
            print(f"  [B1] {i+1}/{len(prompts)}")
    return results


def run_b2_or_bstar(prompts, model, tokenizer, saes_by_layer,
                    feature_map, alpha, method_label):
    results = []
    for i, item in enumerate(prompts):
        cwe_id     = item.get("cwe_identifier", "")
        prompt_txt = item["test_case_prompt"]
        cwe_cfg    = feature_map.get(cwe_id)

        inputs = tokenizer(
            prompt_txt,
            return_tensors="pt", truncation=True, max_length=MAX_CTX_LEN
        ).to(model.device)

        if cwe_cfg is None:
            # No matching CWE — raw generation (same fallback as Phase 5)
            with torch.no_grad():
                gen = model.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOK,
                    temperature=TEMPERATURE, top_p=TOP_P, do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            new_tokens     = gen[0][inputs["input_ids"].shape[1]:]
            generated_code = tokenizer.decode(new_tokens, skip_special_tokens=True)
            steered        = False
            layer_used     = None
            feature_used   = None
        else:
            layer      = cwe_cfg["layer"]
            feature_id = cwe_cfg["feature"]
            sae        = saes_by_layer[layer]
            hook = model.model.layers[layer].register_forward_hook(
                make_b2_hook(sae, feature_id, alpha)
            )
            try:
                with torch.no_grad():
                    gen = model.generate(
                        **inputs, max_new_tokens=MAX_NEW_TOK,
                        temperature=TEMPERATURE, top_p=TOP_P, do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                new_tokens     = gen[0][inputs["input_ids"].shape[1]:]
                generated_code = tokenizer.decode(new_tokens, skip_special_tokens=True)
                steered        = True
                layer_used     = layer
                feature_used   = feature_id
            except Exception as e:
                print(f"  ERROR {item['prompt_id']}: {e}")
                generated_code = ""
                steered        = False
                layer_used     = None
                feature_used   = None
            finally:
                hook.remove()

        results.append({
            "prompt_id":      item["prompt_id"],
            "prompt_text":    prompt_txt,
            "generated_code": generated_code,
            "language":       item.get("language", ""),
            "cwe_id":         cwe_id,
            "method":         method_label,
            "steered":        steered,
            "alpha":          alpha if steered else None,
            "layer":          layer_used,
            "feature":        feature_used,
        })
        if (i + 1) % 50 == 0:
            print(f"  [{method_label}] {i+1}/{len(prompts)}")
    return results


def run_b3(prompts, model, tokenizer, saes_by_layer, best_groups):
    results = []
    for i, item in enumerate(prompts):
        cwe_id     = item.get("cwe_identifier", "")
        prompt_txt = item["test_case_prompt"]
        group      = best_groups.get(cwe_id)

        inputs = tokenizer(
            prompt_txt,
            return_tensors="pt", truncation=True, max_length=MAX_CTX_LEN
        ).to(model.device)

        if group is None:
            with torch.no_grad():
                gen = model.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOK,
                    temperature=TEMPERATURE, top_p=TOP_P, do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            new_tokens     = gen[0][inputs["input_ids"].shape[1]:]
            generated_code = tokenizer.decode(new_tokens, skip_special_tokens=True)
            gate_log       = []
            steered        = False
        else:
            try:
                generated_code, gate_log = generate_with_pg(
                    model, tokenizer,
                    inputs["input_ids"],
                    saes_by_layer[group["layer_id"]],
                    group["layer_id"],
                    group["feature_ids"],
                    group["base_alpha"],
                )
                steered = True
            except Exception as e:
                print(f"  ERROR {item['prompt_id']}: {e}")
                generated_code = ""
                gate_log       = []
                steered        = False

        n_gated = sum(
            1 for s in gate_log
            if s["alpha_used"] < (group["base_alpha"] if group else 0)
        )
        results.append({
            "prompt_id":      item["prompt_id"],
            "prompt_text":    prompt_txt,
            "generated_code": generated_code,
            "language":       item.get("language", ""),
            "cwe_id":         cwe_id,
            "method":         "B3",
            "steered":        steered,
            "group_id":       group["group_id"] if group else None,
            "n_steps":        len(gate_log),
            "n_gated":        n_gated,
        })
        if (i + 1) % 50 == 0:
            print(f"  [B3] {i+1}/{len(prompts)}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True,
                        choices=["B0", "B1", "B2", "B3", "BSTAR"])
    args   = parser.parse_args()
    method = args.method

    output_path = OUTPUT_DIR / OUTPUT_FILES[method]
    ckpt_path   = CKPT_FILES[method]

    if output_path.exists():
        print(f"[{method}] Output already exists: {output_path.name}")
        print("Delete it to regenerate.")
        return

    torch.manual_seed(SEED)

    # Load test prompts
    prompts = json.loads(TEST_PROMPTS.read_text(encoding="utf-8"))
    print(f"Loaded {len(prompts)} test prompts.")

    # Resume from checkpoint
    if ckpt_path.exists():
        done      = json.loads(ckpt_path.read_text(encoding="utf-8"))
        done_ids  = {r["prompt_id"] for r in done}
        prompts_todo = [p for p in prompts if p["prompt_id"] not in done_ids]
        print(f"Resuming: {len(done_ids)} done, {len(prompts_todo)} remaining.")
    else:
        done         = []
        prompts_todo = prompts

    if not prompts_todo:
        print("All prompts already done. Writing final output.")
        output_path.write_text(
            json.dumps(done, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print("Model loaded.")

    # Load SAEs and configs if needed
    saes_by_layer = {}
    feature_map   = None
    alpha         = None
    best_groups   = None

    if method in ("B2", "BSTAR"):
        bstar_cfg   = json.loads((CONFIG_DIR / "bstar_config.json").read_text())
        feature_map = bstar_cfg["feature_map"]
        alpha       = bstar_cfg["alpha"]
        req_layers  = set(v["layer"] for v in feature_map.values())
        print(f"Loading SAEs for layers: {sorted(req_layers)}")
        for layer in sorted(req_layers):
            print(f"  Layer {layer}...")
            saes_by_layer[layer] = load_sae(layer)

    elif method == "B3":
        lib         = json.loads((CONFIG_DIR / "intervention_library.json").read_text())
        best_groups = select_best_groups(lib)
        req_layers  = set(g["layer_id"] for g in best_groups.values())
        print(f"Loading SAEs for layers: {sorted(req_layers)}")
        for layer in sorted(req_layers):
            print(f"  Layer {layer}...")
            saes_by_layer[layer] = load_sae(layer)

    # Run generation
    t0 = time.time()
    print(f"\n=== [{method}] generating {len(prompts_todo)} prompts ===")

    if method == "B0":
        batch_results = run_b0(prompts_todo, model, tokenizer)
    elif method == "B1":
        batch_results = run_b1(prompts_todo, model, tokenizer)
    elif method in ("B2", "BSTAR"):
        batch_results = run_b2_or_bstar(
            prompts_todo, model, tokenizer,
            saes_by_layer, feature_map, alpha, method
        )
    elif method == "B3":
        batch_results = run_b3(
            prompts_todo, model, tokenizer, saes_by_layer, best_groups
        )

    all_results = done + batch_results

    # Save checkpoint and final output
    ckpt_path.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    output_path.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    elapsed = time.time() - t0
    print(f"\n[{method}] Done. {len(all_results)} outputs → {output_path.name}")
    print(f"Total time: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()