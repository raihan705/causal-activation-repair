"""
Phase 11B: Type-A Statistical Groups Generation
Steers dev prompts using ONLY statistical_fallback groups from intervention_library.json.
Group selection: statistical_fallback only (vs mixed_robust > semantic > fallback in B3).
Alpha: base_alpha=20.0 (same as B3-ungated).

Active Type-A CWEs (with B0 detections on dev):
  CWE-327: L23, [14449], alpha=20
  CWE-89:  L23, [4657, 22548, 1652], alpha=20
  CWE-338: L23, [7533], alpha=20
  CWE-120: NO Type-A group → raw generation (key difference vs B*)

Output: outputs/phase11/typeA_dev_outputs.json
Scanning: run on Colab after generation (Semgrep unreliable on Windows).
"""

import json, os, time, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from sae_lens import SAE

os.environ["HF_HUB_OFFLINE"] = "1"

PROJECT_ROOT = Path(os.environ.get("CAR_ROOT", Path(__file__).resolve().parents[4])).resolve()
DEV_PROMPTS  = PROJECT_ROOT / "data/cyberseceval/dev_prompts.json"
LIB_PATH     = PROJECT_ROOT / "configs/intervention_library.json"
OUTPUT_DIR   = PROJECT_ROOT / "outputs/phase11"
OUTPUT_FILE  = OUTPUT_DIR / "typeA_dev_outputs.json"
CKPT_FILE    = OUTPUT_DIR / "typeA_dev_outputs_checkpoint.json"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID    = "meta-llama/Meta-Llama-3.1-8B-Instruct"
SAE_RELEASE = "llama_scope_lxr_8x"
SEED        = 42
TEMPERATURE = 0.2
TOP_P       = 0.95
MAX_NEW_TOK = 512
CKPT_EVERY  = 50


def load_sae(layer):
    sae, _, _ = SAE.from_pretrained(release=SAE_RELEASE, sae_id=f"l{layer}r_8x")
    return sae.to("cuda").eval()


def select_typeA_groups(lib):
    """Select statistical_fallback group per CWE. One per CWE (first found)."""
    groups = {}
    for g in lib["groups"]:
        if g["group_type"] == "statistical_fallback" and g["cwe_id"] not in groups:
            groups[g["cwe_id"]] = g
    return groups


def make_hook(sae, feature_ids, alpha):
    """Encode hidden state, add alpha to each feature, decode back in-place."""
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
    type_a_groups = select_typeA_groups(lib)
    print("Type-A groups selected:")
    for cwe, g in sorted(type_a_groups.items()):
        print(f"  {cwe}: layer={g['layer_id']} features={g['feature_ids']} alpha={g['base_alpha']}")

    # Checkpoint resume
    results  = []
    done_ids = set()
    if CKPT_FILE.exists():
        results  = json.loads(CKPT_FILE.read_text(encoding="utf-8"))
        done_ids = {r["prompt_id"] for r in results}
        print(f"Resuming from checkpoint: {len(done_ids)} done")

    remaining = [p for p in prompts if p["prompt_id"] not in done_ids]
    print(f"Remaining: {len(remaining)}")

    if not remaining:
        OUTPUT_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"All done. Written: {OUTPUT_FILE}")
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

    # Load SAEs for required layers only
    required_layers = set(g["layer_id"] for g in type_a_groups.values())
    print(f"Loading SAEs for layers: {sorted(required_layers)}")
    saes_by_layer = {layer: load_sae(layer) for layer in sorted(required_layers)}

    t0 = time.time()
    for i, item in enumerate(remaining):
        prompt_id  = item["prompt_id"]
        prompt_txt = item["test_case_prompt"]
        language   = item.get("language", "")
        cwe_id     = item.get("cwe_identifier", "")

        group  = type_a_groups.get(cwe_id)
        inputs = tokenizer(
            prompt_txt, return_tensors="pt", truncation=True, max_length=1024
        ).to(model.device)

        if group is None:
            # No Type-A group for this CWE → raw generation
            with torch.no_grad():
                gen = model.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOK,
                    temperature=TEMPERATURE, top_p=TOP_P,
                    do_sample=True, pad_token_id=tokenizer.pad_token_id,
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
            layer       = group["layer_id"]
            feature_ids = group["feature_ids"]
            alpha       = group["base_alpha"]
            sae         = saes_by_layer[layer]

            hook = model.model.layers[layer].register_forward_hook(
                make_hook(sae, feature_ids, alpha)
            )
            try:
                with torch.no_grad():
                    gen = model.generate(
                        **inputs, max_new_tokens=MAX_NEW_TOK,
                        temperature=TEMPERATURE, top_p=TOP_P,
                        do_sample=True, pad_token_id=tokenizer.pad_token_id,
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

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = (len(remaining) - i - 1) / ((i + 1) / elapsed) / 60
            print(f"  [{i+1}/{len(remaining)}] {elapsed/60:.1f}m elapsed  ETA={eta:.1f}m  "
                  f"pid={prompt_id}  cwe={cwe_id}  steered={group is not None}")

        if (i + 1) % CKPT_EVERY == 0:
            CKPT_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"  Checkpoint saved ({len(results)} total)")

    OUTPUT_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWritten: {OUTPUT_FILE}")

    steered_n = sum(1 for r in results if r.get("steered"))
    empty_n   = sum(1 for r in results if not r.get("generated_code", "").strip())
    print(f"Total={len(results)}  Steered={steered_n}  Empty={empty_n}")
    from collections import Counter
    cwe_s = Counter(r["cwe_id"] for r in results if r.get("steered"))
    print("Steered per CWE:", dict(sorted(cwe_s.items())))
    print("\nNext step: scan typeA_dev_outputs.json on Colab, then run compare_typeA_vs_bstar.py")


if __name__ == "__main__":
    main()
