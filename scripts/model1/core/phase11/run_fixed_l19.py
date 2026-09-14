"""
Phase 11C: Fixed Layer L19 Generation
Uses best validated feature at L19 for each active CWE, alpha=40.
Comparison target: B* (B2_a40) which uses dynamic layer per CWE
(L19 for CWE-120, L23 for CWE-327/89/338).

Fixed L19 feature map (best repair_rate=1.0, corrupt=0 from validated_features_layer_19.json):
  CWE-120: feature 14193 (same as B*)
  CWE-327: feature 16897
  CWE-89:  feature 11462
  CWE-338: feature 151

Output: outputs/phase11/fixed_l19_dev_outputs.json
Scanning: run on Colab after generation.
"""
import json, os, time, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from sae_lens import SAE

os.environ["HF_HUB_OFFLINE"] = "1"

PROJECT_ROOT = Path(os.environ.get("CAR_ROOT", Path(__file__).resolve().parents[4])).resolve()
DEV_PROMPTS  = PROJECT_ROOT / "data/cyberseceval/dev_prompts.json"
OUTPUT_DIR   = PROJECT_ROOT / "outputs/phase11"
OUTPUT_FILE  = OUTPUT_DIR / "fixed_l19_dev_outputs.json"
CKPT_FILE    = OUTPUT_DIR / "fixed_l19_dev_outputs_checkpoint.json"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID    = "meta-llama/Meta-Llama-3.1-8B-Instruct"
SAE_RELEASE = "llama_scope_lxr_8x"
SEED        = 42
TEMPERATURE = 0.2
TOP_P       = 0.95
MAX_NEW_TOK = 512
ALPHA       = 40.0
FIXED_LAYER = 19
CKPT_EVERY  = 50

# Fixed L19 feature map — best validated feature at L19 per active CWE
FIXED_L19_MAP = {
    "CWE-120": {"layer": 19, "feature": 14193},  # repair_rate=1.0 (same as B*)
    "CWE-327": {"layer": 19, "feature": 16897},  # repair_rate=1.0
    "CWE-89":  {"layer": 19, "feature": 11462},  # repair_rate=1.0
    "CWE-338": {"layer": 19, "feature": 151},    # repair_rate=1.0
}


def load_sae(layer):
    sae, _, _ = SAE.from_pretrained(release=SAE_RELEASE, sae_id=f"l{layer}r_8x")
    return sae.to("cuda").eval()


def make_hook(sae, feature_id, alpha):
    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        z = sae.encode(hidden)
        z[..., feature_id] += alpha
        h_steered = sae.decode(z).to(hidden.dtype)
        if isinstance(output, tuple):
            return (h_steered,) + output[1:]
        return h_steered
    return hook_fn


def main():
    torch.manual_seed(SEED)

    prompts = json.loads(DEV_PROMPTS.read_text(encoding="utf-8"))
    print(f"Loaded {len(prompts)} dev prompts")
    print(f"Fixed L19 feature map: {FIXED_L19_MAP}")

    # Checkpoint resume
    results  = []
    done_ids = set()
    if CKPT_FILE.exists():
        results  = json.loads(CKPT_FILE.read_text(encoding="utf-8"))
        done_ids = {r["prompt_id"] for r in results}
        print(f"Resuming: {len(done_ids)} done")

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

    # Only L19 needed
    print(f"Loading SAE layer {FIXED_LAYER}...")
    sae = load_sae(FIXED_LAYER)
    print("SAE loaded.")

    t0 = time.time()
    for i, item in enumerate(remaining):
        prompt_id  = item["prompt_id"]
        prompt_txt = item["test_case_prompt"]
        language   = item.get("language", "")
        cwe_id     = item.get("cwe_identifier", "")

        cfg    = FIXED_L19_MAP.get(cwe_id)
        inputs = tokenizer(
            prompt_txt, return_tensors="pt", truncation=True, max_length=1024
        ).to(model.device)

        if cfg is None:
            # CWE not in active set → raw generation
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
                "layer": None, "feature": None, "alpha": None,
            })
        else:
            hook = model.model.layers[FIXED_LAYER].register_forward_hook(
                make_hook(sae, cfg["feature"], ALPHA)
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
                "layer": FIXED_LAYER, "feature": cfg["feature"], "alpha": ALPHA,
            })

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = (len(remaining) - i - 1) / ((i + 1) / elapsed) / 60
            print(f"  [{i+1}/{len(remaining)}] {elapsed/60:.1f}m  ETA={eta:.1f}m  "
                  f"pid={prompt_id}  cwe={cwe_id}  steered={cfg is not None}")

        if (i + 1) % CKPT_EVERY == 0:
            CKPT_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"  Checkpoint saved ({len(results)} total)")

    OUTPUT_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWritten: {OUTPUT_FILE}")

    steered_n = sum(1 for r in results if r.get("steered"))
    empty_n   = sum(1 for r in results if not r.get("generated_code","").strip())
    print(f"Total={len(results)}  Steered={steered_n}  Empty={empty_n}")

    from collections import Counter
    cwe_s = Counter(r["cwe_id"] for r in results if r.get("steered"))
    print("Steered per CWE:", dict(sorted(cwe_s.items())))
    print("\nNext step: scan fixed_l19_dev_outputs.json on Colab.")


if __name__ == "__main__":
    main()
