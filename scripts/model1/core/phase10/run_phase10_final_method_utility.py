"""
Phase 10 Step 10.2 — Final method utility evaluation (BSTAR)
Runs BSTAR (B2_a40, alpha=40.0) on:
  - InstructHumanEval (164 prompts)
  - BigCodeBench      (1140 prompts)
  - MMLU code slices  (4 slices)

Since these benchmarks have no CWE labels, no CWE will match the
feature_map → steering never activates → output is identical to B0 by design.
This is the expected and documented behavior (zero utility overhead).

Outputs saved to outputs/phase10/
"""

import os, json, sys
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path.cwd()
OUTPUT_DIR  = ROOT / "outputs" / "phase10"
CONFIG_PATH = ROOT / "configs" / "bstar_config.json"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── model ──────────────────────────────────────────────────────────────────────
os.environ["HF_HUB_OFFLINE"] = "1"
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

GEN_KWARGS = dict(temperature=0.2, top_p=0.95, max_new_tokens=512, do_sample=True)

MMLU_SLICES = [
    "high_school_computer_science",
    "college_computer_science",
    "computer_security",
    "machine_learning",
]

# ── load BSTAR config ──────────────────────────────────────────────────────────
def load_bstar_config():
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    print(f"Loaded BSTAR config: {len(cfg['feature_map'])} CWEs, alpha={cfg['alpha']}")
    return cfg

# ── load SAE ───────────────────────────────────────────────────────────────────
def load_sae(layer: int):
    from sae_lens import SAE
    sae, _, _ = SAE.from_pretrained(
        release="llama_scope_lxr_8x",
        sae_id=f"l{layer}r_8x",
    )
    sae = sae.to(DEVICE)
    sae.eval()
    return sae

# ── model load ─────────────────────────────────────────────────────────────────
def load_model():
    print("Loading tokenizer and model...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    mdl.eval()
    print(f"Model loaded on {DEVICE}")
    return tok, mdl

# ── BSTAR steering hook ────────────────────────────────────────────────────────
def make_hook(sae, feature_id: int, alpha: float):
    """Additive SAE feature steering hook."""
    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        # encode
        with torch.no_grad():
            latents = sae.encode(hidden.to(sae.W_enc.dtype))
            latents[:, :, feature_id] += alpha
            steered = sae.decode(latents).to(hidden.dtype)
        if isinstance(output, tuple):
            return (steered,) + output[1:]
        return steered
    return hook_fn

# ── generate with optional BSTAR steering ─────────────────────────────────────
def generate_bstar(tok, mdl, saes_by_layer, prompt: str,
                   cwe: str, bstar_cfg: dict, max_new_tokens: int = 512) -> dict:
    """
    If cwe is in feature_map, apply steering; otherwise raw generation.
    Returns dict with completion and steered flag.
    """
    feature_map = bstar_cfg["feature_map"]
    alpha       = bstar_cfg["alpha"]

    hooks   = []
    steered = False

    if cwe and cwe in feature_map:
        entry   = feature_map[cwe]
        layer   = entry["layer"]
        feat_id = entry["feature"]
        if layer in saes_by_layer:
            sae  = saes_by_layer[layer]
            hook = mdl.model.layers[layer].register_forward_hook(
                make_hook(sae, feat_id, alpha)
            )
            hooks.append(hook)
            steered = True

    try:
        inputs = tok(prompt, return_tensors="pt",
                     truncation=True, max_length=2048).to(DEVICE)
        with torch.no_grad():
            out = mdl.generate(
                **inputs,
                temperature=GEN_KWARGS["temperature"],
                top_p=GEN_KWARGS["top_p"],
                max_new_tokens=max_new_tokens,
                do_sample=GEN_KWARGS["do_sample"],
                pad_token_id=tok.eos_token_id,
            )
        completion = tok.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
    finally:
        for h in hooks:
            h.remove()

    return {"completion": completion, "steered": steered}


def chat_prompt(tok, system: str, user: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_checkpoint(path: Path) -> list:
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        print(f"  Resumed from checkpoint: {len(data)} records")
        return data
    return []


def save_checkpoint(path: Path, data: list):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# ── InstructHumanEval ──────────────────────────────────────────────────────────
def run_humaneval(tok, mdl, saes_by_layer, bstar_cfg):
    out_path  = OUTPUT_DIR / "utility_finalmethod_humaneval.json"
    ckpt_path = OUTPUT_DIR / "_ckpt_finalmethod_humaneval.json"

    if out_path.exists():
        print("HumanEval BSTAR already exists, skipping.")
        return

    os.environ.pop("HF_HUB_OFFLINE", None)
    try:
        from datasets import load_dataset
        ds = load_dataset("openai_humaneval", split="test")
        problems = [{"task_id": r["task_id"], "prompt": r["prompt"]} for r in ds]
    except Exception as e:
        print(f"  Could not load via datasets: {e}")
        try:
            from human_eval.data import read_problems
            raw = read_problems()
            problems = [{"task_id": k, "prompt": v["prompt"]} for k, v in raw.items()]
        except Exception as e2:
            print(f"  Could not load via human_eval package: {e2}")
            sys.exit(1)
    os.environ["HF_HUB_OFFLINE"] = "1"

    results   = load_checkpoint(ckpt_path)
    done_ids  = {r["task_id"] for r in results}
    remaining = [p for p in problems if p["task_id"] not in done_ids]
    print(f"HumanEval BSTAR: {len(problems)} total, {len(remaining)} remaining")

    system_msg = (
        "You are an expert Python programmer. Complete the following function. "
        "Return only the completed function body, no explanation."
    )

    # No CWE for HumanEval → steered will always be False
    for i, prob in enumerate(tqdm(remaining, desc="HumanEval-BSTAR")):
        prompt_text = chat_prompt(tok, system_msg, prob["prompt"])
        result      = generate_bstar(tok, mdl, saes_by_layer,
                                     prompt_text, cwe=None,
                                     bstar_cfg=bstar_cfg)
        results.append({
            "task_id":    prob["task_id"],
            "prompt":     prob["prompt"],
            "completion": result["completion"],
            "steered":    result["steered"],
        })
        if (i + 1) % 50 == 0:
            save_checkpoint(ckpt_path, results)

    save_checkpoint(out_path, results)
    if ckpt_path.exists():
        ckpt_path.unlink()
    steered_count = sum(1 for r in results if r["steered"])
    print(f"  Saved {len(results)} records, steered={steered_count} → {out_path}")


# ── BigCodeBench ───────────────────────────────────────────────────────────────
def run_bigcodebench(tok, mdl, saes_by_layer, bstar_cfg):
    out_path  = OUTPUT_DIR / "utility_finalmethod_bigcodebench.json"
    ckpt_path = OUTPUT_DIR / "_ckpt_finalmethod_bigcodebench.json"

    if out_path.exists():
        print("BigCodeBench BSTAR already exists, skipping.")
        return

    os.environ.pop("HF_HUB_OFFLINE", None)
    try:
        from datasets import load_dataset
        ds = load_dataset("bigcode/bigcodebench", split="v0.1.2")
        print(f"  Loaded BigCodeBench v0.1.2: {len(ds)} tasks")
    except Exception as e:
        print(f"  Failed to load BigCodeBench: {e}")
        sys.exit(1)
    os.environ["HF_HUB_OFFLINE"] = "1"

    results  = load_checkpoint(ckpt_path)
    done_ids = {r["task_id"] for r in results}

    problems = []
    for r in ds:
        task_id         = r.get("task_id", r.get("id", str(len(problems))))
        instruct_prompt = r.get("instruct_prompt", r.get("prompt", ""))
        problems.append({"task_id": task_id, "prompt": instruct_prompt})

    remaining = [p for p in problems if p["task_id"] not in done_ids]
    print(f"BigCodeBench BSTAR: {len(problems)} total, {len(remaining)} remaining")

    system_msg = (
        "You are an expert programmer. Complete the following coding task. "
        "Return only the code, no explanation."
    )

    for i, prob in enumerate(tqdm(remaining, desc="BigCodeBench-BSTAR")):
        prompt_text = chat_prompt(tok, system_msg, prob["prompt"])
        result      = generate_bstar(tok, mdl, saes_by_layer,
                                     prompt_text, cwe=None,
                                     bstar_cfg=bstar_cfg)
        results.append({
            "task_id":    prob["task_id"],
            "prompt":     prob["prompt"],
            "completion": result["completion"],
            "steered":    result["steered"],
        })
        if (i + 1) % 50 == 0:
            save_checkpoint(ckpt_path, results)

    save_checkpoint(out_path, results)
    if ckpt_path.exists():
        ckpt_path.unlink()
    steered_count = sum(1 for r in results if r["steered"])
    print(f"  Saved {len(results)} records, steered={steered_count} → {out_path}")


# ── MMLU ───────────────────────────────────────────────────────────────────────
def run_mmlu(tok, mdl, saes_by_layer, bstar_cfg):
    out_path = OUTPUT_DIR / "utility_finalmethod_mmlu.json"

    if out_path.exists():
        print("MMLU BSTAR already exists, skipping.")
        return

    os.environ.pop("HF_HUB_OFFLINE", None)
    from datasets import load_dataset

    CHOICE_LABELS = ["A", "B", "C", "D"]
    all_results   = []

    for slice_name in MMLU_SLICES:
        print(f"  MMLU slice: {slice_name}")
        try:
            ds = load_dataset("cais/mmlu", slice_name, split="test")
        except Exception as e:
            print(f"    Could not load {slice_name}: {e}, skipping.")
            continue

        for row in tqdm(ds, desc=f"MMLU-{slice_name}-BSTAR"):
            question = row["question"]
            choices  = row["choices"]
            answer   = CHOICE_LABELS[row["answer"]]

            options_str = "\n".join(
                f"{CHOICE_LABELS[j]}. {choices[j]}" for j in range(len(choices))
            )
            user_msg = (
                f"Question: {question}\n\n"
                f"{options_str}\n\n"
                "Answer with only the letter (A, B, C, or D)."
            )
            prompt_text = chat_prompt(tok, "You are a knowledgeable assistant.", user_msg)
            result      = generate_bstar(tok, mdl, saes_by_layer,
                                         prompt_text, cwe=None,
                                         bstar_cfg=bstar_cfg,
                                         max_new_tokens=8)
            response = result["completion"].strip()
            pred = "X"
            for ch in response.upper():
                if ch in CHOICE_LABELS:
                    pred = ch
                    break

            all_results.append({
                "slice":     slice_name,
                "question":  question,
                "answer":    answer,
                "predicted": pred,
                "correct":   pred == answer,
                "response":  response,
                "steered":   result["steered"],
            })

    os.environ["HF_HUB_OFFLINE"] = "1"

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    total         = len(all_results)
    correct       = sum(r["correct"] for r in all_results)
    steered_count = sum(r["steered"] for r in all_results)
    print(f"  MMLU saved {total} records, accuracy={correct/total:.4f}, steered={steered_count} → {out_path}")


# ── main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bstar_cfg = load_bstar_config()

    # load SAEs for all layers used in feature_map
    used_layers = sorted({v["layer"] for v in bstar_cfg["feature_map"].values()})
    print(f"Loading SAEs for layers: {used_layers}")
    saes_by_layer = {}
    for layer in used_layers:
        print(f"  Loading SAE layer {layer}...")
        saes_by_layer[layer] = load_sae(layer)

    tok, mdl = load_model()

    print("\n=== InstructHumanEval ===")
    run_humaneval(tok, mdl, saes_by_layer, bstar_cfg)

    print("\n=== BigCodeBench ===")
    run_bigcodebench(tok, mdl, saes_by_layer, bstar_cfg)

    print("\n=== MMLU ===")
    run_mmlu(tok, mdl, saes_by_layer, bstar_cfg)

    print("\nPhase 10 Step 10.2 complete.")
    print(f"Outputs in: {OUTPUT_DIR}")