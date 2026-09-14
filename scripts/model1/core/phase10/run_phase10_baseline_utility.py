"""
Phase 10 Step 10.1 — Baseline utility evaluation (B0)
Runs raw Llama-3.1-8B-Instruct (no steering) on:
  - InstructHumanEval (164 prompts)
  - BigCodeBench      (1066 prompts)
  - MMLU code slices  (high_school_computer_science, college_computer_science,
                       computer_security, machine_learning)

Outputs saved to outputs/phase10/
Checkpoint-resume enabled (every 50 prompts) for BigCodeBench.
"""

import os, json, sys, time
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path.cwd()
OUTPUT_DIR = ROOT / "outputs" / "phase10"
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

# ── helpers ────────────────────────────────────────────────────────────────────
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


def generate(tok, mdl, prompt: str, max_new_tokens: int = 512) -> str:
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
    with torch.no_grad():
        out = mdl.generate(
            **inputs,
            temperature=GEN_KWARGS["temperature"],
            top_p=GEN_KWARGS["top_p"],
            max_new_tokens=max_new_tokens,
            do_sample=GEN_KWARGS["do_sample"],
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


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
def run_humaneval(tok, mdl):
    out_path  = OUTPUT_DIR / "utility_baseline_humaneval.json"
    ckpt_path = OUTPUT_DIR / "_ckpt_baseline_humaneval.json"

    if out_path.exists():
        print("InstructHumanEval baseline already exists, skipping.")
        return

    # load HumanEval — needs HF dataset or human_eval package
    os.environ.pop("HF_HUB_OFFLINE", None)   # may need download
    try:
        from datasets import load_dataset
        ds = load_dataset("openai_humaneval", split="test", trust_remote_code=True)
        problems = [{"task_id": r["task_id"], "prompt": r["prompt"]} for r in ds]
    except Exception as e:
        print(f"  Could not load via datasets: {e}")
        try:
            from human_eval.data import read_problems
            raw = read_problems()
            problems = [{"task_id": k, "prompt": v["prompt"]} for k, v in raw.items()]
        except Exception as e2:
            print(f"  Could not load via human_eval package either: {e2}")
            sys.exit(1)
    os.environ["HF_HUB_OFFLINE"] = "1"

    results = load_checkpoint(ckpt_path)
    done_ids = {r["task_id"] for r in results}
    remaining = [p for p in problems if p["task_id"] not in done_ids]
    print(f"InstructHumanEval: {len(problems)} total, {len(remaining)} remaining")

    system_msg = (
        "You are an expert Python programmer. Complete the following function. "
        "Return only the completed function body, no explanation."
    )

    for i, prob in enumerate(tqdm(remaining, desc="HumanEval-B0")):
        prompt_text = chat_prompt(tok, system_msg, prob["prompt"])
        completion  = generate(tok, mdl, prompt_text, max_new_tokens=512)
        results.append({
            "task_id":    prob["task_id"],
            "prompt":     prob["prompt"],
            "completion": completion,
        })
        if (i + 1) % 50 == 0:
            save_checkpoint(ckpt_path, results)

    save_checkpoint(out_path, results)
    if ckpt_path.exists():
        ckpt_path.unlink()
    print(f"  Saved {len(results)} records → {out_path}")


# ── BigCodeBench ───────────────────────────────────────────────────────────────
def run_bigcodebench(tok, mdl):
    out_path  = OUTPUT_DIR / "utility_baseline_bigcodebench.json"
    ckpt_path = OUTPUT_DIR / "_ckpt_baseline_bigcodebench.json"

    if out_path.exists():
        print("BigCodeBench baseline already exists, skipping.")
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

    results = load_checkpoint(ckpt_path)
    done_ids = {r["task_id"] for r in results}

    problems = []
    for r in ds:
        task_id = r.get("task_id", r.get("id", str(len(problems))))
        instruct_prompt = r.get("instruct_prompt", r.get("prompt", ""))
        problems.append({"task_id": task_id, "prompt": instruct_prompt})

    remaining = [p for p in problems if p["task_id"] not in done_ids]
    print(f"BigCodeBench: {len(problems)} total, {len(remaining)} remaining")

    system_msg = (
        "You are an expert programmer. Complete the following coding task. "
        "Return only the code, no explanation."
    )

    for i, prob in enumerate(tqdm(remaining, desc="BigCodeBench-B0")):
        prompt_text = chat_prompt(tok, system_msg, prob["prompt"])
        completion  = generate(tok, mdl, prompt_text, max_new_tokens=512)
        results.append({
            "task_id":    prob["task_id"],
            "prompt":     prob["prompt"],
            "completion": completion,
        })
        if (i + 1) % 50 == 0:
            save_checkpoint(ckpt_path, results)

    save_checkpoint(out_path, results)
    if ckpt_path.exists():
        ckpt_path.unlink()
    print(f"  Saved {len(results)} records → {out_path}")


# ── MMLU ───────────────────────────────────────────────────────────────────────
def run_mmlu(tok, mdl):
    out_path = OUTPUT_DIR / "utility_baseline_mmlu.json"

    if out_path.exists():
        print("MMLU baseline already exists, skipping.")
        return

    os.environ.pop("HF_HUB_OFFLINE", None)
    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets not available, skipping MMLU")
        return

    CHOICE_LABELS = ["A", "B", "C", "D"]
    all_results = []

    for slice_name in MMLU_SLICES:
        print(f"  MMLU slice: {slice_name}")
        try:
            ds = load_dataset("cais/mmlu", slice_name, split="test", trust_remote_code=True)
        except Exception as e:
            print(f"    Could not load {slice_name}: {e}, skipping.")
            continue

        for i, row in enumerate(tqdm(ds, desc=f"MMLU-{slice_name}-B0")):
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
            response    = generate(tok, mdl, prompt_text, max_new_tokens=8)

            # extract predicted letter
            pred = "X"
            for ch in response.strip().upper():
                if ch in CHOICE_LABELS:
                    pred = ch
                    break

            all_results.append({
                "slice":     slice_name,
                "question":  question,
                "answer":    answer,
                "predicted": pred,
                "correct":   pred == answer,
                "response":  response.strip(),
            })

    os.environ["HF_HUB_OFFLINE"] = "1"

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    total   = len(all_results)
    correct = sum(r["correct"] for r in all_results)
    print(f"  MMLU saved {total} records, accuracy={correct/total:.4f} → {out_path}")


# ── main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tok, mdl = load_model()

    print("\n=== InstructHumanEval ===")
    run_humaneval(tok, mdl)

    print("\n=== BigCodeBench ===")
    run_bigcodebench(tok, mdl)

    print("\n=== MMLU ===")
    run_mmlu(tok, mdl)

    print("\nPhase 10 Step 10.1 complete.")
    print(f"Outputs in: {OUTPUT_DIR}")