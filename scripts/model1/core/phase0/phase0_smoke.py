"""
Phase 0 - Step 0.4: Smoke-test generation and ICD scanning.
- Loads 10 prompts from CyberSecEval (2 per language: c, cpp, python, java, javascript)
- Generates code using frozen model
- Runs semgrep scanner on each output
- Verifies all outputs non-empty and scanner-processable
Output: outputs/phase0/phase0_smoke.json
"""

import os
import json
import subprocess
import tempfile
import datetime
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs/phase0")

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LANGUAGES = ["c", "cpp", "python", "java", "javascript"]
SAMPLES_PER_LANG = 2
SEED = 42

LANG_EXT = {
    "c": "c",
    "cpp": "cpp",
    "python": "py",
    "java": "java",
    "javascript": "js",
}


def load_prompts():
    ds = load_dataset("walledai/CyberSecEval", "instruct")
    prompts = []
    for lang in LANGUAGES:
        split = ds[lang]
        for i in range(SAMPLES_PER_LANG):
            row = split[i]
            prompts.append({
                "prompt_id": f"{lang}_{i}",
                "language": lang,
                "cwe_id": row["cwe_identifier"],
                "prompt": row["prompt"],
            })
    print(f"[OK] Loaded {len(prompts)} prompts across {LANGUAGES}")
    return prompts


def load_model():
    print(f"[*] Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    print(f"[OK] Model loaded.")
    return model, tokenizer


def generate_code(model, tokenizer, prompt_text):
    messages = [{"role": "user", "content": prompt_text}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=512,
            temperature=0.2,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def run_semgrep(code, language):
    ext = LANG_EXT.get(language, "txt")
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=f".{ext}", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [
                "semgrep", "scan",
                "--config", "p/security-audit",
                "--json",
                "--quiet",
                tmp_path,
            ],
            capture_output=True,
            timeout=60,
            encoding="utf-8",
            errors="replace",
        )
        scanner_ok = True
        try:
            findings = json.loads(result.stdout)
            finding_count = len(findings.get("results", []))
        except json.JSONDecodeError:
            findings = {}
            finding_count = -1
    except subprocess.TimeoutExpired:
        scanner_ok = False
        finding_count = -1
        findings = {}
    except FileNotFoundError:
        scanner_ok = False
        finding_count = -1
        findings = {"error": "semgrep not found in PATH"}
    finally:
        os.unlink(tmp_path)

    return {
        "scanner_ok": scanner_ok,
        "finding_count": finding_count,
        "findings": findings.get("results", [])[:3],  # keep first 3 only
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    prompts = load_prompts()
    model, tokenizer = load_model()

    results = []
    all_generated = True
    all_scannable = True

    for i, p in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] {p['prompt_id']} | {p['cwe_id']}")

        # Generate
        try:
            code = generate_code(model, tokenizer, p["prompt"])
            generated_ok = len(code.strip()) > 0
            if not generated_ok:
                all_generated = False
            print(f"  Generated: {len(code)} chars | OK={generated_ok}")
        except Exception as e:
            code = ""
            generated_ok = False
            all_generated = False
            print(f"  Generation FAILED: {e}")

        # Scan
        scan = run_semgrep(code, p["language"])
        if not scan["scanner_ok"]:
            all_scannable = False
        print(f"  Scanner: ok={scan['scanner_ok']} | findings={scan['finding_count']}")

        results.append({
            "prompt_id": p["prompt_id"],
            "language": p["language"],
            "cwe_id": p["cwe_id"],
            "generated_code": code,
            "generated_ok": generated_ok,
            "scanner_ok": scan["scanner_ok"],
            "finding_count": scan["finding_count"],
        })

    # Summary
    smoke = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "total_prompts": len(prompts),
        "all_generated": all_generated,
        "all_scannable": all_scannable,
        "checkpoint_pass": all_generated and all_scannable,
        "results": results,
    }

    out_path = os.path.join(OUTPUT_DIR, "phase0_smoke.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(smoke, f, indent=2, ensure_ascii=False)

    print(f"\n=== Step 0.4 Summary ===")
    print(f"  Total prompts    : {len(prompts)}")
    print(f"  All generated    : {all_generated}")
    print(f"  All scannable    : {all_scannable}")
    print(f"  Checkpoint PASS  : {smoke['checkpoint_pass']}")
    print(f"\n[{'OK' if smoke['checkpoint_pass'] else 'FAIL'}] phase0_smoke.json -> {out_path}")


if __name__ == "__main__":
    main()