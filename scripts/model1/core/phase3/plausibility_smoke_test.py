"""
Phase 3, Step 3.6 — Plausibility-guided decoding smoke test
Tests KL-divergence gating between raw and steered token distributions.
Three conditions per prompt: raw baseline, steered without gate, steered with gate.
Output:
  outputs/phase3/plausibility_smoke_report.txt
  outputs/phase3/plausibility_smoke_metrics.csv
"""

import csv
import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from sae_lens import SAE

# ── constants ──────────────────────────────────────────────────────────────────
INTERVENTION_LAYER  = 19      # use single best layer for plausibility test
SAE_RELEASE         = "llama_scope_lxr_8x"
SAE_ID              = "l19r_8x"
MODEL_ID            = "meta-llama/Meta-Llama-3.1-8B-Instruct"

N_PROMPTS           = 25      # ~half unsafe, half safe
TOP_K_FEATURES      = 5
DELTA               = 2.0
MAX_INPUT_TOKENS    = 256
MAX_NEW_TOKENS      = 32      # short — we only need token-level divergence signal
KL_THRESHOLD        = 0.5     # gate fires if KL(raw || steered) > this
DECAY_FACTOR        = 0.5     # scale steering by this when gate fires

PROJECT_ROOT    = Path(__file__).resolve().parents[2]
BASELINE_JSON   = PROJECT_ROOT / "outputs/phase2/baseline_dev_outputs.json"
UNSAFE_IDS_JSON = PROJECT_ROOT / "outputs/phase2/unsafe_dev_ids_b0.json"
OUT_DIR         = PROJECT_ROOT / "outputs/phase3"
LOG_PATH        = OUT_DIR / "plausibility_smoke_test.log"

# ── logging ────────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"),
              logging.StreamHandler()],
)
log = logging.getLogger(__name__)


# ── data loading ───────────────────────────────────────────────────────────────
def load_prompts(n):
    with open(UNSAFE_IDS_JSON, "r", encoding="utf-8") as f:
        unsafe_ids = set(json.load(f))
    with open(BASELINE_JSON, "r", encoding="utf-8") as f:
        all_outputs = json.load(f)

    unsafe = [x for x in all_outputs if x.get("prompt_id") in unsafe_ids
              and x.get("generated_code", "").strip()]
    safe   = [x for x in all_outputs if x.get("prompt_id") not in unsafe_ids
              and x.get("generated_code", "").strip()]

    n_unsafe = min(n // 2, len(unsafe))
    n_safe   = min(n - n_unsafe, len(safe))
    selected = unsafe[:n_unsafe] + safe[:n_safe]
    log.info(f"Loaded {n_unsafe} unsafe + {n_safe} safe = {len(selected)} prompts")
    return selected, unsafe_ids


# ── hook ───────────────────────────────────────────────────────────────────────
class HookEditor:
    def __init__(self, model, layer_idx):
        self.captured    = None
        self.replacement = None
        self.handle      = model.model.layers[layer_idx].register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        self.captured = hidden.detach().clone()
        if self.replacement is not None:
            if isinstance(out, tuple):
                return (self.replacement,) + out[1:]
            return self.replacement

    def remove(self):
        self.handle.remove()


# ── compute steered hidden for one step ───────────────────────────────────────
def build_steered_hidden(hidden, sae, layer_idx, device, scale=1.0):
    """
    Encode last token, perturb top-K features, decode, reinsert.
    scale: multiplier on DELTA (1.0 = full steering, <1.0 = reduced)
    Returns new_hidden (same shape as hidden) or None if no active features.
    """
    seq_len  = hidden.shape[1]
    lt_idx   = seq_len - 1   # for generation, last position is always the new token
    last_h   = hidden[0, lt_idx, :].unsqueeze(0).to(device=device, dtype=torch.float32)

    z        = sae.encode(last_h).squeeze(0)
    active   = z > 0
    k        = min(TOP_K_FEATURES, int(active.sum().item()))
    if k == 0:
        return None

    topk_idx       = (z * active.float()).topk(k).indices
    z_edited       = z.clone()
    z_edited[topk_idx] += scale * DELTA

    recon          = sae.decode(z_edited.unsqueeze(0))
    new_hidden     = hidden.clone().to(device)
    new_hidden[0, lt_idx, :] = recon[0].to(new_hidden.dtype)
    return new_hidden


# ── get next-token distribution ────────────────────────────────────────────────
def get_next_token_dist(model, input_ids, attention_mask, device,
                        layer_idx=None, sae=None, scale=1.0):
    """
    Returns softmax probability distribution over vocab for the next token.
    If layer_idx and sae provided, applies steered hidden at that layer.
    """
    editor = None
    if layer_idx is not None and sae is not None:
        # we need to inject replacement — do a pre-pass to get hidden
        editor = HookEditor(model, layer_idx)

    enc = {"input_ids": input_ids, "attention_mask": attention_mask}
    with torch.no_grad():
        out = model(**enc)

    if editor is not None:
        hidden = editor.captured
        editor.remove()
        new_hidden = build_steered_hidden(hidden, sae, layer_idx, device, scale)
        if new_hidden is not None:
            editor2             = HookEditor(model, layer_idx)
            editor2.replacement = new_hidden
            with torch.no_grad():
                out = model(**enc)
            editor2.remove()

    logits = out.logits[0, -1, :]
    return F.softmax(logits.float(), dim=-1)


# ── KL divergence ─────────────────────────────────────────────────────────────
def kl_div(p, q):
    """KL(p || q) — p is reference (raw), q is steered."""
    eps = 1e-10
    p   = p.clamp(min=eps)
    q   = q.clamp(min=eps)
    return (p * (p / q).log()).sum().item()


# ── run one prompt through all 3 conditions ───────────────────────────────────
def run_prompt(text, tokenizer, model, sae, layer_idx, device):
    """
    Returns dict with per-condition metrics and gate activation count.
    """
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    ).to(device)

    input_ids      = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    result = {
        "kl_values_no_gate":   [],
        "kl_values_gated":     [],
        "gate_activations":    0,
        "total_steps":         MAX_NEW_TOKENS,
        "raw_valid":           True,
        "steered_valid":       True,
        "gated_valid":         True,
    }

    try:
        # simulate token-by-token generation for MAX_NEW_TOKENS steps
        cur_ids  = input_ids.clone()
        cur_mask = attention_mask.clone()

        for step in range(MAX_NEW_TOKENS):
            # raw distribution
            p_raw = get_next_token_dist(model, cur_ids, cur_mask, device)

            # steered distribution (no gate)
            p_steered = get_next_token_dist(
                model, cur_ids, cur_mask, device, layer_idx, sae, scale=1.0
            )

            kl_val = kl_div(p_raw, p_steered)
            result["kl_values_no_gate"].append(round(kl_val, 4))

            # gated: if KL > threshold, reduce steering
            if kl_val > KL_THRESHOLD:
                result["gate_activations"] += 1
                p_gated = get_next_token_dist(
                    model, cur_ids, cur_mask, device, layer_idx, sae, scale=DECAY_FACTOR
                )
                kl_gated = kl_div(p_raw, p_gated)
            else:
                p_gated  = p_steered
                kl_gated = kl_val
            result["kl_values_gated"].append(round(kl_gated, 4))

            # advance: sample next token from raw distribution (greedy)
            next_token = p_raw.argmax().unsqueeze(0).unsqueeze(0)
            cur_ids    = torch.cat([cur_ids, next_token], dim=1)
            cur_mask   = torch.cat(
                [cur_mask, torch.ones(1, 1, device=device, dtype=cur_mask.dtype)],
                dim=1,
            )

    except Exception as e:
        log.warning(f"  prompt failed: {e}")
        result["steered_valid"] = False
        result["gated_valid"]   = False

    return result


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    sae = SAE.from_pretrained(SAE_RELEASE, SAE_ID)
    sae = sae.to(device)
    sae.eval()

    prompts, unsafe_ids = load_prompts(N_PROMPTS)
    csv_rows            = []

    for i, item in enumerate(prompts):
        text      = item.get("generated_code", "")
        pid       = item.get("prompt_id", "")
        cwe       = item.get("cwe_id", "unknown")
        is_unsafe = pid in unsafe_ids

        log.info(f"[{i+1}/{len(prompts)}] prompt_id={pid} cwe={cwe} unsafe={is_unsafe}")

        res = run_prompt(text, tokenizer, model, sae, INTERVENTION_LAYER, device)

        kl_no_gate = res["kl_values_no_gate"]
        kl_gated   = res["kl_values_gated"]
        n_steps    = len(kl_no_gate)

        mean_kl_no_gate = sum(kl_no_gate) / n_steps if n_steps > 0 else 0
        mean_kl_gated   = sum(kl_gated)   / n_steps if n_steps > 0 else 0
        gate_rate       = res["gate_activations"] / n_steps if n_steps > 0 else 0

        log.info(f"  mean_kl_no_gate={mean_kl_no_gate:.4f}, "
                 f"mean_kl_gated={mean_kl_gated:.4f}, "
                 f"gate_activations={res['gate_activations']}/{n_steps} "
                 f"({gate_rate:.2%})")

        csv_rows.append({
            "prompt_id":          pid,
            "cwe_id":             cwe,
            "is_unsafe":          is_unsafe,
            "n_steps":            n_steps,
            "mean_kl_no_gate":    round(mean_kl_no_gate, 4),
            "mean_kl_gated":      round(mean_kl_gated, 4),
            "gate_activations":   res["gate_activations"],
            "gate_rate":          round(gate_rate, 4),
            "kl_reduction":       round(mean_kl_no_gate - mean_kl_gated, 4),
            "steered_valid":      res["steered_valid"],
            "gated_valid":        res["gated_valid"],
        })

    # ── save CSV ───────────────────────────────────────────────────────────────
    csv_path = OUT_DIR / "plausibility_smoke_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    log.info(f"Saved {csv_path}")

    # ── summary report ─────────────────────────────────────────────────────────
    total_gate   = sum(r["gate_activations"] for r in csv_rows)
    total_steps  = sum(r["n_steps"] for r in csv_rows)
    prompts_with_gate = sum(1 for r in csv_rows if r["gate_activations"] > 0)
    mean_kl_reduction = sum(r["kl_reduction"] for r in csv_rows) / len(csv_rows)
    n_valid_steered   = sum(1 for r in csv_rows if r["steered_valid"])
    n_valid_gated     = sum(1 for r in csv_rows if r["gated_valid"])

    gate_fires     = total_gate > 0
    kl_reduced     = mean_kl_reduction > 0
    no_corruption  = n_valid_gated == len(csv_rows)

    report_lines = [
        "Phase 3 Step 3.6 — Plausibility Smoke Test Report",
        "=" * 50,
        f"Layer tested:            {INTERVENTION_LAYER}",
        f"KL threshold:            {KL_THRESHOLD}",
        f"Decay factor:            {DECAY_FACTOR}",
        f"Total prompts:           {len(csv_rows)}",
        f"Prompts with gate fire:  {prompts_with_gate}/{len(csv_rows)}",
        f"Total gate activations:  {total_gate}/{total_steps} steps "
        f"({total_gate/total_steps:.2%})",
        f"Mean KL no gate:         "
        f"{sum(r['mean_kl_no_gate'] for r in csv_rows)/len(csv_rows):.4f}",
        f"Mean KL gated:           "
        f"{sum(r['mean_kl_gated'] for r in csv_rows)/len(csv_rows):.4f}",
        f"Mean KL reduction:       {mean_kl_reduction:.4f}",
        f"Valid steered outputs:   {n_valid_steered}/{len(csv_rows)}",
        f"Valid gated outputs:     {n_valid_gated}/{len(csv_rows)}",
        "",
        "Checkpoint conditions:",
        f"  Gate fires on some steps:     {gate_fires}",
        f"  KL reduced by gating:         {kl_reduced}",
        f"  No validity failures (gated): {no_corruption}",
        "",
        f"CHECKPOINT: {'PASS' if gate_fires and kl_reduced else 'FAIL'}",
    ]

    report_text = "\n".join(report_lines)
    report_path = OUT_DIR / "plausibility_smoke_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    log.info("\n" + report_text)
    log.info(f"Saved {report_path}")
    log.info("Step 3.6 complete.")


if __name__ == "__main__":
    main()