"""
Phase 3, Step 3.4 — Layer-wise security sensitivity analysis
For each viable layer {16, 19, 23}, measures how sensitive the model's
next-token logits for security-relevant tokens are to small SAE latent
perturbations on unsafe dev samples.

Protocol per layer:
  1. Load unsafe dev samples from Phase 2
  2. Forward pass -> capture activation at layer l
  3. SAE encode -> latent z
  4. Select top-K active features
  5. Perturb each feature: z[j] += delta
  6. SAE decode -> perturbed hidden state
  7. Reinsert -> measure logit shift for security-relevant tokens

Output:
  outputs/phase3/layer_sensitivity_metrics.csv
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
VIABLE_LAYERS      = [16, 19, 23]
SAE_RELEASE        = "llama_scope_lxr_8x"
SAE_ID_TEMPLATE    = "l{l}r_8x"
MODEL_ID           = "meta-llama/Meta-Llama-3.1-8B-Instruct"

N_SAMPLES          = 50      # unsafe dev samples to test per layer
TOP_K_FEATURES     = 10      # top active features to perturb per sample
DELTA              = 2.0     # perturbation magnitude (additive to latent)
MAX_INPUT_TOKENS   = 512

# Security-relevant token strings — covers key vulnerability patterns
# across CWE-120, 327, 89, 338, 79
SECURITY_TOKEN_STRINGS = [
    # memory
    "strcpy", "strcat", "sprintf", "gets", "memcpy", "malloc", "free",
    # crypto weak
    "MD5", "SHA1", "md5", "sha1", "DES", "RC4",
    # rand / prng
    "rand", "random", "srand", "seed",
    # sql injection
    "SELECT", "INSERT", "UPDATE", "DELETE", "WHERE", "execute",
    # xss
    "innerHTML", "document", "eval", "script",
    # exec
    "exec", "system", "popen", "subprocess",
    # safe alternatives (negative sensitivity target)
    "SHA256", "bcrypt", "prepared", "parameterized",
]

PROJECT_ROOT    = Path(__file__).resolve().parents[2]
BASELINE_JSON   = PROJECT_ROOT / "outputs/phase2/baseline_dev_outputs.json"
UNSAFE_IDS_JSON = PROJECT_ROOT / "outputs/phase2/unsafe_dev_ids_b0.json"
OUT_DIR         = PROJECT_ROOT / "outputs/phase3"
LOG_PATH        = OUT_DIR / "layer_sensitivity.log"

# ── logging ────────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"),
              logging.StreamHandler()],
)
log = logging.getLogger(__name__)


# ── load unsafe dev samples ────────────────────────────────────────────────────
def load_unsafe_dev_samples(n):
    with open(UNSAFE_IDS_JSON, "r", encoding="utf-8") as f:
        unsafe_ids = set(json.load(f))
    with open(BASELINE_JSON, "r", encoding="utf-8") as f:
        all_outputs = json.load(f)

    samples = [
        item for item in all_outputs
        if item.get("prompt_id") in unsafe_ids
        and item.get("generated_code", "").strip()
    ]
    if len(samples) > n:
        samples = samples[:n]
    log.info(f"Loaded {len(samples)} unsafe dev samples for sensitivity analysis")
    return samples


# ── build security token ID set ────────────────────────────────────────────────
def build_security_token_ids(tokenizer):
    token_ids = set()
    for s in SECURITY_TOKEN_STRINGS:
        ids = tokenizer.encode(s, add_special_tokens=False)
        token_ids.update(ids)
        # also try with leading space (common for subword tokenizers)
        ids2 = tokenizer.encode(" " + s, add_special_tokens=False)
        token_ids.update(ids2)
    log.info(f"Security token vocabulary: {len(token_ids)} token IDs")
    return token_ids


# ── hook: capture + replace activation at one layer ───────────────────────────
class ActivationEditor:
    """
    Registers a forward hook at a given layer.
    On forward pass, captures the hidden state.
    If self.replacement is set, injects it instead.
    """
    def __init__(self, model, layer_idx):
        self.captured    = None
        self.replacement = None
        self.handle      = model.model.layers[layer_idx].register_forward_hook(
            self._hook
        )

    def _hook(self, module, inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        self.captured = hidden.detach()
        if self.replacement is not None:
            if isinstance(out, tuple):
                return (self.replacement,) + out[1:]
            return self.replacement

    def remove(self):
        self.handle.remove()


# ── last-token extract ─────────────────────────────────────────────────────────
def get_last_token(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """hidden: (1, T, D), mask: (1, T) -> (1, D)"""
    last_idx = attention_mask.sum(dim=1) - 1
    last_idx = last_idx.clamp(min=0)
    return hidden[:, last_idx[0], :]   # (1, D)


# ── sensitivity for one sample at one layer ────────────────────────────────────
def measure_sample_sensitivity(
    text, tokenizer, model, sae, layer_idx, security_token_ids, device
):
    """
    Returns mean absolute logit shift for security tokens
    averaged over top-K feature perturbations.
    """
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    ).to(device)

    # ── baseline forward pass ──────────────────────────────────────────────────
    editor = ActivationEditor(model, layer_idx)
    with torch.no_grad():
        out_base  = model(**enc)
    logits_base   = out_base.logits[0, -1, :]   # (vocab,) last position
    hidden_raw    = editor.captured              # (1, T, D)
    mask          = enc["attention_mask"]
    last_hidden   = get_last_token(hidden_raw, mask.cpu())  # (1, D)
    editor.remove()

    # ── SAE encode ────────────────────────────────────────────────────────────
    z = sae.encode(last_hidden.to(device=device, dtype=torch.float32))  # (1, d_sae)
    z = z.squeeze(0)   # (d_sae,)

    # ── select top-K active features ──────────────────────────────────────────
    active_mask = z > 0
    active_vals = z * active_mask.float()
    k           = min(TOP_K_FEATURES, int(active_mask.sum().item()))
    if k == 0:
        return None   # no active features — skip

    topk_indices = active_vals.topk(k).indices   # (k,)

    # ── perturb each feature and measure logit shift ──────────────────────────
    shifts = []
    for feat_idx in topk_indices:
        z_perturbed       = z.clone()
        z_perturbed[feat_idx] += DELTA

        # decode perturbed latent -> hidden
        recon_perturbed = sae.decode(z_perturbed.unsqueeze(0))  # (1, D)

        # build full hidden state with replacement at last token position
        last_pos    = int(mask.sum().item()) - 1
        new_hidden  = hidden_raw.clone().to(device)
        new_hidden[0, last_pos, :] = recon_perturbed[0].to(new_hidden.dtype)

        # steered forward pass
        editor2             = ActivationEditor(model, layer_idx)
        editor2.replacement = new_hidden
        with torch.no_grad():
            out_steered = model(**enc)
        editor2.remove()

        logits_steered = out_steered.logits[0, -1, :]   # (vocab,)

        # logit shift for security tokens only
        sec_ids = list(security_token_ids)
        shift   = (logits_steered[sec_ids] - logits_base[sec_ids]).abs().mean().item()
        shifts.append(shift)

    return sum(shifts) / len(shifts) if shifts else None


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    log.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    security_token_ids = build_security_token_ids(tokenizer)
    samples            = load_unsafe_dev_samples(N_SAMPLES)

    results = []   # list of dicts for CSV

    for l in VIABLE_LAYERS:
        sae_id = SAE_ID_TEMPLATE.format(l=l)
        log.info(f"=== Layer {l} | SAE: {sae_id} ===")

        sae = SAE.from_pretrained(SAE_RELEASE, sae_id)
        sae = sae.to(device)
        sae.eval()

        layer_shifts = []
        n_skipped    = 0

        for i, item in enumerate(samples):
            text = item.get("generated_code", "")
            cwe  = item.get("cwe_id", "unknown")
            try:
                shift = measure_sample_sensitivity(
                    text, tokenizer, model, sae, l, security_token_ids, device
                )
                if shift is None:
                    n_skipped += 1
                    continue
                layer_shifts.append(shift)
                if i % 10 == 0:
                    log.info(f"  [{l}] sample {i}/{len(samples)} "
                             f"shift={shift:.4f} cwe={cwe}")
            except Exception as e:
                log.warning(f"  [{l}] sample {i} failed: {e}")
                n_skipped += 1

        if layer_shifts:
            mean_shift   = sum(layer_shifts) / len(layer_shifts)
            max_shift    = max(layer_shifts)
            non_trivial  = mean_shift > 0.01   # threshold for non-trivial sensitivity
            log.info(f"  Layer {l}: mean_shift={mean_shift:.4f}, "
                     f"max_shift={max_shift:.4f}, "
                     f"n_valid={len(layer_shifts)}, n_skipped={n_skipped}, "
                     f"non_trivial={non_trivial}")
            results.append({
                "layer":        l,
                "sae_id":       sae_id,
                "n_samples":    len(layer_shifts),
                "n_skipped":    n_skipped,
                "mean_logit_shift": round(mean_shift, 6),
                "max_logit_shift":  round(max_shift, 6),
                "non_trivial":  non_trivial,
            })
        else:
            log.warning(f"  Layer {l}: no valid samples — sensitivity unmeasured")
            results.append({
                "layer": l, "sae_id": sae_id,
                "n_samples": 0, "n_skipped": n_skipped,
                "mean_logit_shift": None, "max_logit_shift": None,
                "non_trivial": False,
            })

        del sae
        torch.cuda.empty_cache()

    # ── save CSV ───────────────────────────────────────────────────────────────
    out_path = OUT_DIR / "layer_sensitivity_metrics.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    log.info(f"Saved {out_path}")

    # ── checkpoint summary ─────────────────────────────────────────────────────
    log.info("=== Checkpoint assessment ===")
    sensitive_layers = [r["layer"] for r in results if r["non_trivial"]]
    log.info(f"  Non-trivial sensitivity layers: {sensitive_layers}")
    if sensitive_layers:
        log.info("  CHECKPOINT PASS: >= 1 layer shows non-trivial security sensitivity.")
    else:
        log.warning("  CHECKPOINT FAIL: no layer shows non-trivial sensitivity.")

    log.info("Step 3.4 complete.")


if __name__ == "__main__":
    main()