"""
Phase 3, Step 3.1 (revised) — Multi-layer activation extraction
Uses LAST-TOKEN pooling to stay on the SAE training distribution.
SAELens SAEs for decoder-only models are trained on last-token activations.
"""

import json
import logging
import random
from pathlib import Path

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── constants ──────────────────────────────────────────────────────────────────
SEED                = 42
CANDIDATE_LAYERS    = [12, 16, 19, 23]
MAX_SAMPLES_PER_CWE = 500
MAX_INPUT_TOKENS    = 512
BATCH_SIZE          = 4
MODEL_ID            = "meta-llama/Meta-Llama-3.1-8B-Instruct"

DERIVATION_CWES = [
    "CWE-120", "CWE-125", "CWE-787", "CWE-190",
    "CWE-476", "CWE-89",  "CWE-79",  "CWE-327",
]
EXPLORATORY_CWES = ["CWE-338"]
TARGET_CWES      = DERIVATION_CWES + EXPLORATORY_CWES

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
TRAIN_CSV     = PROJECT_ROOT / "data/processed/train_pairs.csv"
VAL_CSV       = PROJECT_ROOT / "data/processed/val_pairs.csv"
BASELINE_JSON = PROJECT_ROOT / "outputs/phase2/baseline_dev_outputs.json"
OUT_ROOT      = PROJECT_ROOT / "data/activations/raw"
LOG_PATH      = PROJECT_ROOT / "outputs/phase3/extract_activations.log"

# ── logging ────────────────────────────────────────────────────────────────────
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"),
              logging.StreamHandler()],
)
log = logging.getLogger(__name__)

random.seed(SEED)
torch.manual_seed(SEED)


# ── hook utilities ─────────────────────────────────────────────────────────────
def setup_hooks(model, layers):
    activations = {l: [] for l in layers}
    handles     = []

    def make_hook(l):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            activations[l].append(h.detach().cpu())
        return hook

    for l in layers:
        handle = model.model.layers[l].register_forward_hook(make_hook(l))
        handles.append(handle)
    return activations, handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


# ── last-token pooling ─────────────────────────────────────────────────────────
def last_token_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Extract the activation at the last real (non-padding) token position.
    hidden:           (B, T, D)
    attention_mask:   (B, T)  — 1 for real tokens, 0 for padding
    Returns:          (B, D)
    This matches the SAE training convention for decoder-only models.
    """
    last_idx = attention_mask.sum(dim=1) - 1          # (B,)
    last_idx = last_idx.clamp(min=0)
    batch_size, _, d = hidden.shape
    idx = last_idx.view(batch_size, 1, 1).expand(batch_size, 1, d)
    return hidden.gather(dim=1, index=idx).squeeze(1)  # (B, D)


# ── single batch encode ────────────────────────────────────────────────────────
def encode_batch(texts, tokenizer, model, layers, device):
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    ).to(device)

    act_store, handles = setup_hooks(model, layers)
    with torch.no_grad():
        model(**enc)
    remove_hooks(handles)

    pooled = {}
    for l in layers:
        raw       = act_store[l][0]              # (B, T, D)
        mask      = enc["attention_mask"].cpu()
        pooled[l] = last_token_pool(raw, mask)   # (B, D)
    return pooled


# ── process full list ──────────────────────────────────────────────────────────
def process_texts(texts, metas, tokenizer, model, layers, device, desc=""):
    all_pooled  = {l: [] for l in layers}
    valid_metas = []
    n           = len(texts)

    for start in range(0, n, BATCH_SIZE):
        batch_texts = texts[start : start + BATCH_SIZE]
        batch_metas = metas[start : start + BATCH_SIZE]
        try:
            pooled = encode_batch(batch_texts, tokenizer, model, layers, device)
            for l in layers:
                all_pooled[l].append(pooled[l])
            valid_metas.extend(batch_metas)
        except Exception as e:
            log.warning(f"[{desc}] batch starting at {start} failed: {e}")

        if (start // BATCH_SIZE) % 50 == 0:
            log.info(f"[{desc}] processed {start}/{n}")

    result = {}
    for l in layers:
        if all_pooled[l]:
            result[l] = torch.cat(all_pooled[l], dim=0)
        else:
            result[l] = torch.empty(0)
            log.warning(f"[{desc}] layer {l}: no activations collected")
    return result, valid_metas


# ── save ───────────────────────────────────────────────────────────────────────
def save_activations(tensors_per_layer, metas, out_root, prefix):
    for l, tensor in tensors_per_layer.items():
        layer_dir = out_root / f"layer_{l}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        out_path  = layer_dir / f"{prefix}.pt"
        torch.save({"activations": tensor, "metadata": metas}, out_path)
        log.info(f"  saved {out_path} shape={tuple(tensor.shape)}, {len(metas)} samples")


# ── data loaders ──────────────────────────────────────────────────────────────
def load_cve_pairs():
    dfs = []
    for p in [TRAIN_CSV, VAL_CSV]:
        if p.exists():
            dfs.append(pd.read_csv(p))
        else:
            log.warning(f"File not found: {p}")
    df = pd.concat(dfs, ignore_index=True)
    log.info(f"Loaded {len(df)} CVE pairs (train + val)")
    return df


def sample_cve_by_cwe(df, cwe_list, max_per_cwe):
    sampled = []
    for cwe in cwe_list:
        subset = df[df["cwe_id"] == cwe]
        if len(subset) == 0:
            log.info(f"  CWE {cwe}: 0 rows — skipped")
            continue
        if len(subset) > max_per_cwe:
            subset = subset.sample(max_per_cwe, random_state=SEED)
        log.info(f"  CWE {cwe}: {len(subset)} rows sampled")
        sampled.append(subset)
    return pd.concat(sampled, ignore_index=True) if sampled else pd.DataFrame()


def load_cyberseceval_dev():
    with open(BASELINE_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    log.info(f"Loaded {len(data)} CyberSecEval dev outputs from Phase 2")
    return data


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    log.info("Pooling strategy: LAST-TOKEN (SAE convention for decoder-only models)")

    log.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    log.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    log.info("Model loaded.")

    # ── source 1: CVE vulnerable ───────────────────────────────────────────────
    log.info("=== Source 1: CVE vulnerable code ===")
    df         = load_cve_pairs()
    df_sampled = sample_cve_by_cwe(df, TARGET_CWES, MAX_SAMPLES_PER_CWE)

    texts_vuln = df_sampled["vulnerable_code"].fillna("").tolist()
    metas_vuln = [
        {"source": "cve", "sample_type": "vulnerable",
         "cwe_id": row["cwe_id"], "language": row.get("language", ""),
         "cve_id": row.get("cve_id", "")}
        for _, row in df_sampled.iterrows()
    ]
    pooled_vuln, mv = process_texts(
        texts_vuln, metas_vuln, tokenizer, model, CANDIDATE_LAYERS, device, "cve_vuln"
    )
    save_activations(pooled_vuln, mv, OUT_ROOT, "cve_vuln")

    # ── source 2: CVE fixed ────────────────────────────────────────────────────
    log.info("=== Source 2: CVE fixed code ===")
    texts_fixed = df_sampled["fixed_code"].fillna("").tolist()
    metas_fixed = [
        {"source": "cve", "sample_type": "fixed",
         "cwe_id": row["cwe_id"], "language": row.get("language", ""),
         "cve_id": row.get("cve_id", "")}
        for _, row in df_sampled.iterrows()
    ]
    pooled_fixed, mf = process_texts(
        texts_fixed, metas_fixed, tokenizer, model, CANDIDATE_LAYERS, device, "cve_fixed"
    )
    save_activations(pooled_fixed, mf, OUT_ROOT, "cve_fixed")

    # ── source 3: CyberSecEval dev ─────────────────────────────────────────────
    log.info("=== Source 3: CyberSecEval dev generations ===")
    dev_outputs = load_cyberseceval_dev()
    texts_dev, metas_dev = [], []
    for item in dev_outputs:
        code = item.get("generated_code", "")
        if not code or not code.strip():
            continue
        texts_dev.append(code)
        metas_dev.append({
            "source": "cyberseceval", "sample_type": "dev_generation",
            "prompt_id": item.get("prompt_id", ""),
            "cwe_id": item.get("cwe_id", "unknown"),
            "language": item.get("language", ""),
        })
    pooled_dev, md = process_texts(
        texts_dev, metas_dev, tokenizer, model, CANDIDATE_LAYERS, device, "cyber_dev"
    )
    save_activations(pooled_dev, md, OUT_ROOT, "cyber_dev")

    # ── summary ────────────────────────────────────────────────────────────────
    log.info("=== Extraction summary ===")
    for l in CANDIDATE_LAYERS:
        for prefix in ["cve_vuln", "cve_fixed", "cyber_dev"]:
            p = OUT_ROOT / f"layer_{l}/{prefix}.pt"
            if p.exists():
                d     = torch.load(p, map_location="cpu")
                shape = tuple(d["activations"].shape) if d["activations"].numel() > 0 else "(empty)"
                log.info(f"  layer {l} | {prefix:12s} | shape={shape} | n={len(d['metadata'])}")
            else:
                log.warning(f"  layer {l} | {prefix:12s} | FILE MISSING")

    log.info("Step 3.1 (revised) complete.")


if __name__ == "__main__":
    main()