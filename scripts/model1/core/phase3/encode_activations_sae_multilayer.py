"""
Phase 3, Step 3.2 — SAE encoding of raw activations
Loads raw activations from data/activations/raw/layer_{l}/ and encodes
them with the matching pretrained SAE for each candidate layer.
Saves latent vectors to data/activations/latent/layer_{l}/.
Also records per-layer encoding statistics in latent_encoding_summary_layer_{l}.json.
"""

import json
import logging
from pathlib import Path

import torch
from sae_lens import SAE

# ── constants ──────────────────────────────────────────────────────────────────
CANDIDATE_LAYERS = [12, 16, 19, 23]
SAE_RELEASE      = "llama_scope_lxr_8x"
SAE_ID_TEMPLATE  = "l{l}r_8x"
BATCH_SIZE       = 64          # encode-only, no model needed — larger batch is fine

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT     = PROJECT_ROOT / "data/activations/raw"
LAT_ROOT     = PROJECT_ROOT / "data/activations/latent"
OUT_DIR      = PROJECT_ROOT / "outputs/phase3"
LOG_PATH     = OUT_DIR / "encode_activations.log"

SOURCES = ["cve_vuln", "cve_fixed", "cyber_dev"]

# ── logging ────────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"),
              logging.StreamHandler()],
)
log = logging.getLogger(__name__)


# ── encode one tensor ──────────────────────────────────────────────────────────
def encode_tensor(activations: torch.Tensor, sae, device) -> torch.Tensor:
    """
    Encode (N, D) activation tensor through SAE encoder in batches.
    Returns latent tensor (N, d_sae).
    """
    all_latents = []
    n = activations.shape[0]
    for start in range(0, n, BATCH_SIZE):
        batch = activations[start : start + BATCH_SIZE].to(device=device, dtype=torch.float32)
        with torch.no_grad():
            latents = sae.encode(batch)   # (B, d_sae)
        all_latents.append(latents.cpu())
    return torch.cat(all_latents, dim=0)  # (N, d_sae)


# ── sparsity / magnitude stats ─────────────────────────────────────────────────
def compute_encoding_stats(latents: torch.Tensor) -> dict:
    """
    latents: (N, d_sae)
    Returns dict with avg_sparsity, avg_magnitude, mean_active_features.
    """
    active      = (latents > 0).float()
    sparsity    = active.mean(dim=1)           # fraction of active features per sample
    magnitudes  = latents.norm(dim=1)          # L2 norm per sample
    n_active    = active.sum(dim=1)            # count of active features per sample
    return {
        "n_samples":           int(latents.shape[0]),
        "d_sae":               int(latents.shape[1]),
        "avg_sparsity":        float(sparsity.mean()),
        "std_sparsity":        float(sparsity.std()),
        "avg_l2_magnitude":    float(magnitudes.mean()),
        "mean_active_features": float(n_active.mean()),
        "max_active_features": float(n_active.max()),
    }


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    for l in CANDIDATE_LAYERS:
        sae_id = SAE_ID_TEMPLATE.format(l=l)
        log.info(f"=== Layer {l} | SAE: {sae_id} ===")

        # load SAE
        sae, _, _ = SAE.from_pretrained(SAE_RELEASE, sae_id)
        sae = sae.to(device)
        sae.eval()
        log.info(f"  SAE loaded: d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")

        layer_lat_dir = LAT_ROOT / f"layer_{l}"
        layer_lat_dir.mkdir(parents=True, exist_ok=True)

        summary = {"layer": l, "sae_id": sae_id,
                   "d_in": sae.cfg.d_in, "d_sae": sae.cfg.d_sae,
                   "sources": {}}

        for src in SOURCES:
            raw_path = RAW_ROOT / f"layer_{l}" / f"{src}.pt"
            if not raw_path.exists():
                log.warning(f"  [{src}] raw file not found: {raw_path} — skipped")
                continue

            data        = torch.load(raw_path, map_location="cpu")
            activations = data["activations"]   # (N, D)
            metadata    = data["metadata"]

            if activations.numel() == 0:
                log.warning(f"  [{src}] empty activations — skipped")
                continue

            log.info(f"  [{src}] encoding {activations.shape[0]} samples...")
            latents = encode_tensor(activations, sae, device)  # (N, d_sae)

            # save
            out_path = layer_lat_dir / f"{src}_latents.pt"
            torch.save({"latents": latents, "metadata": metadata}, out_path)
            log.info(f"  [{src}] saved {out_path} shape={tuple(latents.shape)}")

            # stats
            stats = compute_encoding_stats(latents)
            summary["sources"][src] = stats
            log.info(f"  [{src}] avg_sparsity={stats['avg_sparsity']:.4f}, "
                     f"mean_active_features={stats['mean_active_features']:.1f}, "
                     f"avg_l2={stats['avg_l2_magnitude']:.4f}")

        # save per-layer summary JSON
        summary_path = OUT_DIR / f"latent_encoding_summary_layer_{l}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        log.info(f"  Saved {summary_path}")

        # free SAE VRAM before next layer
        del sae
        torch.cuda.empty_cache()

    log.info("Step 3.2 complete.")


if __name__ == "__main__":
    main()