"""
Phase 3, Step 3.3 — SAE reconstruction validation
For each candidate layer, decodes latent vectors back to hidden space and
computes relative L2 reconstruction error per source.
Viability threshold: mean relative L2 error < 0.6 AND avg_sparsity (fraction active) <= 0.20.
Outputs:
  outputs/phase3/sae_reconstruction_metrics_layer_{l}.csv
  outputs/phase3/sae_reconstruction_summary.csv
"""

import csv
import logging
from pathlib import Path

import torch
from sae_lens import SAE

# ── constants ──────────────────────────────────────────────────────────────────
CANDIDATE_LAYERS    = [12, 16, 19, 23]
SAE_RELEASE         = "llama_scope_lxr_8x"
SAE_ID_TEMPLATE     = "l{l}r_8x"
BATCH_SIZE          = 64
RECON_ERROR_THRESH  = 0.65    # mean relative L2 error must be below this
SPARSITY_THRESH     = 0.20   # fraction of active features must be below this

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT     = PROJECT_ROOT / "data/activations/raw"
LAT_ROOT     = PROJECT_ROOT / "data/activations/latent"
OUT_DIR      = PROJECT_ROOT / "outputs/phase3"
LOG_PATH     = OUT_DIR / "validate_reconstruction.log"

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


# ── relative L2 error ─────────────────────────────────────────────────────────
def relative_l2(original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """
    Per-sample relative L2: ||x - x_hat||_2 / ||x||_2
    Returns Tensor of shape (N,).
    """
    diff_norm = (original - reconstructed).norm(dim=1)
    orig_norm = original.norm(dim=1).clamp(min=1e-9)
    return diff_norm / orig_norm


def decode_latents_batched(latents: torch.Tensor, sae, device) -> torch.Tensor:
    """Decode (N, d_sae) latents back to (N, d_in) hidden space in batches."""
    results = []
    for start in range(0, latents.shape[0], BATCH_SIZE):
        batch = latents[start : start + BATCH_SIZE].to(device=device, dtype=torch.float32)
        with torch.no_grad():
            recon = sae.decode(batch)   # (B, d_in)
        results.append(recon.cpu())
    return torch.cat(results, dim=0)


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    summary_rows = []   # for sae_reconstruction_summary.csv

    for l in CANDIDATE_LAYERS:
        sae_id = SAE_ID_TEMPLATE.format(l=l)
        log.info(f"=== Layer {l} | SAE: {sae_id} ===")

        sae = SAE.from_pretrained(SAE_RELEASE, sae_id)
        sae = sae.to(device)
        sae.eval()

        per_layer_rows = []   # for sae_reconstruction_metrics_layer_{l}.csv

        for src in SOURCES:
            raw_path = RAW_ROOT  / f"layer_{l}" / f"{src}.pt"
            lat_path = LAT_ROOT  / f"layer_{l}" / f"{src}_latents.pt"

            if not raw_path.exists() or not lat_path.exists():
                log.warning(f"  [{src}] missing raw or latent file — skipped")
                continue

            raw_data = torch.load(raw_path, map_location="cpu")
            lat_data = torch.load(lat_path, map_location="cpu")

            originals = raw_data["activations"].float()   # (N, d_in)
            latents   = lat_data["latents"]               # (N, d_sae)

            if originals.numel() == 0 or latents.numel() == 0:
                log.warning(f"  [{src}] empty tensors — skipped")
                continue

            # decode
            reconstructed = decode_latents_batched(latents, sae, device)  # (N, d_in)

            # reconstruction error
            rel_l2  = relative_l2(originals, reconstructed)   # (N,)
            mean_err = float(rel_l2.mean())
            median_err = float(rel_l2.median())
            max_err = float(rel_l2.max())

            # sparsity (fraction active)
            avg_sparsity = float((latents > 0).float().mean())

            viable = (mean_err < RECON_ERROR_THRESH) and (avg_sparsity <= SPARSITY_THRESH)

            log.info(f"  [{src}] mean_rel_l2={mean_err:.4f}, "
                     f"median={median_err:.4f}, max={max_err:.4f}, "
                     f"sparsity(active)={avg_sparsity:.4f} | viable={viable}")

            row = {
                "layer":        l,
                "source":       src,
                "n_samples":    originals.shape[0],
                "mean_rel_l2":  round(mean_err, 6),
                "median_rel_l2": round(median_err, 6),
                "max_rel_l2":   round(max_err, 6),
                "avg_sparsity_active": round(avg_sparsity, 6),
                "viable":       viable,
            }
            per_layer_rows.append(row)
            summary_rows.append(row)

        # write per-layer CSV
        per_layer_path = OUT_DIR / f"sae_reconstruction_metrics_layer_{l}.csv"
        if per_layer_rows:
            with open(per_layer_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=per_layer_rows[0].keys())
                writer.writeheader()
                writer.writerows(per_layer_rows)
            log.info(f"  Saved {per_layer_path}")

        # viability verdict for this layer
        if per_layer_rows:
            layer_viable = all(r["viable"] for r in per_layer_rows)
            log.info(f"  Layer {l} overall viable: {layer_viable}")

        del sae
        torch.cuda.empty_cache()

    # write summary CSV
    summary_path = OUT_DIR / "sae_reconstruction_summary.csv"
    if summary_rows:
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        log.info(f"Saved {summary_path}")

    # final checkpoint summary
    log.info("=== Checkpoint assessment ===")
    viable_layers = []
    for l in CANDIDATE_LAYERS:
        layer_rows = [r for r in summary_rows if r["layer"] == l]
        if not layer_rows:
            log.info(f"  Layer {l}: no data")
            continue
        all_viable = all(r["viable"] for r in layer_rows)
        mean_errs  = [r["mean_rel_l2"] for r in layer_rows]
        sparsities = [r["avg_sparsity_active"] for r in layer_rows]
        log.info(f"  Layer {l}: viable={all_viable} | "
                 f"mean_rel_l2={[round(e,4) for e in mean_errs]} | "
                 f"sparsity_active={[round(s,4) for s in sparsities]}")
        if all_viable:
            viable_layers.append(l)

    log.info(f"Viable layers (all sources pass): {viable_layers}")
    if len(viable_layers) >= 2:
        log.info("CHECKPOINT PASS: >= 2 viable layers found.")
    else:
        log.warning(f"CHECKPOINT: only {len(viable_layers)} viable layer(s) — see plan fallback.")

    log.info("Step 3.3 complete.")


if __name__ == "__main__":
    main()