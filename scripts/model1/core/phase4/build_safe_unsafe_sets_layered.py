"""
Phase 4 — Step 4.1
Build layered safe/unsafe latent sets per CWE per intervention layer.

CWE strategy:
- Detected CWEs (120, 327, 89, 338): use cyber_dev latents split by safe/unsafe IDs
- Zero-detection CWEs (79, 125, 787, 190, 476): use cve_vuln vs cve_fixed latents
"""

import os
import json
import torch
import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path.cwd()
LATENT_DIR   = PROJECT_ROOT / "data/activations/latent"
PHASE2_DIR   = PROJECT_ROOT / "outputs/phase2"
PHASE1_DIR   = PROJECT_ROOT / "data/processed"
OUT_DIR      = PROJECT_ROOT / "outputs/phase4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
LAYERS = [16, 19, 23]

# CWEs with CyberSecEval detections → use cyber_dev partitions
CYBER_CWES = [120, 327, 89, 338]

# CWEs with zero CyberSecEval detections → use CVE pairs
CVE_CWES = [79, 125, 787, 190, 476]

ALL_CWES = CYBER_CWES + CVE_CWES

# ── Helpers ────────────────────────────────────────────────────────────────────
def load_json(path):
    with open(path) as f:
        return json.load(f)

def load_latent(path):
    t = torch.load(path, map_location="cpu")
    if isinstance(t, dict):
        # expect {"latents": Tensor, "meta": [...]}
        return t["latents"], t.get("meta", [])
    return t, []

def save_latent(tensor, meta, path):
    torch.save({"latents": tensor, "meta": meta}, path)
    print(f"  saved {path.name}  shape={tuple(tensor.shape)}")

# ── Step 4.1a — Cyber-detected CWEs ───────────────────────────────────────────
def build_cyber_sets(layer):
    ldir = LATENT_DIR / f"layer_{layer}"
    cyber_path = ldir / "cyber_dev_latents.pt"

    if not cyber_path.exists():
        print(f"[WARN] Missing: {cyber_path}")
        return

    latents, meta = load_latent(cyber_path)
    # meta is a list of dicts with keys: prompt_id, cwe_id, is_vulnerable, etc.
    # If meta is empty try to load from phase2 icd
    if not meta:
        # Use baseline_dev_outputs.json to get ordered prompt_ids matching latent extraction order
        outputs = load_json(PHASE2_DIR / "baseline_dev_outputs.json")
        icd = load_json(PHASE2_DIR / "baseline_dev_icd.json")
        icd_map = {rec["prompt_id"]: rec for rec in icd}
        meta = []
        for rec in outputs:
            pid = rec["prompt_id"]
            icd_rec = icd_map.get(pid, {})
            cwes = [f["cwe_id"] for f in icd_rec.get("findings", []) if f.get("cwe_id")]
            meta.append({"prompt_id": pid, "cwe_ids": cwes, "is_vulnerable": len(cwes) > 0})
        if len(meta) != latents.shape[0]:
            print(f"[WARN] layer {layer}: cyber latents {latents.shape[0]} != outputs entries {len(meta)}")

    for cwe in CYBER_CWES:
        cwe_str = f"CWE-{cwe}"
        # load per-CWE partition files
        suffix = "_exploratory" if cwe == 338 else ""
        unsafe_id_path = PHASE2_DIR / f"unsafe_dev_ids_CWE-{cwe}{suffix}.json"
        safe_id_path   = PHASE2_DIR / f"safe_dev_ids_CWE-{cwe}{suffix}.json"
        if not unsafe_id_path.exists() or not safe_id_path.exists():
            print(f"[WARN] CWE-{cwe}: partition files missing, skipping cyber split")
            continue

        unsafe_ids = set(load_json(unsafe_id_path))
        safe_ids   = set(load_json(safe_id_path))

        # align indices using meta
        unsafe_idx, safe_idx = [], []
        for i, m in enumerate(meta):
            pid = m.get("prompt_id", m.get("id", i))
            if pid in unsafe_ids:
                unsafe_idx.append(i)
            elif pid in safe_ids:
                safe_idx.append(i)

        if not unsafe_idx:
            print(f"[WARN] layer {layer} CWE-{cwe}: 0 unsafe samples from cyber_dev")
            continue

        unsafe_t = latents[unsafe_idx]
        safe_t   = latents[safe_idx]

        out_unsafe = OUT_DIR / f"unsafe_latents_layer_{layer}_cwe{cwe}.pt"
        out_safe   = OUT_DIR / f"safe_latents_layer_{layer}_cwe{cwe}.pt"
        save_latent(unsafe_t, [meta[i] for i in unsafe_idx], out_unsafe)
        save_latent(safe_t,   [meta[i] for i in safe_idx],   out_safe)

# ── Step 4.1b — CVE-pair CWEs ─────────────────────────────────────────────────
def build_cve_sets(layer):
    ldir = LATENT_DIR / f"layer_{layer}"
    vuln_path  = ldir / "cve_vuln_latents.pt"
    fixed_path = ldir / "cve_fixed_latents.pt"

    if not vuln_path.exists() or not fixed_path.exists():
        print(f"[WARN] layer {layer}: CVE latent files missing")
        return

    vuln_latents,  vuln_meta  = load_latent(vuln_path)
    fixed_latents, fixed_meta = load_latent(fixed_path)

    # Load train + val pairs to get CWE labels per sample
    train = pd.read_csv(PHASE1_DIR / "train_pairs.csv")
    val   = pd.read_csv(PHASE1_DIR / "val_pairs.csv")
    pairs = pd.concat([train, val], ignore_index=True)

    for cwe in CVE_CWES:
        cwe_str = f"CWE-{cwe}"

        # find row indices for this CWE
        mask = pairs["cwe_id"].astype(str).str.contains(str(cwe), na=False)
        cwe_ids = set(pairs[mask].index.tolist())

        # filter latent tensors by matching meta pair_id / index
        # meta entries should have "pair_id" or "index" field
        def get_cwe_idx(meta, pairs_df):
            idx = []
            for i, m in enumerate(meta):
                pid = m.get("pair_id", m.get("index", None))
                if pid is not None and pid in cwe_ids:
                    idx.append(i)
                elif pid is None and i < len(pairs_df):
                    # fallback: positional alignment
                    row_cwe = str(pairs_df.iloc[i].get("cwe_id", ""))
                    if str(cwe) in row_cwe:
                        idx.append(i)
            return idx

        vuln_idx  = get_cwe_idx(vuln_meta,  pairs)
        fixed_idx = get_cwe_idx(fixed_meta, pairs)

        # fallback: positional if meta is empty
        if not vuln_meta:
            vuln_idx  = pairs[mask].index.tolist()
            fixed_idx = pairs[mask].index.tolist()
            vuln_idx  = [i for i in vuln_idx  if i < vuln_latents.shape[0]]
            fixed_idx = [i for i in fixed_idx if i < fixed_latents.shape[0]]

        if not vuln_idx:
            print(f"[WARN] layer {layer} CWE-{cwe}: 0 vuln samples in CVE latents")
            continue

        vuln_t  = vuln_latents[vuln_idx]
        fixed_t = fixed_latents[fixed_idx]

        # For CVE-based CWEs: vuln = unsafe proxy, fixed = safe proxy
        out_vuln  = OUT_DIR / f"unsafe_latents_layer_{layer}_cwe{cwe}.pt"
        out_fixed = OUT_DIR / f"safe_latents_layer_{layer}_cwe{cwe}.pt"
        out_vuln_cve  = OUT_DIR / f"cve_vuln_latents_layer_{layer}_cwe{cwe}.pt"
        out_fixed_cve = OUT_DIR / f"cve_fixed_latents_layer_{layer}_cwe{cwe}.pt"

        save_latent(vuln_t,  [vuln_meta[i]  if vuln_meta  else {} for i in vuln_idx],  out_vuln)
        save_latent(fixed_t, [fixed_meta[i] if fixed_meta else {} for i in fixed_idx], out_fixed)
        save_latent(vuln_t,  [vuln_meta[i]  if vuln_meta  else {} for i in vuln_idx],  out_vuln_cve)
        save_latent(fixed_t, [fixed_meta[i] if fixed_meta else {} for i in fixed_idx], out_fixed_cve)

# ── Summary ────────────────────────────────────────────────────────────────────
def summarize():
    summary = {}
    for layer in LAYERS:
        summary[f"layer_{layer}"] = {}
        for cwe in ALL_CWES:
            unsafe_f = OUT_DIR / f"unsafe_latents_layer_{layer}_cwe{cwe}.pt"
            safe_f   = OUT_DIR / f"safe_latents_layer_{layer}_cwe{cwe}.pt"
            n_unsafe = n_safe = 0
            if unsafe_f.exists():
                t = torch.load(unsafe_f, map_location="cpu")
                n_unsafe = t["latents"].shape[0] if isinstance(t, dict) else t.shape[0]
            if safe_f.exists():
                t = torch.load(safe_f, map_location="cpu")
                n_safe = t["latents"].shape[0] if isinstance(t, dict) else t.shape[0]
            summary[f"layer_{layer}"][f"CWE-{cwe}"] = {
                "n_unsafe": n_unsafe,
                "n_safe":   n_safe,
                "source":   "cyber_dev" if cwe in CYBER_CWES else "cve_pairs"
            }

    out_path = OUT_DIR / "step4_1_latent_set_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {out_path}")

    # Print table
    print(f"\n{'Layer':<8} {'CWE':<10} {'Source':<12} {'n_unsafe':>8} {'n_safe':>8}")
    print("-" * 52)
    for layer in LAYERS:
        for cwe in ALL_CWES:
            row = summary[f"layer_{layer}"][f"CWE-{cwe}"]
            print(f"{layer:<8} CWE-{cwe:<6} {row['source']:<12} {row['n_unsafe']:>8} {row['n_safe']:>8}")

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for layer in LAYERS:
        print(f"\n=== Layer {layer} ===")
        print("  [Cyber-detected CWEs]")
        build_cyber_sets(layer)
        print("  [CVE-pair CWEs]")
        build_cve_sets(layer)

    summarize()
    print("\nStep 4.1 complete.")