"""
Phase 6 Step 6.1 - Build Bandit States
State dim: 64 (Block A) + 11 (Block B) + 27 (Block C: 9 groups x 3) = 102 <= 140
"""
import json
import numpy as np
import torch
import os

SEED = 42
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

ROOT = "."
LATENT_DIR = os.path.join(ROOT, "data/activations/latent")
OUT_DIR = os.path.join(ROOT, "outputs/phase6")
os.makedirs(OUT_DIR, exist_ok=True)

# Active groups only (CWE-120, CWE-327, CWE-89)
ACTIVE_CWES = ["CWE-120", "CWE-327", "CWE-89"]

DERIVATION_CWES = [
    "CWE-120", "CWE-125", "CWE-787", "CWE-190", "CWE-476",
    "CWE-89", "CWE-79", "CWE-327", "CWE-338"
]
HELDOUT_CWES = ["CWE-22", "CWE-290"]

DIM_A = 64
DIM_B = 11  # 9 derivation + 1 heldout + 1 unknown
FEATS_PER_GROUP = 3  # max, mean, frac_active

print("Loading dev_prompts.json...")
dev_prompts = json.load(open(os.path.join(ROOT, "data/cyberseceval/dev_prompts.json")))
pid2prompt = {p["prompt_id"]: p for p in dev_prompts}
prompt_ids = [p["prompt_id"] for p in dev_prompts]
print(f"  {len(prompt_ids)} dev prompts")

print("Loading intervention library (active groups only)...")
lib = json.load(open(os.path.join(ROOT, "configs/intervention_library.json")))
all_groups = lib["groups"]
selected_groups = [g for g in all_groups if g["cwe_id"] in ACTIVE_CWES]
selected_groups = sorted(selected_groups, key=lambda g: (g["cwe_id"], g["group_id"]))
assert len(selected_groups) == 9, f"Expected 9 groups, got {len(selected_groups)}"
print(f"  Selected {len(selected_groups)} groups: {[g['group_id'] for g in selected_groups]}")

print("Loading latents for all intervention layers...")
layer_latents = {}
layer_meta_idx = {}
for layer in [16, 19, 23]:
    data = torch.load(os.path.join(LATENT_DIR, f"layer_{layer}/cyber_dev_latents.pt"))
    latents = data["latents"].float().numpy()  # [1341, 32768]
    metadata = data["metadata"]
    pid2idx = {m["prompt_id"]: i for i, m in enumerate(metadata)}
    layer_latents[layer] = latents
    layer_meta_idx[layer] = pid2idx
    print(f"  Layer {layer}: {latents.shape}, {len(pid2idx)} prompts indexed")

# Fixed random projection matrix for Block A (layer 19 latents -> 64 dims)
SAE_DIM = 32768
proj_matrix = rng.standard_normal((SAE_DIM, DIM_A)).astype(np.float32)
proj_matrix /= np.linalg.norm(proj_matrix, axis=0, keepdims=True)

def build_block_a(pid):
    """64-dim projection of layer-19 SAE latent."""
    idx = layer_meta_idx[19].get(pid)
    if idx is None:
        return np.zeros(DIM_A, dtype=np.float32)
    latent = layer_latents[19][idx]  # [32768]
    proj = latent @ proj_matrix  # [64]
    norm = np.linalg.norm(proj)
    if norm > 0:
        proj = proj / norm
    return proj.astype(np.float32)

def build_block_b(pid):
    """11-dim CWE prior vector."""
    prompt = pid2prompt[pid]
    cwe = prompt.get("cwe_identifier", "unknown")
    vec = np.zeros(DIM_B, dtype=np.float32)
    # dims 0-8: derivation CWE one-hots
    if cwe in DERIVATION_CWES:
        vec[DERIVATION_CWES.index(cwe)] = 1.0
    # dim 9: held-out CWE marker
    if cwe in HELDOUT_CWES:
        vec[9] = 1.0
    # dim 10: unknown/other
    if cwe not in DERIVATION_CWES and cwe not in HELDOUT_CWES:
        vec[10] = 1.0
    return vec

def build_block_c(pid):
    """27-dim group activation summary (9 groups x 3 features)."""
    vec = np.zeros(len(selected_groups) * FEATS_PER_GROUP, dtype=np.float32)
    for i, group in enumerate(selected_groups):
        layer = group["layer_id"]
        feature_ids = group["feature_ids"]
        idx = layer_meta_idx[layer].get(pid)
        if idx is None:
            continue
        latent = layer_latents[layer][idx]  # [32768]
        activations = latent[feature_ids]
        vec[i*3 + 0] = float(activations.max())
        vec[i*3 + 1] = float(activations.mean())
        vec[i*3 + 2] = float((activations > 0).mean())
    return vec

print("\nBuilding states for all dev prompts...")
states = {}
missing = 0
for pid in prompt_ids:
    if pid not in layer_meta_idx[19]:
        missing += 1
        continue
    block_a = build_block_a(pid)
    block_b = build_block_b(pid)
    block_c = build_block_c(pid)
    state = np.concatenate([block_a, block_b, block_c])
    states[pid] = state.tolist()

print(f"  Built states: {len(states)}, missing: {missing}")
print(f"  State dim: {len(list(states.values())[0])}")
assert len(list(states.values())[0]) == DIM_A + DIM_B + len(selected_groups) * FEATS_PER_GROUP

# Save states
states_path = os.path.join(OUT_DIR, "bandit_states.json")
json.dump({str(k): v for k, v in states.items()}, open(states_path, "w"))
print(f"  Saved: {states_path}")

# Save schema
schema = {
    "state_dim": DIM_A + DIM_B + len(selected_groups) * FEATS_PER_GROUP,
    "block_a": {"start": 0, "end": DIM_A, "description": "layer19 SAE latent random projection"},
    "block_b": {"start": DIM_A, "end": DIM_A + DIM_B,
                 "description": "CWE prior", "cwe_order": DERIVATION_CWES + ["heldout", "unknown"]},
    "block_c": {"start": DIM_A + DIM_B, "end": DIM_A + DIM_B + len(selected_groups) * FEATS_PER_GROUP,
                 "description": "group activation summaries",
                 "groups": [{"index": i, "group_id": g["group_id"], "layer": g["layer_id"],
                              "cwe_id": g["cwe_id"], "features": g["feature_ids"]}
                             for i, g in enumerate(selected_groups)],
                 "dims_per_group": ["max_activation", "mean_activation", "frac_active"]},
    "seed": SEED,
    "projection_shape": [SAE_DIM, DIM_A],
    "n_prompts": len(states)
}
schema_path = os.path.join(OUT_DIR, "bandit_state_schema.json")
json.dump(schema, open(schema_path, "w"), indent=2)
print(f"  Saved: {schema_path}")

print("\nDone. State dim:", schema["state_dim"])