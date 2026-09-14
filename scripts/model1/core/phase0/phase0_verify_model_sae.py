"""
Phase 0 - Step 0.3: Verify model and SAE loading.
- Loads Meta-Llama-3.1-8B-Instruct
- Loads SAELens layer-19 SAE
- Verifies encode -> decode has no NaN/Inf
- Verifies layer-19 activations capturable via hook
Output: outputs/phase0/model_sae_verification.json
"""

import os
import json
import torch
import datetime

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs/phase0")

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
SAE_RELEASE = "llama_scope_lxr_8x"
SAE_ID = "l19r_8x"
LAYER = 19


def verify_model_loading():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"[1/4] Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    print(f"      Model loaded. Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'N/A'}")
    return model, tokenizer


def verify_sae_loading():
    from sae_lens import SAE
    print(f"[2/4] Loading SAE: release={SAE_RELEASE}, id={SAE_ID}")
    sae, cfg_dict, _ = SAE.from_pretrained(
        release=SAE_RELEASE,
        sae_id=SAE_ID,
    )
    sae.eval()
    print(f"      SAE loaded. d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")
    return sae


def verify_encode_decode(model, tokenizer, sae):
    print(f"[3/4] Verifying SAE encode -> decode (no NaN/Inf)...")
    device = next(sae.parameters()).device

    # Capture layer-19 activation via hook
    captured = {}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            captured["activation"] = output[0].detach()
        else:
            captured["activation"] = output.detach()

    hook = model.model.layers[LAYER].register_forward_hook(hook_fn)

    prompt = "Write a Python function to add two numbers."
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        model(**inputs)

    hook.remove()

    assert "activation" in captured, "Hook did not capture activation"
    act = captured["activation"]  # shape: [1, seq_len, d_model]
    print(f"      Captured activation shape: {act.shape}")

    # Take mean over sequence positions, move to SAE device
    act_flat = act.mean(dim=1).to(device=device, dtype=torch.float32)

    # Encode
    latents = sae.encode(act_flat)
    assert not torch.isnan(latents).any(), "NaN in SAE latents"
    assert not torch.isinf(latents).any(), "Inf in SAE latents"

    # Decode
    reconstructed = sae.decode(latents)
    assert not torch.isnan(reconstructed).any(), "NaN in SAE reconstruction"
    assert not torch.isinf(reconstructed).any(), "Inf in SAE reconstruction"

    sparsity = (latents == 0).float().mean().item()
    print(f"      Latent sparsity: {sparsity:.4f}")
    print(f"      Encode -> decode: OK, no NaN/Inf")

    return {
        "activation_shape": list(act.shape),
        "latent_shape": list(latents.shape),
        "sparsity": round(sparsity, 4),
        "nan_in_latents": False,
        "inf_in_latents": False,
        "nan_in_reconstruction": False,
        "inf_in_reconstruction": False,
    }


def verify_hook_capture(model, tokenizer):
    print(f"[4/4] Verifying layer-{LAYER} activation hook capture...")
    captured = {}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            captured["activation"] = output[0].detach()
        else:
            captured["activation"] = output.detach()

    hook = model.model.layers[LAYER].register_forward_hook(hook_fn)
    prompt = "def hello(): pass"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        model(**inputs)

    hook.remove()

    assert "activation" in captured, "Hook failed to capture activation"
    shape = list(captured["activation"].shape)
    print(f"      Hook capture OK. Shape: {shape}")
    return shape


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "sae_release": SAE_RELEASE,
        "sae_id": SAE_ID,
        "layer": LAYER,
        "checks": {}
    }

    try:
        model, tokenizer = verify_model_loading()
        results["checks"]["model_load"] = "PASS"
    except Exception as e:
        results["checks"]["model_load"] = f"FAIL: {e}"
        print(f"[FAIL] Model loading: {e}")
        _save_and_exit(results)
        return

    try:
        sae = verify_sae_loading()
        results["checks"]["sae_load"] = "PASS"
    except Exception as e:
        results["checks"]["sae_load"] = f"FAIL: {e}"
        print(f"[FAIL] SAE loading: {e}")
        _save_and_exit(results)
        return

    try:
        enc_dec_results = verify_encode_decode(model, tokenizer, sae)
        results["checks"]["encode_decode"] = "PASS"
        results["encode_decode_details"] = enc_dec_results
    except Exception as e:
        results["checks"]["encode_decode"] = f"FAIL: {e}"
        print(f"[FAIL] Encode-decode: {e}")

    try:
        hook_shape = verify_hook_capture(model, tokenizer)
        results["checks"]["hook_capture"] = "PASS"
        results["hook_activation_shape"] = hook_shape
    except Exception as e:
        results["checks"]["hook_capture"] = f"FAIL: {e}"
        print(f"[FAIL] Hook capture: {e}")

    _save_and_exit(results)


def _save_and_exit(results):
    out_path = os.path.join(OUTPUT_DIR, "model_sae_verification.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Step 0.3 Summary ===")
    for check, status in results["checks"].items():
        print(f"  {check}: {status}")

    all_pass = all("PASS" in str(v) for v in results["checks"].values())
    print(f"\n[{'OK' if all_pass else 'FAIL'}] model_sae_verification.json -> outputs/phase0/")


if __name__ == "__main__":
    main()