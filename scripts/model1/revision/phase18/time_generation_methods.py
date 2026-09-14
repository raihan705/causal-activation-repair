#!/usr/bin/env python3
"""Time one exact frozen Phase 18 generation method at batch size one."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
PHASE = ROOT / "revision/model1/phase18"
OUTPUTS = PHASE / "outputs"
SUBSET = OUTPUTS / "timing_subset_manifest.json"
RAW_DIR = OUTPUTS / "raw_timing"

PHASE16_SCRIPTS = ROOT / "revision/model1/phase16/scripts"
sys.path.insert(0, str(PHASE16_SCRIPTS))
from phase16_common import (  # noqa: E402
    MODEL_ID, MODEL_REVISION, SAE_RELEASE, SAE_REVISION, SUBMITTED_B1_PREFIX,
    load_model, load_saes,
)

METHODS = ("B0", "B1", "B*", "B3-ungated", "B3-PG", "B1-CWE", "RCI-1", "CAA-CWE")
SEED = 42
REPETITIONS = (1, 2, 3)
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.2
TOP_P = 0.95
KL_THRESHOLD = 0.5
DECAY = 0.5

BSTAR = ROOT / "configs/bstar_config.json"
LIBRARY = ROOT / "configs/intervention_library.json"
B1_SPEC = ROOT / "revision/model1/phase14/outputs/b1cwe_method_spec.json"
B1_GUIDANCE = ROOT / "revision/model1/phase14/outputs/b1cwe_guidance_map.json"
RCI_SPEC = ROOT / "revision/model1/phase14/outputs/rci_method_spec.json"
RCI_SELECTION = ROOT / "revision/model1/phase14/outputs/rci_selection.json"
CAA_CONFIG = ROOT / "revision/model1/phase14/outputs/caa_cwe_config.json"
CAA_MANIFEST = ROOT / "revision/model1/phase14/outputs/caa_vector_manifest.json"
CAA_HOOK_SOURCE = ROOT / "revision/model1/phase14/scripts/caa_position_mask.py"

EXPECTED = {
    SUBSET: "b25fc05efc915f1f6fcde7b4a78b06f771605c356ce991551a0c38b451665c3c",
    BSTAR: "f4f5d4d6a697aa14f654763f31e52cd852ce87b9f284276edd4f784604ea36e6",
    LIBRARY: "d5ceb2c2ac8b9dccd2f6b9f76476ac0a0ff4f0d6ca1944dd1f174c01c15f4cf2",
    B1_SPEC: "9a4116027c50f2a714a977af6eca2c37035d0920c42cfbbb2131114ddb396550",
    B1_GUIDANCE: "1ab8f53f437d59b870ff3e5512f5c3728acf748a1d379a52e8a25f7fb4326c8b",
    RCI_SPEC: "101ffa3e74ceda2117e46794d029bd9d80b6429c578b67aea99808ce7b6b427e",
    RCI_SELECTION: "c81e75be759f64784afd2a3c98e730a4f16ed7f6465be5ad8a207a557f9b1753",
    CAA_CONFIG: "5e11a4f21f2cc3e120880be1e1d99d48a684ba79d0ee6e3695d59bbf70b04e18",
    CAA_MANIFEST: "172e252e8e7955491bfed1118258cc6f39ec2a3e59b099dbd4df31a49270f36e",
    CAA_HOOK_SOURCE: "01014341aed9ad38383b34710a8320932a42c2a61b69c677cb602e3ea332f38a",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def slug(method: str) -> str:
    return method.lower().replace("*", "star").replace("-", "_")


def selected_groups() -> dict[str, dict[str, Any]]:
    priority = {"mixed_robust": 0, "semantic": 1, "statistical_fallback": 2}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_json(LIBRARY)["groups"]:
        grouped[row["cwe_id"]].append(row)
    return {cwe: sorted(rows, key=lambda row: priority.get(row["group_type"], 3))[0] for cwe, rows in grouped.items()}


def method_artifacts(method: str) -> list[Path]:
    mapping = {
        "B0": [], "B1": [], "B*": [BSTAR],
        "B3-ungated": [LIBRARY], "B3-PG": [LIBRARY],
        "B1-CWE": [B1_SPEC, B1_GUIDANCE],
        "RCI-1": [RCI_SPEC, RCI_SELECTION],
        "CAA-CWE": [CAA_CONFIG, CAA_MANIFEST, CAA_HOOK_SOURCE],
    }
    return mapping[method]


def preflight(method: str) -> dict[str, Any]:
    require(method in METHODS, f"Unsupported method {method}")
    for path in [SUBSET, *method_artifacts(method)]:
        require(path.is_file(), f"Missing frozen input: {path}")
        require(sha256(path) == EXPECTED[path], f"Frozen input hash mismatch: {path}")
    subset = read_json(SUBSET)
    require(subset["status"] == "FROZEN_BEFORE_TIMING", "Timing subset is not frozen")
    require(subset["prompt_count"] == 50 and len(subset["ordered_prompt_ids"]) == 50, "Timing subset count mismatch")
    require(all(subset["validation"].values()), "Timing subset validation failed")
    active = set(subset["active_cwe_labels"])
    detail: dict[str, Any] = {}
    if method == "B*":
        config = read_json(BSTAR)
        require(config["bstar"] == "B2_a40" and float(config["alpha"]) == 40.0, "B* config mismatch")
        require(active <= set(config["feature_map"]), "B* route missing")
        detail = {"layers": sorted({int(config["feature_map"][cwe]["layer"]) for cwe in active}), "alpha": 40.0}
    elif method in ("B3-ungated", "B3-PG"):
        groups = selected_groups()
        require(active <= set(groups), "B3 group route missing")
        detail = {"layers": sorted({int(groups[cwe]["layer_id"]) for cwe in active}), "base_alpha": 20.0,
                  "plausibility_gate": method == "B3-PG"}
    elif method == "B1-CWE":
        guidance = read_json(B1_GUIDANCE)
        require(active <= set(guidance["routes"]), "B1-CWE route missing")
        require(all(not guidance["routes"][cwe]["guidance_available"] for cwe in active), "Unexpected B1-CWE guidance")
        detail = {"all_active_routes": "FALLBACK_SUBMITTED_B1", "model_calls_per_prompt": 1}
    elif method == "RCI-1":
        selection = read_json(RCI_SELECTION)
        require(selection.get("selected_variant") in ("RCI-1", "rci1"), "RCI-1 is not frozen selected variant")
        detail = {"stages": ["initial", "critique_1", "improve_1"], "intended_model_calls_per_prompt": 3}
    elif method == "CAA-CWE":
        config = read_json(CAA_CONFIG)
        require(config["status"] == "FROZEN" and int(config["selected_layer"]) == 16 and float(config["selected_multiplier"]) == 1.0, "CAA config mismatch")
        require(active <= set(config["route_map"]), "CAA route missing")
        for cwe in active:
            vector = ROOT / config["route_map"][cwe]["vector_path"]
            require(vector.is_file() and sha256(vector) == config["route_map"][cwe]["vector_sha256"], f"CAA vector mismatch: {cwe}")
        detail = {"layer": 16, "multiplier": 1.0, "active_vectors": sorted(active)}
    return {
        "status": "PASS", "method": method, "batch_size": 1, "seed": SEED,
        "repetitions": list(REPETITIONS), "warmup_prompt_id": subset["warmup_prompt_id"],
        "timing_subset_sha256": EXPECTED[SUBSET], "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION, "sae_release": SAE_RELEASE,
        "sae_revision": SAE_REVISION, "method_detail": detail,
        "artifact_hashes": {path.relative_to(ROOT).as_posix(): EXPECTED[path] for path in method_artifacts(method)},
        "heldout_used": False, "optimization_introduced": False,
    }


def make_sae_hook(sae: Any, features: list[int], alpha: float):
    def hook_fn(module: Any, inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        z = sae.encode(hidden)
        for feature in features:
            z[..., feature] += alpha
        edited = sae.decode(z).to(hidden.dtype)
        return (edited,) + output[1:] if isinstance(output, tuple) else edited
    return hook_fn


def generate_once(torch: Any, model: Any, tokenizer: Any, prompt: str) -> tuple[str, int, int]:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE,
                                top_p=TOP_P, do_sample=True, pad_token_id=tokenizer.pad_token_id)
    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True), int(inputs["input_ids"].shape[1]), int(new_tokens.shape[0])


def kl_divergence(torch: Any, raw: Any, steered: Any) -> float:
    import torch.nn.functional as functional
    p = functional.softmax(raw, dim=-1).clamp(min=1e-9)
    q = functional.softmax(steered, dim=-1).clamp(min=1e-9)
    return float((p * (p.log() - q.log())).sum().item())


def generate_pg(torch: Any, model: Any, tokenizer: Any, input_ids: Any, sae: Any,
                layer: int, features: list[int], base_alpha: float) -> tuple[str, int, list[dict[str, Any]], int]:
    import torch.nn.functional as functional
    generated = input_ids.clone()
    gate_log: list[dict[str, Any]] = []
    model_calls = 0
    for step in range(MAX_NEW_TOKENS):
        with torch.no_grad():
            raw_output = model(generated)
        model_calls += 1
        torch.cuda.synchronize()
        raw_logits = raw_output.logits[:, -1, :]
        alpha_used = base_alpha
        steered_logits = None
        kl = 0.0
        for attempt in range(2):
            handle = model.model.layers[layer].register_forward_hook(make_sae_hook(sae, features, alpha_used))
            try:
                with torch.no_grad():
                    steered_output = model(generated)
                model_calls += 1
                torch.cuda.synchronize()
            finally:
                handle.remove()
            steered_logits = steered_output.logits[:, -1, :]
            kl = kl_divergence(torch, raw_logits[0], steered_logits[0])
            if kl <= KL_THRESHOLD:
                break
            if attempt == 0:
                alpha_used = base_alpha * DECAY
            else:
                steered_logits = raw_logits
                alpha_used = 0.0
        gate_log.append({"step": step, "kl": round(kl, 4) if math.isfinite(kl) else str(kl), "alpha_used": alpha_used})
        logits = steered_logits[0] / TEMPERATURE
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probabilities = functional.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probabilities, dim=-1)
        remove = cumulative - probabilities > TOP_P
        sorted_logits[remove] = float("-inf")
        next_index = torch.multinomial(functional.softmax(sorted_logits, dim=-1), num_samples=1)
        next_token = sorted_indices[next_index].unsqueeze(0)
        generated = torch.cat([generated, next_token], dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break
    new_tokens = generated[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True), int(new_tokens.shape[0]), gate_log, model_calls


def language_tag(language: str) -> str:
    mapping = {"c++": "cpp", "cpp": "cpp", "js": "javascript"}
    value = language.strip().lower()
    return mapping.get(value, value or "text")


def extract_fenced_code(text: str) -> tuple[str, str | None]:
    match = re.search(r"```([A-Za-z0-9_+.#-]+)[ \t]*\r?\n([\s\S]*?)```", text)
    if not match or not match.group(2).strip():
        return "", "EXTRACTION_FAILED_INVALID_OR_EMPTY"
    return match.group(2), None


def load_method_assets(torch: Any, method: str) -> dict[str, Any]:
    assets: dict[str, Any] = {}
    if method == "B*":
        assets["config"] = read_json(BSTAR)
        active = set(read_json(SUBSET)["active_cwe_labels"])
        assets["saes"] = load_saes([int(assets["config"]["feature_map"][cwe]["layer"]) for cwe in active])
    elif method in ("B3-ungated", "B3-PG"):
        assets["groups"] = selected_groups()
        active = set(read_json(SUBSET)["active_cwe_labels"])
        assets["saes"] = load_saes([int(assets["groups"][cwe]["layer_id"]) for cwe in active])
    elif method == "RCI-1":
        assets["spec"] = read_json(RCI_SPEC)
    elif method == "CAA-CWE":
        config = read_json(CAA_CONFIG)
        vectors = {}
        for cwe, route in config["route_map"].items():
            payload = torch.load(ROOT / route["vector_path"], map_location="cpu", weights_only=True)
            vectors[cwe] = payload["prepared_vector"]
        sys.path.insert(0, str(CAA_HOOK_SOURCE.parent))
        from caa_position_mask import CAAResidualPositionHook
        assets.update({"vectors": vectors, "hook_class": CAAResidualPositionHook})
    return assets


def execute_prompt(torch: Any, model: Any, tokenizer: Any, method: str, source: dict[str, Any], assets: dict[str, Any]) -> dict[str, Any]:
    prompt = source["prompt_text"]
    cwe = source["cwe_identifier"]
    route_status = "RAW"
    fallback_status = "NONE"
    calls = 0
    input_tokens = 0
    generated_tokens = 0
    generated_code = ""
    extra: dict[str, Any] = {}
    failure = None
    try:
        if method in ("B1", "B1-CWE"):
            prompt = SUBMITTED_B1_PREFIX + prompt
            if method == "B1-CWE":
                route_status = "AUTHORIZED_ROUTE_GUIDANCE_UNAVAILABLE"
                fallback_status = "SUBMITTED_B1"
        if method in ("B0", "B1", "B1-CWE"):
            generated_code, input_tokens, generated_tokens = generate_once(torch, model, tokenizer, prompt)
            calls = 1
        elif method == "B*":
            route = assets["config"]["feature_map"][cwe]
            layer, feature = int(route["layer"]), int(route["feature"])
            handle = model.model.layers[layer].register_forward_hook(make_sae_hook(assets["saes"][layer], [feature], 40.0))
            try:
                generated_code, input_tokens, generated_tokens = generate_once(torch, model, tokenizer, prompt)
                calls = 1
            finally:
                handle.remove()
            route_status = "SUPPORTED_FEATURE_APPLIED"
            extra = {"layer": layer, "feature_ids": [feature], "alpha": 40.0}
        elif method == "B3-ungated":
            group = assets["groups"][cwe]
            layer = int(group["layer_id"])
            features = list(map(int, group["feature_ids"]))
            alpha = float(group["base_alpha"])
            handle = model.model.layers[layer].register_forward_hook(make_sae_hook(assets["saes"][layer], features, alpha))
            try:
                generated_code, input_tokens, generated_tokens = generate_once(torch, model, tokenizer, prompt)
                calls = 1
            finally:
                handle.remove()
            route_status = "SUPPORTED_GROUP_APPLIED"
            extra = {"group_id": group["group_id"], "layer": layer, "feature_ids": features, "alpha": alpha}
        elif method == "B3-PG":
            group = assets["groups"][cwe]
            layer = int(group["layer_id"])
            features = list(map(int, group["feature_ids"]))
            alpha = float(group["base_alpha"])
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
            input_tokens = int(inputs["input_ids"].shape[1])
            generated_code, generated_tokens, gate_log, calls = generate_pg(
                torch, model, tokenizer, inputs["input_ids"], assets["saes"][layer], layer, features, alpha)
            route_status = "SUPPORTED_GROUP_PLAUSIBILITY_GATE_APPLIED"
            extra = {"group_id": group["group_id"], "layer": layer, "feature_ids": features, "alpha": alpha,
                     "gate_steps": len(gate_log), "gated_steps": sum(row["alpha_used"] < alpha for row in gate_log), "gate_log": gate_log}
        elif method == "CAA-CWE":
            vector = assets["vectors"][cwe]
            hook = assets["hook_class"](vector, 1.0, expected_width=4096)
            handle = model.model.layers[16].register_forward_hook(hook)
            try:
                generated_code, input_tokens, generated_tokens = generate_once(torch, model, tokenizer, prompt)
                calls = 1
            finally:
                handle.remove()
            route_status = "SUPPORTED_VECTOR_APPLIED"
            extra = {"layer": 16, "multiplier": 1.0, "hook_call_count": len(hook.call_records)}
        else:
            spec = assets["spec"]
            tag = language_tag(source["language"])
            current_code = ""
            critique = ""
            stage_records = []
            stages = ("initial", "critique_1", "improve_1")
            for stage_index, stage in enumerate(stages):
                if stage == "initial":
                    rendered = spec["templates"]["initial"]["template"].format(language=source["language"], language_tag=tag, coding_task=prompt)
                elif stage == "critique_1":
                    rendered = spec["templates"]["criticism"]["template"].format(current_code=current_code)
                else:
                    rendered = spec["templates"]["improvement"]["template"].format(language_tag=tag, current_critique=critique, current_code=current_code)
                text, stage_input, stage_generated = generate_once(torch, model, tokenizer, rendered)
                calls += 1
                input_tokens += stage_input
                generated_tokens += stage_generated
                stage_status = "COMPLETED"
                stage_failure = None
                extracted = ""
                if stage in ("initial", "improve_1"):
                    extracted, stage_failure = extract_fenced_code(text)
                    if stage_failure:
                        stage_status = stage_failure
                    else:
                        current_code = extracted
                elif text.strip():
                    critique = text
                else:
                    stage_status = "INVALID_OR_EMPTY"
                    stage_failure = "Empty criticism output"
                stage_records.append({"stage_index": stage_index, "stage_name": stage, "input_tokens": stage_input,
                                      "generated_tokens": stage_generated, "status": stage_status, "failure": stage_failure})
                if stage_failure:
                    break
            generated_code = current_code if len(stage_records) == 3 and stage_records[-1]["status"] == "COMPLETED" else ""
            route_status = "METADATA_FREE_RCI1"
            extra = {"stage_records": stage_records}
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
    return {
        "prompt_id": int(source["prompt_id"]), "cwe_identifier": cwe, "language": source["language"],
        "input_tokens": input_tokens, "generated_tokens": generated_tokens,
        "model_call_count": calls, "generated_code": generated_code,
        "generation_status": "COMPLETED" if failure is None else "FAILED", "failure": failure,
        "fallback_status": fallback_status, "route_status": route_status, **extra,
    }


def capture_rng(torch: Any) -> dict[str, Any]:
    def encode(tensor: Any) -> str:
        return base64.b64encode(bytes(tensor.detach().cpu().tolist())).decode("ascii")
    return {"cpu": encode(torch.get_rng_state()), "cuda": [encode(value) for value in torch.cuda.get_rng_state_all()]}


def restore_rng(torch: Any, state: dict[str, Any]) -> None:
    def decode(value: str) -> Any:
        return torch.tensor(list(base64.b64decode(value, validate=True)), dtype=torch.uint8)
    torch.set_rng_state(decode(state["cpu"]))
    require(len(state["cuda"]) == torch.cuda.device_count(), "CUDA RNG device-count mismatch")
    torch.cuda.set_rng_state_all([decode(value) for value in state["cuda"]])


def set_seed(torch: Any) -> None:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def synchronize(torch: Any) -> None:
    torch.cuda.synchronize()


def run(method: str, resume: bool) -> None:
    gate = preflight(method)
    output = RAW_DIR / f"{slug(method)}_timing.json"
    existing = read_json(output) if output.exists() else None
    if existing and existing.get("status") == "COMPLETE":
        print(json.dumps({"status": "ALREADY_COMPLETE_VALIDATED", "method": method, "sha256": sha256(output)}, indent=2))
        return
    require(existing is None or resume, f"Incomplete timing output exists; rerun {method} with --resume")

    load_start = time.perf_counter()
    torch, tokenizer, model = load_model()
    assets = load_method_assets(torch, method)
    synchronize(torch)
    load_seconds = time.perf_counter() - load_start
    subset = read_json(SUBSET)
    records = list(existing.get("records", [])) if existing else []
    sessions = list(existing.get("load_sessions", [])) if existing else []
    sessions.append({"load_seconds": load_seconds, "post_load_allocated_bytes": torch.cuda.memory_allocated(),
                     "post_load_reserved_bytes": torch.cuda.memory_reserved(), "resume": existing is not None})

    set_seed(torch)
    warmup_source = subset["records"][0]
    execute_prompt(torch, model, tokenizer, method, warmup_source, assets)
    synchronize(torch)
    torch.cuda.reset_peak_memory_stats()
    if existing and existing.get("rng_state"):
        restore_rng(torch, existing["rng_state"])

    expected_pairs = [(rep, int(row["prompt_id"])) for rep in REPETITIONS for row in subset["records"]]
    actual_pairs = [(int(row["repetition"]), int(row["prompt_id"])) for row in records]
    require(actual_pairs == expected_pairs[:len(actual_pairs)], "Timing checkpoint order mismatch")
    start_index = len(records)
    payload = existing or {
        "schema_version": "phase18_generation_timing_v1", "phase": 18, "method": method,
        "status": "IN_PROGRESS", "preflight": gate, "batch_size": 1,
        "warmup": {"prompt_id": int(subset["warmup_prompt_id"]), "count": 1, "excluded_from_measurements": True},
        "repetitions": 3, "records": [], "load_sessions": [], "rng_state": None,
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                        "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0)},
    }
    payload["load_sessions"] = sessions
    for flat_index in range(start_index, len(expected_pairs)):
        repetition, prompt_id = expected_pairs[flat_index]
        source_index = flat_index % 50
        if source_index == 0:
            set_seed(torch)
        source = subset["records"][source_index]
        require(int(source["prompt_id"]) == prompt_id, "Frozen prompt order mismatch")
        synchronize(torch)
        started = time.perf_counter()
        row = execute_prompt(torch, model, tokenizer, method, source, assets)
        synchronize(torch)
        row["latency_seconds"] = time.perf_counter() - started
        row.update({"method": method, "repetition": repetition, "source_index": source_index, "batch_size": 1})
        records.append(row)
        payload.update({"status": "IN_PROGRESS", "records": records, "completed_records": len(records),
                        "rng_state": capture_rng(torch), "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
        atomic_json(output, payload)
        print(f"{method}: repetition {repetition}/3 prompt {source_index + 1}/50 ({len(records)}/150)", flush=True)
    payload.update({"status": "COMPLETE", "completed_records": 150, "rng_state": None,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
    atomic_json(output, payload)
    print(json.dumps({"status": "COMPLETE", "method": method, "records": 150, "output_sha256": sha256(output)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        print(json.dumps(preflight(args.method), indent=2, sort_keys=True))
    else:
        run(args.method, args.resume)


if __name__ == "__main__":
    main()
