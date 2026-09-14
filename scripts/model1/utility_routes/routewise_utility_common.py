#!/usr/bin/env python3
"""Shared contracts for the Model 1 route-wise active B* utility experiment."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT = ROOT / "revision/model1/routewise_bstar_utility"
SCRIPTS = EXPERIMENT / "scripts"
OUTPUTS = EXPERIMENT / "outputs"
PROTOCOL = OUTPUTS / "routewise_utility_protocol.json"
APPROVAL = OUTPUTS / "routewise_utility_execution_approval.json"

PHASE17_INPUT = ROOT / "revision/model1/phase17/outputs/phase17_input_manifest.json"
BSTAR_CONFIG = ROOT / "configs/bstar_config.json"

EXPECTED_PHASE17_INPUT_SHA256 = "eb53962ce08a6d5dcab1a382e9b62ff867d1e29f9dda6711c23de7a10e6714d1"
EXPECTED_BSTAR_CONFIG_SHA256 = "f4f5d4d6a697aa14f654763f31e52cd852ce87b9f284276edd4f784604ea36e6"

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
MODEL_CACHE_REPO = "models--meta-llama--Meta-Llama-3.1-8B-Instruct"
SAE_RELEASE = "llama_scope_lxr_8x"
SAE_REVISION = "8dbc1d85edfced43081c03c38b05514dbab1368b"
SAE_CACHE_REPO = "models--fnlp--Llama3_1-8B-Base-LXR-8x"
SAE_CACHE_DIRS = {
    16: "Llama3_1-8B-Base-L16R-8x",
    19: "Llama3_1-8B-Base-L19R-8x",
    23: "Llama3_1-8B-Base-L23R-8x",
}

SEED = 42
ALPHA = 40.0
CHECKPOINT_INTERVAL = 25
EXPECTED_COUNTS = {"humaneval": 164, "bigcodebench": 1140, "mmlu": 412}
BENCHMARK_ORDER = ("humaneval", "bigcodebench", "mmlu")

EXPECTED_ROUTES = {
    "CWE-120": {"layer": 19, "feature": 14193},
    "CWE-787": {"layer": 19, "feature": 1515},
    "CWE-190": {"layer": 19, "feature": 16897},
    "CWE-327": {"layer": 23, "feature": 14449},
    "CWE-89": {"layer": 23, "feature": 1652},
    "CWE-338": {"layer": 23, "feature": 7533},
    "CWE-79": {"layer": 16, "feature": 9816},
    "CWE-125": {"layer": 23, "feature": 16655},
    "CWE-476": {"layer": 23, "feature": 18397},
}


class ExperimentError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ExperimentError(message)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_text(payload)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def route_slug(cwe_id: str, route: dict[str, Any]) -> str:
    return f"{cwe_id.lower().replace('-', '')}_l{int(route['layer'])}_f{int(route['feature'])}_a40_seed42"


def route_output(cwe_id: str, route: dict[str, Any]) -> Path:
    return OUTPUTS / f"route_{route_slug(cwe_id, route)}_outputs.json"


def route_checkpoint(cwe_id: str, route: dict[str, Any]) -> Path:
    return OUTPUTS / f"route_{route_slug(cwe_id, route)}_checkpoint.json"


def route_run_manifest(cwe_id: str, route: dict[str, Any]) -> Path:
    return OUTPUTS / f"route_{route_slug(cwe_id, route)}_run_manifest.json"


def baseline_output() -> Path:
    return OUTPUTS / "paired_b0_seed42_outputs.json"


def baseline_checkpoint() -> Path:
    return OUTPUTS / "paired_b0_seed42_checkpoint.json"


def baseline_run_manifest() -> Path:
    return OUTPUTS / "paired_b0_seed42_run_manifest.json"


def validate_sources() -> tuple[dict[str, Any], dict[str, Any]]:
    require(PHASE17_INPUT.is_file(), f"Missing input source: {relative(PHASE17_INPUT)}")
    require(BSTAR_CONFIG.is_file(), f"Missing B* config: {relative(BSTAR_CONFIG)}")
    require(sha256_path(PHASE17_INPUT) == EXPECTED_PHASE17_INPUT_SHA256,
            "Phase 17 input-manifest hash mismatch")
    require(sha256_path(BSTAR_CONFIG) == EXPECTED_BSTAR_CONFIG_SHA256,
            "B* config hash mismatch")
    source = read_json(PHASE17_INPUT)
    config = read_json(BSTAR_CONFIG)
    require(config.get("bstar") == "B2_a40" and float(config.get("alpha")) == ALPHA,
            "B* identity/alpha mismatch")
    require(config.get("feature_map") == EXPECTED_ROUTES, "B* route map mismatch")
    return source, config


def ordered_tasks(source: dict[str, Any]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for benchmark in BENCHMARK_ORDER:
        population = source.get("populations", {}).get(benchmark, {})
        rows = population.get("records")
        require(isinstance(rows, list) and len(rows) == EXPECTED_COUNTS[benchmark],
                f"{benchmark} population mismatch")
        for row in rows:
            tasks.append({"benchmark": benchmark, **row})
    require(len(tasks) == sum(EXPECTED_COUNTS.values()), "Global task count mismatch")
    ids = [str(row["stable_id"]) for row in tasks]
    require(len(set(ids)) == len(ids), "Utility stable IDs are not globally unique")
    return tasks


def load_protocol() -> dict[str, Any]:
    source, config = validate_sources()
    require(PROTOCOL.is_file(), "Protocol is absent; run prepare_routewise_utility.py")
    protocol = read_json(PROTOCOL)
    require(protocol.get("schema_version") == "routewise_bstar_utility_protocol_v1",
            "Protocol schema mismatch")
    require(protocol.get("status") == "FROZEN", "Protocol is not frozen")
    require(protocol.get("source_manifest_sha256") == EXPECTED_PHASE17_INPUT_SHA256,
            "Protocol source-manifest binding mismatch")
    require(protocol.get("bstar_config_sha256") == EXPECTED_BSTAR_CONFIG_SHA256,
            "Protocol B* config binding mismatch")
    require(protocol.get("routes") == config["feature_map"], "Protocol route map mismatch")
    require(protocol.get("task_ids_sha256") == canonical_sha256(
        [str(row["stable_id"]) for row in ordered_tasks(source)]), "Protocol task-ID hash mismatch")
    return protocol


def validate_approval(runner: Path, common: Path) -> dict[str, Any]:
    protocol = load_protocol()
    require(APPROVAL.is_file(), "Execution approval artifact is absent")
    approval = read_json(APPROVAL)
    require(approval.get("status") == "APPROVED_FOR_EXECUTION", "Experiment is not approved")
    require(approval.get("protocol_sha256") == sha256_path(PROTOCOL),
            "Approval protocol hash mismatch")
    hashes = approval.get("execution_hashes", {})
    require(hashes.get(relative(runner)) == sha256_path(runner), "Approved runner hash mismatch")
    require(hashes.get(relative(common)) == sha256_path(common), "Approved common-helper hash mismatch")
    require(approval.get("routes") == protocol["routes"], "Approval route map mismatch")
    return approval


def huggingface_hub_cache() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser().resolve()
    if os.environ.get("HF_HOME"):
        return (Path(os.environ["HF_HOME"]).expanduser() / "hub").resolve()
    return (Path.home() / ".cache/huggingface/hub").resolve()


def validate_cache(require_sae: bool = True) -> dict[str, Any]:
    hub = huggingface_hub_cache()
    model_snapshot = hub / MODEL_CACHE_REPO / "snapshots" / MODEL_REVISION
    require(model_snapshot.is_dir(), f"Frozen model snapshot is missing: {model_snapshot}")
    model_ref = hub / MODEL_CACHE_REPO / "refs/main"
    require(model_ref.is_file() and model_ref.read_text(encoding="utf-8").strip() == MODEL_REVISION,
            "Model cache main ref differs from the frozen snapshot")
    result: dict[str, Any] = {"hub": str(hub), "model_snapshot": str(model_snapshot)}
    if require_sae:
        sae_snapshot = hub / SAE_CACHE_REPO / "snapshots" / SAE_REVISION
        require(sae_snapshot.is_dir(), f"Frozen SAE snapshot is missing: {sae_snapshot}")
        sae_ref = hub / SAE_CACHE_REPO / "refs/main"
        require(sae_ref.is_file() and sae_ref.read_text(encoding="utf-8").strip() == SAE_REVISION,
                "SAE cache main ref differs from the frozen snapshot")
        for layer, directory in SAE_CACHE_DIRS.items():
            require((sae_snapshot / directory / "hyperparams.json").is_file(),
                    f"Missing layer-{layer} SAE hyperparameters")
            require((sae_snapshot / directory / "checkpoints/final.safetensors").is_file(),
                    f"Missing layer-{layer} SAE weights")
        result["sae_snapshot"] = str(sae_snapshot)
    return result


def capture_rng_state(torch_module: Any) -> dict[str, Any]:
    def encode(tensor: Any) -> str:
        return base64.b64encode(bytes(tensor.detach().cpu().tolist())).decode("ascii")
    cuda_states = torch_module.cuda.get_rng_state_all()
    return {
        "torch_cpu_b64": encode(torch_module.get_rng_state()),
        "torch_cuda_all_b64": [encode(state) for state in cuda_states],
        "cuda_device_count": len(cuda_states),
    }


def restore_rng_state(torch_module: Any, payload: dict[str, Any]) -> None:
    require(torch_module.cuda.device_count() == int(payload.get("cuda_device_count", -1)),
            "CUDA device count differs from the paired B0 RNG schedule")

    def decode(value: str) -> Any:
        raw = base64.b64decode(value, validate=True)
        require(bool(raw), "Empty RNG-state payload")
        return torch_module.tensor(list(raw), dtype=torch_module.uint8)

    torch_module.set_rng_state(decode(payload["torch_cpu_b64"]))
    torch_module.cuda.set_rng_state_all([decode(value) for value in payload["torch_cuda_all_b64"]])


def render_task(tokenizer: Any, task: dict[str, Any]) -> tuple[str, int, int, int | list[int] | None]:
    benchmark = task["benchmark"]
    if benchmark == "humaneval":
        stop_strings = ["\ndef ", "\nclass ", "\n#", "\nif __name__"]
        stop_ids = [ids[0] for value in stop_strings
                    if (ids := tokenizer.encode(value, add_special_tokens=False))]
        return task["prompt"], 2048, 256, sorted(set(stop_ids + [tokenizer.eos_token_id]))
    if benchmark == "bigcodebench":
        messages = [
            {"role": "system", "content": "You are an expert programmer. Complete the following coding task. Return only the code, no explanation."},
            {"role": "user", "content": task["prompt"]},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), 2048, 512, None
    labels = "ABCD"
    options = "\n".join(f"{labels[index]}. {choice}" for index, choice in enumerate(task["choices"]))
    user = f"Question: {task['question']}\n\n{options}\n\nAnswer with only the letter (A, B, C, or D)."
    messages = [
        {"role": "system", "content": "You are a knowledgeable assistant."},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), 2048, 8, None


def mmlu_prediction(response: str) -> str:
    for character in response.strip().upper():
        if character in "ABCD":
            return character
    return "X"


def empty_status(text: str) -> str:
    if text == "":
        return "STRICT_EMPTY"
    if not text.strip():
        return "WHITESPACE_ONLY"
    return "NONEMPTY"

