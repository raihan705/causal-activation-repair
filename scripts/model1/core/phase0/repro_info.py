"""
Phase 0 - Step 0.1: Record environment information.
Outputs: outputs/phase0/repro_info.txt
         outputs/phase0/pip_freeze.txt
"""

import os
import sys
import platform
import subprocess
import datetime

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../../outputs/phase0")


def get_gpu_info():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception as e:
        return f"nvidia-smi unavailable: {e}"


def get_cuda_version():
    try:
        import torch
        return torch.version.cuda or "N/A"
    except Exception:
        return "torch not installed"


def get_library_versions():
    libs = ["torch", "transformers", "sae_lens", "numpy", "accelerate"]
    versions = {}
    for lib in libs:
        try:
            mod = __import__(lib.replace("-", "_"))
            versions[lib] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[lib] = "not installed"

    try:
        result = subprocess.run(
            ["semgrep", "--version"], capture_output=True, text=True
        )
        versions["semgrep"] = result.stdout.strip()
    except Exception:
        versions["semgrep"] = "not installed or not in PATH"

    return versions


def get_pip_freeze():
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True, check=True
        )
        return result.stdout
    except Exception as e:
        return f"pip freeze failed: {e}"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    gpu_info = get_gpu_info()
    cuda_version = get_cuda_version()
    lib_versions = get_library_versions()
    pip_freeze = get_pip_freeze()

    repro_lines = [
        f"Timestamp: {timestamp}",
        f"OS: {platform.platform()}",
        f"Python: {sys.version}",
        f"GPU: {gpu_info}",
        f"CUDA: {cuda_version}",
        "",
        "Library Versions:",
    ]
    for lib, ver in lib_versions.items():
        repro_lines.append(f"  {lib}: {ver}")

    repro_text = "\n".join(repro_lines)

    repro_path = os.path.join(OUTPUT_DIR, "repro_info.txt")
    pip_path = os.path.join(OUTPUT_DIR, "pip_freeze.txt")

    with open(repro_path, "w") as f:
        f.write(repro_text)

    with open(pip_path, "w") as f:
        f.write(pip_freeze)

    print(repro_text)
    print(f"\n[OK] repro_info.txt -> {repro_path}")
    print(f"[OK] pip_freeze.txt -> {pip_path}")


if __name__ == "__main__":
    main()