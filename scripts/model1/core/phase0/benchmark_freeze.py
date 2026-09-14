"""
Phase 0 - Step 0.2: Freeze benchmark versions and counts.
Output: configs/dataset_manifest.json
"""

import os
import json
import datetime

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CONFIG_DIR = os.path.join(ROOT_DIR, "configs")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs/phase0")

BENCHMARKS = {
    "CyberSecEval": {
        "source": "walledai/CyberSecEval",
        "config": "instruct",
        "revision": "main",
        "languages": ["c", "cpp", "python", "java", "javascript", "java", "rust", "csharp"],
        "target_languages": ["c", "cpp", "python", "java", "javascript"],
        "splits": {
            "dev": {"fraction": 0.70, "prompt_count": -1},
            "test": {"fraction": 0.30, "prompt_count": -1},
        },
        "split_seed": 42,
        "notes": "Splits are by language. Dev/test split by prompt_id with fixed seed.",
    },
    "InstructHumanEval": {
        "source": "bigcode/humanevalpack",
        "revision": "main",
        "splits": {"test": {"prompt_count": -1}},
        "notes": "Instruction-tuned HumanEval variant for pass@1 utility evaluation.",
    },
    "BigCodeBench": {
        "source": "bigcode/bigcodebench",
        "revision": "main",
        "splits": {"test": {"prompt_count": -1}},
        "notes": "Realistic programming tasks for utility evaluation.",
    },
    "MMLU": {
        "source": "cais/mmlu",
        "revision": "main",
        "slices": ["abstract_algebra", "college_computer_science",
                   "computer_security", "high_school_computer_science",
                   "machine_learning"],
        "splits": {"test": {"prompt_count": -1}},
        "notes": "Code-related MMLU slices only. Optional utility check.",
    },
}

VULNERABILITY_DATASETS = {
    "CVEFixes": {
        "source": "https://github.com/secureIT-project/CVEFixes",
        "revision": "main",
        "target_languages": ["C", "C++", "Python", "Java", "JavaScript"],
        "notes": "Primary vulnerability dataset. Cleaned with mono.",
    },
    "BigVul": {
        "source": "https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset",
        "revision": "main",
        "target_languages": ["C", "C++"],
        "notes": "Optional supplement if CVEFixes coverage insufficient after mono.",
    },
}

TARGET_CWES = {
    "derivation": ["CWE-338", "CWE-327", "CWE-120", "CWE-89",
                   "CWE-807", "CWE-502", "CWE-328", "CWE-611"],
    "held_out":   ["CWE-22", "CWE-290"],
}

SEEDS = {
    "cyberseceval_split": 42,
    "cve_split": 42,
    "bandit_split": 42,
    "balancing": 42,
}


def main():
    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    manifest = {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "benchmarks": BENCHMARKS,
        "vulnerability_datasets": VULNERABILITY_DATASETS,
        "target_cwes": TARGET_CWES,
        "seeds": SEEDS,
        "status": "frozen_phase0",
        "notes": "Prompt counts marked -1 will be updated in Phase 2 after datasets are downloaded.",
    }

    manifest_path = os.path.join(CONFIG_DIR, "dataset_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("=== Benchmark Freeze Summary ===")
    for name, info in BENCHMARKS.items():
        print(f"  {name}: source={info['source']}, splits={list(info['splits'].keys())}")
    print(f"\n  Derivation CWEs : {TARGET_CWES['derivation']}")
    print(f"  Held-out CWEs   : {TARGET_CWES['held_out']}")
    print(f"  Seeds           : {SEEDS}")
    print(f"\n[OK] dataset_manifest.json -> {manifest_path}")


if __name__ == "__main__":
    main()