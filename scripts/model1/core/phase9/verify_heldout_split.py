"""
Phase 9 Step 9.1 — Verify held-out split integrity.
Outputs: test_split_verification.txt
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd()
CONFIG_DIR   = PROJECT_ROOT / "configs"
DATA_DIR     = PROJECT_ROOT / "data" / "cyberseceval"
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "phase9"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Load manifests ---
with open(CONFIG_DIR / "benchmark_manifest.json") as f:
    bench_manifest = json.load(f)
with open(CONFIG_DIR / "split_manifest.json") as f:
    split_manifest = json.load(f)

# --- Load splits ---
with open(DATA_DIR / "dev_prompts.json") as f:
    dev_prompts = json.load(f)
with open(DATA_DIR / "test_prompts.json") as f:
    test_prompts = json.load(f)

dev_ids  = {p["prompt_id"] for p in dev_prompts}
test_ids = {p["prompt_id"] for p in test_prompts}

lines = []
passed = True

# Check 1: no overlap
overlap = dev_ids & test_ids
if overlap:
    lines.append(f"FAIL: {len(overlap)} prompt IDs appear in both dev and test: {sorted(overlap)[:10]}")
    passed = False
else:
    lines.append(f"PASS: No dev/test prompt ID overlap.")

# Check 2: counts match manifest
expected_dev  = bench_manifest.get("dev_count",  len(dev_prompts))
expected_test = bench_manifest.get("test_count", len(test_prompts))
if len(dev_prompts) != expected_dev:
    lines.append(f"FAIL: dev count {len(dev_prompts)} != manifest {expected_dev}")
    passed = False
else:
    lines.append(f"PASS: dev count = {len(dev_prompts)}")

if len(test_prompts) != expected_test:
    lines.append(f"FAIL: test count {len(test_prompts)} != manifest {expected_test}")
    passed = False
else:
    lines.append(f"PASS: test count = {len(test_prompts)}")

# Check 3: per-CWE distribution in test
from collections import Counter
test_cwe_dist = Counter(p.get("cwe_id", "unknown") for p in test_prompts)
lines.append(f"\nTest CWE distribution:")
for cwe, cnt in sorted(test_cwe_dist.items()):
    lines.append(f"  {cwe}: {cnt}")

# Check 4: bandit train/val IDs don't bleed into test
for split_key in ["bandit_train_ids", "bandit_val_ids"]:
    if split_key in split_manifest:
        bleed = set(split_manifest[split_key]) & test_ids
        if bleed:
            lines.append(f"FAIL: {len(bleed)} {split_key} IDs appear in test.")
            passed = False
        else:
            lines.append(f"PASS: {split_key} has no test overlap.")

# Check 5: held-out CWEs present in test
HELD_OUT_CWES = {"CWE-22", "CWE-290"}
test_cwes_present = set(test_cwe_dist.keys())
for hcwe in HELD_OUT_CWES:
    if hcwe in test_cwes_present:
        lines.append(f"PASS: Held-out CWE {hcwe} present in test ({test_cwe_dist[hcwe]} prompts).")
    else:
        lines.append(f"INFO: Held-out CWE {hcwe} not present in test (expected if CyberSecEval lacks it).")

lines.append(f"\nOverall: {'PASS — held-out split is clean.' if passed else 'FAIL — contamination detected. Do not proceed.'}")

report = "\n".join(lines)
print(report)

out_path = OUTPUT_DIR / "test_split_verification.txt"
out_path.write_text(report)
print(f"\nSaved: {out_path}")

if not passed:
    sys.exit(1)