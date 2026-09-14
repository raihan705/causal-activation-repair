import json, os, random
from collections import defaultdict

SEED = 42
DEV_RATIO = 0.70
SRC = "data/purplellama_repo/CybersecurityBenchmarks/datasets/instruct/instruct.json"
OUT_DIR = "data/cyberseceval"
CFG_DIR = "configs"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CFG_DIR, exist_ok=True)

with open(SRC, encoding="utf-8") as f:
    data = json.load(f)

random.seed(SEED)
random.shuffle(data)

split_idx = int(len(data) * DEV_RATIO)
dev = data[:split_idx]
test = data[split_idx:]

with open(f"{OUT_DIR}/dev_prompts.json", "w") as f:
    json.dump(dev, f, indent=2)
with open(f"{OUT_DIR}/test_prompts.json", "w") as f:
    json.dump(test, f, indent=2)

# CWE counts per split
def cwe_counts(split):
    c = defaultdict(int)
    for r in split:
        c[r["cwe_identifier"]] += 1
    return dict(c)

manifest = {
    "cyberseceval_source": SRC,
    "split_seed": SEED,
    "dev_ratio": DEV_RATIO,
    "total_prompts": len(data),
    "dev_count": len(dev),
    "test_count": len(test),
    "dev_cwe_counts": cwe_counts(dev),
    "test_cwe_counts": cwe_counts(test),
    "dev_prompt_ids": [r["prompt_id"] for r in dev],
    "test_prompt_ids": [r["prompt_id"] for r in test]
}

with open(f"{CFG_DIR}/benchmark_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Dev: {len(dev)}, Test: {len(test)}")
print(f"Dev CWEs: {cwe_counts(dev)}")