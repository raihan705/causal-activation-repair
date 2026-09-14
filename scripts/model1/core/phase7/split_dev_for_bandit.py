"""
Phase 7 — Step 7.1: Split dev prompts into bandit train/val and filter offline dataset.

Inputs:
  data/cyberseceval/dev_prompts.json
  outputs/phase6/offline_bandit_dataset.jsonl

Outputs:
  outputs/phase7/bandit_train_split.json       — 80% prompt IDs
  outputs/phase7/bandit_val_split.json         — 20% prompt IDs
  outputs/phase7/offline_bandit_train.jsonl    — filtered records
  outputs/phase7/offline_bandit_val.jsonl      — filtered records
  outputs/phase7/split_dev_report.txt          — counts and verification
"""

import json
import random
import os
from collections import defaultdict

# ── Config ──────────────────────────────────────────────────────────────────
SEED          = 42
TRAIN_FRAC    = 0.80

DEV_PROMPTS   = "data/cyberseceval/dev_prompts.json"
OFFLINE_DS    = "outputs/phase6/offline_bandit_dataset.jsonl"

OUT_DIR       = "outputs/phase7"
TRAIN_SPLIT   = os.path.join(OUT_DIR, "bandit_train_split.json")
VAL_SPLIT     = os.path.join(OUT_DIR, "bandit_val_split.json")
TRAIN_JSONL   = os.path.join(OUT_DIR, "offline_bandit_train.jsonl")
VAL_JSONL     = os.path.join(OUT_DIR, "offline_bandit_val.jsonl")
REPORT        = os.path.join(OUT_DIR, "split_dev_report.txt")

os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Load dev prompt IDs ───────────────────────────────────────────────────
with open(DEV_PROMPTS) as f:
    dev_prompts = json.load(f)

# Support list-of-dicts or dict-of-dicts
if isinstance(dev_prompts, list):
    all_ids = [p["prompt_id"] for p in dev_prompts]
elif isinstance(dev_prompts, dict):
    all_ids = list(dev_prompts.keys())
else:
    raise ValueError("Unexpected dev_prompts format")

print(f"Total dev prompts: {len(all_ids)}")

# ── 2. Shuffle and split ─────────────────────────────────────────────────────
rng = random.Random(SEED)
shuffled = all_ids[:]
rng.shuffle(shuffled)

n_train = int(len(shuffled) * TRAIN_FRAC)
train_ids = set(shuffled[:n_train])
val_ids   = set(shuffled[n_train:])

print(f"Train prompts: {len(train_ids)}  |  Val prompts: {len(val_ids)}")
assert len(train_ids & val_ids) == 0, "Overlap between train and val!"

# ── 3. Save split ID files ───────────────────────────────────────────────────
with open(TRAIN_SPLIT, "w") as f:
    json.dump({"prompt_ids": sorted(train_ids), "count": len(train_ids), "seed": SEED}, f, indent=2)

with open(VAL_SPLIT, "w") as f:
    json.dump({"prompt_ids": sorted(val_ids), "count": len(val_ids), "seed": SEED}, f, indent=2)

# ── 4. Filter offline dataset ────────────────────────────────────────────────
train_records, val_records = [], []
seen_train, seen_val = defaultdict(int), defaultdict(int)
skipped = 0

with open(OFFLINE_DS) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        pid = rec["prompt_id"]
        if pid in train_ids:
            train_records.append(rec)
            seen_train[pid] += 1
        elif pid in val_ids:
            val_records.append(rec)
            seen_val[pid] += 1
        else:
            skipped += 1

print(f"Train records: {len(train_records)}  |  Val records: {len(val_records)}  |  Skipped: {skipped}")

with open(TRAIN_JSONL, "w") as f:
    for rec in train_records:
        f.write(json.dumps(rec) + "\n")

with open(VAL_JSONL, "w") as f:
    for rec in val_records:
        f.write(json.dumps(rec) + "\n")

# ── 5. Verify every split prompt has ≥1 record ──────────────────────────────
train_missing = train_ids - set(seen_train.keys())
val_missing   = val_ids   - set(seen_val.keys())

# ── 6. Report ────────────────────────────────────────────────────────────────
lines = [
    f"Phase 7 Step 7.1 — Dev Split Report",
    f"Seed: {SEED}",
    f"Total dev prompts:  {len(all_ids)}",
    f"Train prompts:      {len(train_ids)}",
    f"Val prompts:        {len(val_ids)}",
    f"Train records:      {len(train_records)}",
    f"Val records:        {len(val_records)}",
    f"Skipped records:    {skipped}",
    f"Train missing prompts (no records): {len(train_missing)}",
    f"Val missing prompts (no records):   {len(val_missing)}",
    f"Overlap check (must be 0): {len(train_ids & val_ids)}",
]

if train_missing:
    lines.append(f"  WARN train missing IDs: {sorted(train_missing)[:10]}")
if val_missing:
    lines.append(f"  WARN val missing IDs: {sorted(val_missing)[:10]}")

report = "\n".join(lines)
print("\n" + report)

with open(REPORT, "w") as f:
    f.write(report + "\n")

# ── 7. Checkpoint gate ───────────────────────────────────────────────────────
assert len(train_ids & val_ids) == 0,    "FAIL: train/val overlap"
assert len(train_missing) == 0,           f"FAIL: {len(train_missing)} train prompts have no records"
assert len(val_missing)   == 0,           f"FAIL: {len(val_missing)} val prompts have no records"
assert len(train_records) > 0,            "FAIL: empty train JSONL"
assert len(val_records)   > 0,            "FAIL: empty val JSONL"

print("\nStep 7.1 PASSED — splits written and verified.")