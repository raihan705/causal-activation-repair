import json, os, random, csv

OUT_DIR = "outputs/phase2"
SEED = 42
random.seed(SEED)

DERIVATION_CWES = ["CWE-120","CWE-125","CWE-787","CWE-190","CWE-476","CWE-89","CWE-79","CWE-327"]
EXPLORATORY_CWES = ["CWE-338"]

with open(f"{OUT_DIR}/baseline_dev_outputs.json", encoding="utf-8") as f:
    outputs = {r["prompt_id"]: r for r in json.load(f)}
with open(f"{OUT_DIR}/baseline_dev_icd.json", encoding="utf-8") as f:
    icd = {r["prompt_id"]: r for r in json.load(f)}

# Build stratified sample: ~8 per derivation CWE + exploratory, prioritize low-count CWEs
sample = []
per_cwe_target = {"CWE-89": 10, "CWE-79": 5, "CWE-190": 3, "CWE-79": 5,
                  "CWE-120": 5, "CWE-125": 3, "CWE-787": 5, "CWE-476": 5,
                  "CWE-327": 5, "CWE-338": 5}

seen_ids = set()
for cwe, target in per_cwe_target.items():
    # prompts where this CWE was detected
    detected = [pid for pid, r in icd.items()
                if any(f["cwe_id"] == cwe for f in r["findings"])
                and pid not in seen_ids]
    # also sample some where CWE was NOT detected but prompt is for this CWE
    not_detected = [pid for pid, r in icd.items()
                    if r.get("cwe_id") == cwe
                    and not any(f["cwe_id"] == cwe for f in r["findings"])
                    and pid not in seen_ids]
    chosen = random.sample(detected, min(target, len(detected)))
    chosen += random.sample(not_detected, min(3, len(not_detected)))
    for pid in chosen:
        seen_ids.add(pid)
        sample.append({
            "prompt_id": pid,
            "target_cwe": cwe,
            "detected_cwes": icd[pid]["vulnerable_cwes"],
            "is_vulnerable_scanner": icd[pid]["is_vulnerable"],
            "language": outputs[pid]["language"],
            "generated_code": outputs[pid]["generated_code"],
            "findings": icd[pid]["findings"],
        })

# Pad to 50 if needed with random safe samples
if len(sample) < 50:
    remaining = [pid for pid in icd if pid not in seen_ids and not icd[pid]["is_vulnerable"]]
    extra = random.sample(remaining, min(50 - len(sample), len(remaining)))
    for pid in extra:
        sample.append({
            "prompt_id": pid,
            "target_cwe": "none",
            "detected_cwes": [],
            "is_vulnerable_scanner": False,
            "language": outputs[pid]["language"],
            "generated_code": outputs[pid]["generated_code"],
            "findings": [],
        })

print(f"Audit sample size: {len(sample)}")

# Save for manual review
with open(f"{OUT_DIR}/audit_sample.json", "w", encoding="utf-8") as f:
    json.dump(sample, f, indent=2)

# Save CSV for easy manual annotation
with open(f"{OUT_DIR}/audit_sample.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["prompt_id","target_cwe","language","detected_cwes",
                "is_vulnerable_scanner","manual_label","notes"])
    for s in sample:
        w.writerow([s["prompt_id"], s["target_cwe"], s["language"],
                    ";".join(s["detected_cwes"]), s["is_vulnerable_scanner"], "", ""])

print(f"Saved audit_sample.json and audit_sample.csv to {OUT_DIR}")