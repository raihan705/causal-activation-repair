import json, os
from collections import defaultdict

OUT_DIR = "outputs/phase2"
os.makedirs(OUT_DIR, exist_ok=True)

DERIVATION_CWES = {"CWE-120","CWE-125","CWE-787","CWE-190","CWE-476","CWE-89","CWE-79","CWE-327"}
EXPLORATORY_CWES = {"CWE-338"}
HELDOUT_CWES = {"CWE-22","CWE-290"}

def compute_metrics(icd_file, label):
    with open(icd_file, encoding="utf-8") as f:
        records = json.load(f)

    total = len(records)
    vuln_count = sum(1 for r in records if r["is_vulnerable"])
    vuln_rate = vuln_count / total

    # per-CWE counts
    cwe_vuln_count = defaultdict(int)
    cwe_finding_count = defaultdict(int)
    for r in records:
        detected = set()
        for f in r["findings"]:
            cwe_finding_count[f["cwe_id"]] += 1
            detected.add(f["cwe_id"])
        for cwe in detected:
            cwe_vuln_count[cwe] += 1

    # co-occurrence matrix (simple)
    co_occur = defaultdict(int)
    for r in records:
        cwes = list({f["cwe_id"] for f in r["findings"]})
        for i in range(len(cwes)):
            for j in range(i+1, len(cwes)):
                key = tuple(sorted([cwes[i], cwes[j]]))
                co_occur[key] += 1

    metrics = {
        "label": label,
        "total_prompts": total,
        "vulnerable_prompts": vuln_count,
        "vulnerability_rate": round(vuln_rate, 4),
        "vulnerability_density": round(sum(len(r["findings"]) for r in records) / total, 4),
        "per_cwe": {}
    }

    all_cwes = DERIVATION_CWES | EXPLORATORY_CWES | HELDOUT_CWES
    for cwe in sorted(all_cwes):
        category = "derivation" if cwe in DERIVATION_CWES else \
                   "exploratory" if cwe in EXPLORATORY_CWES else "held_out"
        metrics["per_cwe"][cwe] = {
            "category": category,
            "vulnerable_prompts": cwe_vuln_count.get(cwe, 0),
            "total_findings": cwe_finding_count.get(cwe, 0),
        }

    metrics["co_occurrence"] = {str(k): v for k, v in co_occur.items()}
    return metrics, records

b0_metrics, b0_records = compute_metrics(f"{OUT_DIR}/baseline_dev_icd.json", "B0")
b1_metrics, b1_records = compute_metrics(f"{OUT_DIR}/zeroshot_dev_icd.json", "B1")

# VRR for B1 vs B0
b0_vuln = b0_metrics["vulnerable_prompts"]
b1_vuln = b1_metrics["vulnerable_prompts"]
b1_vrr = 1 - b1_vuln / b0_vuln if b0_vuln > 0 else None
b1_metrics["VRR_vs_B0"] = round(b1_vrr, 4) if b1_vrr is not None else None

# Save metrics
with open(f"{OUT_DIR}/baseline_metrics.json", "w") as f:
    json.dump({"B0": b0_metrics, "B1": b1_metrics}, f, indent=2)

# Save CSV summary
import csv
with open(f"{OUT_DIR}/baseline_metrics.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method","total_prompts","vulnerable_prompts","vulnerability_rate","vulnerability_density","VRR_vs_B0"])
    w.writerow(["B0", b0_metrics["total_prompts"], b0_metrics["vulnerable_prompts"],
                b0_metrics["vulnerability_rate"], b0_metrics["vulnerability_density"], 0.0])
    w.writerow(["B1", b1_metrics["total_prompts"], b1_metrics["vulnerable_prompts"],
                b1_metrics["vulnerability_rate"], b1_metrics["vulnerability_density"],
                b1_metrics["VRR_vs_B0"]])

# Save per-CWE CSV
with open(f"{OUT_DIR}/baseline_cwe_distribution.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["cwe_id","category","B0_vulnerable_prompts","B0_total_findings","B1_vulnerable_prompts","B1_total_findings"])
    for cwe in sorted(DERIVATION_CWES | EXPLORATORY_CWES | HELDOUT_CWES):
        b0c = b0_metrics["per_cwe"][cwe]
        b1c = b1_metrics["per_cwe"][cwe]
        w.writerow([cwe, b0c["category"], b0c["vulnerable_prompts"], b0c["total_findings"],
                    b1c["vulnerable_prompts"], b1c["total_findings"]])

# Build safe/unsafe partitions
def build_partitions(records, icd_label):
    unsafe_ids = [r["prompt_id"] for r in records if r["is_vulnerable"]]
    safe_ids = [r["prompt_id"] for r in records if not r["is_vulnerable"]]

    with open(f"{OUT_DIR}/unsafe_dev_ids_{icd_label}.json", "w") as f:
        json.dump(unsafe_ids, f, indent=2)
    with open(f"{OUT_DIR}/safe_dev_ids_{icd_label}.json", "w") as f:
        json.dump(safe_ids, f, indent=2)

    # per-CWE partitions (derivation only for training use)
    all_cwes = DERIVATION_CWES | EXPLORATORY_CWES | HELDOUT_CWES
    for cwe in all_cwes:
        unsafe_cwe = [r["prompt_id"] for r in records
                      if any(f["cwe_id"] == cwe for f in r["findings"])]
        safe_cwe = [r["prompt_id"] for r in records
                    if not any(f["cwe_id"] == cwe for f in r["findings"])]
        suffix = "exploratory" if cwe in EXPLORATORY_CWES else \
                 "heldout" if cwe in HELDOUT_CWES else ""
        tag = f"{cwe}_{suffix}" if suffix else cwe
        with open(f"{OUT_DIR}/unsafe_dev_ids_{tag}.json", "w") as f:
            json.dump(unsafe_cwe, f, indent=2)
        with open(f"{OUT_DIR}/safe_dev_ids_{tag}.json", "w") as f:
            json.dump(safe_cwe, f, indent=2)

build_partitions(b0_records, "b0")
build_partitions(b1_records, "b1")

print("B0:", b0_metrics["vulnerable_prompts"], "/", b0_metrics["total_prompts"],
      "rate:", b0_metrics["vulnerability_rate"])
print("B1:", b1_metrics["vulnerable_prompts"], "/", b1_metrics["total_prompts"],
      "rate:", b1_metrics["vulnerability_rate"], "VRR:", b1_metrics["VRR_vs_B0"])
print("Per-CWE B0:", {c: b0_metrics["per_cwe"][c]["vulnerable_prompts"]
                       for c in sorted(DERIVATION_CWES | EXPLORATORY_CWES)})