import csv
from collections import defaultdict

counts = defaultdict(int)
cwe_fp = defaultdict(int)
cwe_fn = defaultdict(int)

with open('outputs/phase2/audit_sample.csv', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        label = row['manual_label'].strip().upper()
        cwe = row['target_cwe'].strip()
        if label in ('TP','FP','FN','TN'):
            counts[label] += 1
            if label == 'FP':
                cwe_fp[cwe] += 1
            if label == 'FN':
                cwe_fn[cwe] += 1

total = sum(counts.values())
fpr = counts['FP'] / (counts['FP'] + counts['TN']) if (counts['FP'] + counts['TN']) > 0 else None
print(f"TP={counts['TP']}, FP={counts['FP']}, FN={counts['FN']}, TN={counts['TN']}")
print(f"Total labeled: {total}")
print(f"FPR: {fpr:.3f}" if fpr is not None else "FPR: N/A")
print(f"FP by CWE: {dict(cwe_fp)}")
print(f"FN by CWE: {dict(cwe_fn)}")