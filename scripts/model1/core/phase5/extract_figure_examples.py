import json
from pathlib import Path

# Best candidates: short enough to fit in figure
TARGET_IDS = [17, 207, 44, 1615]

b0_out  = {r['prompt_id']: r for r in json.loads(Path('outputs/phase2/baseline_dev_outputs.json').read_text(encoding='utf-8'))}
bst_out = {r['prompt_id']: r for r in json.loads(Path('outputs/phase5/thea_static_dev_outputs_a40.json').read_text(encoding='utf-8'))}
prompts = {r['prompt_id']: r for r in json.loads(Path('data/cyberseceval/dev_prompts.json').read_text(encoding='utf-8'))}

for pid in TARGET_IDS:
    p = prompts[pid]
    print(f"=== prompt_id={pid} | CWE={p['cwe_identifier']} | LANG={p['language']} ===")
    print(f"PROMPT: {p['test_case_prompt'][:300]}")
    print(f"--- B0 CODE ({len(b0_out[pid]['generated_code'])} chars) ---")
    print(b0_out[pid]['generated_code'][:1000])
    print(f"--- B* CODE ({len(bst_out[pid]['generated_code'])} chars) ---")
    print(bst_out[pid]['generated_code'][:1000])
    print()