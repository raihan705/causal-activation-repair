#!/usr/bin/env python3
"""Compute frozen same-seed B0/B* prompt-level consistency evidence."""

from __future__ import annotations

from phase16_analysis_common import OUT, baseline_sets, load_all, valid, write_csv


def main() -> int:
    loaded = load_all()
    rows = []
    for prompt_id in loaded["universe"]:
        row = {"prompt_id": prompt_id,
               "cwe_id": loaded["data"][("B0", 42)]["scan_map"][prompt_id]["cwe_id"]}
        vulnerable_count = safe_count = repair_count = corruption_count = 0
        for seed in (42, 43, 44):
            vulnerable, safe = baseline_sets(loaded, seed)
            bstar = loaded["data"][("B*", seed)]
            is_vulnerable = prompt_id in vulnerable
            is_safe = prompt_id in safe
            repaired = is_vulnerable and valid(bstar["generation_map"][prompt_id]) and not bstar["scan_map"][prompt_id]["skipped"] and not bstar["scan_map"][prompt_id]["is_vulnerable"]
            corrupted = is_safe and not bstar["scan_map"][prompt_id]["skipped"] and bstar["scan_map"][prompt_id]["is_vulnerable"]
            row[f"b0_vulnerable_seed{seed}"] = is_vulnerable
            row[f"bstar_repaired_seed{seed}"] = repaired if is_vulnerable else "NOT_ELIGIBLE"
            row[f"b0_safe_seed{seed}"] = is_safe
            row[f"bstar_corrupted_seed{seed}"] = corrupted if is_safe else "NOT_ELIGIBLE"
            vulnerable_count += int(is_vulnerable)
            safe_count += int(is_safe)
            repair_count += int(repaired)
            corruption_count += int(corrupted)
        row.update({
            "b0_vulnerable_seed_count": vulnerable_count,
            "bstar_repair_eligible_seed_count": vulnerable_count,
            "bstar_repair_count": repair_count,
            "b0_safe_seed_count": safe_count,
            "bstar_corruption_eligible_seed_count": safe_count,
            "bstar_corruption_count": corruption_count,
        })
        rows.append(row)
    write_csv(OUT / "phase16_prompt_consistency.csv", list(rows[0]), rows)
    print(f"COMPLETE: {len(rows)} scanner-eligible prompt rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
