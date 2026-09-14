"""
Phase 1 - Step 1.3
filter_mono_cleaned_pairs.py

Builds (vulnerable_code, fixed_code) pairs from mono-cleaned security_patch commits.
- Retains only mono_decision == security_patch
- Extracts code_before / code_after from file_change with ~20-40 lines of context
- Removes: missing snippets, <5 non-blank lines, duplicate (cve_id, file_path, hunk)
- Retains only target languages: C, C++, Python, Java, JavaScript

Output:
  data/processed/cve_pairs_filtered.csv
"""

import os
import json
import sqlite3
import hashlib
import datetime
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DB_PATH      = os.path.join(PROJECT_ROOT, "data/raw/cvefixes/CVEfixes.db")
REPORT_CSV   = os.path.join(PROJECT_ROOT, "data/mono_reports/mono_cleaning_report.csv")
OUT_DIR      = os.path.join(PROJECT_ROOT, "data/processed")
OUT_CSV      = os.path.join(OUT_DIR, "cve_pairs_filtered.csv")
LOG_PATH     = os.path.join(PROJECT_ROOT, "logs/filter_mono_cleaned_pairs.log")

TARGET_LANGS = {"C", "C++", "Python", "Java", "JavaScript"}

LANG_MAP = {
    "c": "C", "c++": "C++", "cpp": "C++",
    "python": "Python", "java": "Java",
    "javascript": "JavaScript", "js": "JavaScript"
}

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)


def log(msg: str):
    ts = datetime.datetime.utcnow().isoformat()
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def count_non_blank_lines(code: str) -> int:
    if not code:
        return 0
    return sum(1 for line in code.splitlines() if line.strip())


def truncate_with_context(code: str, max_lines: int = 40) -> str:
    """Keep up to max_lines lines of code (context window)."""
    if not code:
        return ""
    lines = code.splitlines()
    if len(lines) <= max_lines:
        return code
    # keep first 20 and last 20 lines as context
    half = max_lines // 2
    return "\n".join(lines[:half] + ["... [truncated] ..."] + lines[-half:])


def make_hunk_id(cve_id: str, file_path: str, code_before: str) -> str:
    """Stable dedup key from (cve_id, file_path, first 200 chars of code_before)."""
    raw = f"{cve_id}|{file_path}|{(code_before or '')[:200]}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def normalize_lang(raw: str) -> str:
    if not raw:
        return "Unknown"
    return LANG_MAP.get(raw.strip().lower(), raw.strip().capitalize())


def main():
    log("=== Step 1.3 -- Filter mono-Cleaned Pairs ===")

    # ── 1. Load mono cleaning report, keep only security_patch ────────────────
    log(f"Loading mono cleaning report from {REPORT_CSV} ...")
    df_mono = pd.read_csv(REPORT_CSV, encoding="utf-8", low_memory=False)
    df_security = df_mono[df_mono["mono_decision"] == "security_patch"].copy()
    log(f"Security patch rows: {len(df_security)}")

    # get set of (hash, cve_id) pairs that passed mono
    security_pairs = set(
        zip(df_security["hash"].astype(str), df_security["cve_id"].astype(str))
    )
    security_hashes = set(df_security["hash"].astype(str).unique())
    log(f"Unique security hashes: {len(security_hashes)}")

    # ── 2. Load file_change data for security hashes ──────────────────────────
    log("Loading file_change records from CVEFixes DB ...")
    con = sqlite3.connect(DB_PATH)

    placeholders = ",".join("?" * len(security_hashes))
    query = f"""
        SELECT
            fc.file_change_id,
            fc.hash,
            fc.filename,
            fc.old_path,
            fc.new_path,
            fc.code_before,
            fc.code_after,
            fc.programming_language,
            fc.num_lines_added,
            fc.num_lines_deleted
        FROM file_change fc
        WHERE fc.hash IN ({placeholders})
    """
    df_fc = pd.read_sql_query(query, con, params=list(security_hashes))
    con.close()
    log(f"File change rows for security hashes: {len(df_fc)}")

    # ── 3. Join with CVE and CWE info ─────────────────────────────────────────
    log("Joining with CVE/CWE info from mono report ...")
    # get unique (hash, cve_id, cwe_id) from security patches
    df_cve_cwe = df_security[["hash", "cve_id", "cwe_id"]].drop_duplicates()
    df_cve_cwe["hash"] = df_cve_cwe["hash"].astype(str)
    df_cve_cwe["cve_id"] = df_cve_cwe["cve_id"].astype(str)

    df_fc["hash"] = df_fc["hash"].astype(str)
    df_joined = df_fc.merge(df_cve_cwe, on="hash", how="inner")
    log(f"Joined rows (file_change x CVE/CWE): {len(df_joined)}")

    # ── 4. Build pairs with filtering ─────────────────────────────────────────
    log("Building and filtering vulnerability pairs ...")
    records = []
    skipped_missing  = 0
    skipped_short    = 0
    skipped_lang     = 0
    seen_hunk_ids    = set()
    skipped_dup      = 0

    for _, row in df_joined.iterrows():
        code_before = str(row["code_before"]) if pd.notna(row["code_before"]) else ""
        code_after  = str(row["code_after"])  if pd.notna(row["code_after"])  else ""

        # filter: missing code
        if not code_before.strip() or not code_after.strip():
            skipped_missing += 1
            continue

        # filter: < 5 non-blank lines
        if count_non_blank_lines(code_before) < 5 or count_non_blank_lines(code_after) < 5:
            skipped_short += 1
            continue

        # filter: language
        lang = normalize_lang(str(row["programming_language"]) if pd.notna(row["programming_language"]) else "")
        if lang not in TARGET_LANGS:
            skipped_lang += 1
            continue

        # filter: duplicate (cve_id, file_path, hunk)
        file_path = str(row["filename"]) if pd.notna(row["filename"]) else ""
        hunk_id   = make_hunk_id(str(row["cve_id"]), file_path, code_before)
        if hunk_id in seen_hunk_ids:
            skipped_dup += 1
            continue
        seen_hunk_ids.add(hunk_id)

        # apply context window
        code_before_ctx = truncate_with_context(code_before, max_lines=40)
        code_after_ctx  = truncate_with_context(code_after,  max_lines=40)

        records.append({
            "pair_id":          hunk_id,
            "cve_id":           str(row["cve_id"]),
            "cwe_id":           str(row["cwe_id"]) if pd.notna(row["cwe_id"]) else "Unknown",
            "file_change_id":   str(row["file_change_id"]),
            "hash":             str(row["hash"]),
            "filename":         file_path,
            "language":         lang,
            "vulnerable_code":  code_before_ctx,
            "fixed_code":       code_after_ctx,
            "num_lines_added":  row["num_lines_added"],
            "num_lines_deleted":row["num_lines_deleted"],
            "mono_decision":    "security_patch",
            "source":           "cvefixes",
            "is_augmented":     False,
            "seed_cve_id":      ""
        })

    df_out = pd.DataFrame(records)
    log(f"Pairs built: {len(df_out)}")
    log(f"Skipped - missing code: {skipped_missing}")
    log(f"Skipped - too short:    {skipped_short}")
    log(f"Skipped - wrong lang:   {skipped_lang}")
    log(f"Skipped - duplicate:    {skipped_dup}")

    # ── 5. Save ───────────────────────────────────────────────────────────────
    df_out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    log(f"Saved {len(df_out)} pairs to {OUT_CSV}")

    # ── 6. Stats ──────────────────────────────────────────────────────────────
    log("Language distribution:")
    log(str(df_out["language"].value_counts().to_dict()))
    log("CWE distribution (top 20):")
    log(str(df_out["cwe_id"].value_counts().head(20).to_dict()))

    # ── 7. Checkpoint ─────────────────────────────────────────────────────────
    if len(df_out) == 0:
        log("CHECKPOINT FAIL: No pairs produced.")
        return False

    log("CHECKPOINT PASS: cve_pairs_filtered.csv produced.")
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)