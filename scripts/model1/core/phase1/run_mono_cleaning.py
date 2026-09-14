"""
Phase 1 - Step 1.2
run_mono_cleaning.py

Classifies each CVEFixes commit as:
  - security_patch   : confirmed via MonoLens conf_0.9 OR heuristic
  - non_security_patch: heuristic indicates non-security
  - undecidable      : insufficient signal

Strategy:
  1. Load MonoLens conf_0.9 CVE IDs -> mono_decision = security_patch (verified)
  2. For remaining commits linked to a CVE -> heuristic classification
  3. Commits not linked to any CVE -> non_security_patch

Outputs:
  data/mono_reports/mono_cleaning_report.csv
  data/mono_reports/mono_cleaning_summary.json
"""

import os
import re
import json
import sqlite3
import datetime
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DB_PATH        = os.path.join(PROJECT_ROOT, "data/raw/cvefixes/CVEfixes.db")
MONOLENS_CSV   = os.path.join(PROJECT_ROOT, "data/raw/mono/MonoLens/conf_0.9/all_platforms_cves.csv")
MONO_STATS     = os.path.join(PROJECT_ROOT, "data/raw/mono/MonoLens/conf_0.9/overall_stats.json")
REPORT_DIR     = os.path.join(PROJECT_ROOT, "data/mono_reports")
REPORT_CSV     = os.path.join(REPORT_DIR, "mono_cleaning_report.csv")
SUMMARY_JSON   = os.path.join(REPORT_DIR, "mono_cleaning_summary.json")
LOG_PATH       = os.path.join(PROJECT_ROOT, "logs/run_mono_cleaning.log")

TARGET_LANGS   = {"C", "C++", "Python", "Java", "JavaScript"}

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)


def log(msg: str):
    ts = datetime.datetime.utcnow().isoformat()
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ── security-related keywords for heuristic ───────────────────────────────────
SECURITY_KEYWORDS = [
    "cve-", "vulnerability", "vuln", "security", "exploit", "overflow",
    "injection", "xss", "csrf", "rce", "sqli", "buffer", "heap", "stack",
    "uaf", "use-after-free", "null deref", "null pointer", "integer overflow",
    "out-of-bounds", "oob", "memory corruption", "privilege escalation",
    "arbitrary code", "denial of service", "dos", "authentication bypass",
    "insecure", "crypto", "cipher", "hash", "random", "entropy", "prng",
    "deserialization", "path traversal", "directory traversal", "fix",
    "patch", "sanitize", "validate", "hardening"
]

NON_SECURITY_KEYWORDS = [
    "refactor", "reformat", "style", "typo", "whitespace", "indent",
    "comment", "doc", "documentation", "test", "unittest", "benchmark",
    "performance", "optimize", "cleanup", "clean up", "rename", "move file",
    "add feature", "new feature", "release", "version bump", "changelog",
    "merge branch", "revert"
]


def heuristic_classify(commit_msg: str, cve_linked: bool) -> str:
    """
    Returns 'security_patch', 'non_security_patch', or 'undecidable'.
    """
    if not commit_msg:
        return "undecidable" if not cve_linked else "security_patch"

    msg_lower = commit_msg.lower()

    sec_hits = sum(1 for kw in SECURITY_KEYWORDS if kw in msg_lower)
    non_hits = sum(1 for kw in NON_SECURITY_KEYWORDS if kw in msg_lower)

    # CVE pattern in message is strong signal
    has_cve_in_msg = bool(re.search(r"cve-\d{4}-\d+", msg_lower))

    if has_cve_in_msg or (cve_linked and sec_hits > 0):
        return "security_patch"
    if cve_linked and sec_hits == 0 and non_hits > sec_hits:
        return "undecidable"
    if not cve_linked and non_hits > 0 and sec_hits == 0:
        return "non_security_patch"
    if sec_hits > non_hits:
        return "security_patch"
    if non_hits > sec_hits:
        return "non_security_patch"
    return "undecidable"


def load_monolens_cve_ids() -> set:
    """Load CVE IDs from MonoLens conf_0.9. Returns set of lowercase CVE IDs."""
    if not os.path.isfile(MONOLENS_CSV):
        log("WARNING: MonoLens CSV not found (placeholder). Falling back to heuristic only.")
        return set()

    try:
        df = pd.read_csv(MONOLENS_CSV, encoding="utf-8")
        if df.shape[1] < 2 or df.shape[0] == 0:
            log("WARNING: MonoLens CSV is empty placeholder. Falling back to heuristic only.")
            return set()
        # find cve column
        cve_col = next((c for c in df.columns if "cve" in c.lower()), None)
        if cve_col is None:
            log("WARNING: No CVE column found in MonoLens CSV.")
            return set()
        ids = set(df[cve_col].dropna().str.lower().str.strip())
        log(f"Loaded {len(ids)} MonoLens conf_0.9 CVE IDs.")
        return ids
    except Exception as e:
        log(f"WARNING: Could not load MonoLens CSV: {e}. Falling back to heuristic only.")
        return set()


def load_monolens_cwe_ids() -> set:
    """Load CWE IDs present in MonoLens overall_stats."""
    try:
        with open(MONO_STATS, encoding="utf-8") as f:
            stats = json.load(f)
        cwe_ids = set(stats.get("cwe_counts_overall", {}).keys())
        log(f"Loaded {len(cwe_ids)} CWE IDs from MonoLens stats.")
        return cwe_ids
    except Exception as e:
        log(f"WARNING: Could not load MonoLens stats: {e}")
        return set()


def main():
    log("=== Step 1.2 -- mono Cleaning ===")

    # ── 1. Load MonoLens verified CVE IDs ─────────────────────────────────────
    monolens_cve_ids = load_monolens_cve_ids()

    # ── 2. Connect to CVEFixes DB ─────────────────────────────────────────────
    log(f"Connecting to {DB_PATH} ...")
    con = sqlite3.connect(DB_PATH)

    # ── 3. Load all commits with CVE linkage via fixes table ──────────────────
    log("Querying commits with CVE linkage ...")
    query = """
        SELECT
            c.hash,
            c.repo_url,
            c.msg,
            f.cve_id,
            cc.cwe_id,
            fc.programming_language
        FROM commits c
        LEFT JOIN fixes f ON c.hash = f.hash
        LEFT JOIN cwe_classification cc ON f.cve_id = cc.cve_id
        LEFT JOIN file_change fc ON c.hash = fc.hash
    """
    df_raw = pd.read_sql_query(query, con)
    con.close()
    log(f"Raw joined rows: {len(df_raw)}")

    # ── 4. Deduplicate to one row per (hash, cve_id, cwe_id, language) ────────
    df_raw = df_raw.drop_duplicates(subset=["hash", "cve_id", "cwe_id", "programming_language"])
    log(f"After dedup: {len(df_raw)} rows")

    # ── 5. Classify each commit ───────────────────────────────────────────────
    log("Classifying commits ...")

    def classify_row(row):
        cve_id = str(row["cve_id"]).lower().strip() if pd.notna(row["cve_id"]) else ""
        cve_linked = cve_id != "" and cve_id != "nan"

        # MonoLens verified
        if cve_linked and cve_id in monolens_cve_ids:
            return "security_patch", "monolens_verified"

        # Heuristic
        decision = heuristic_classify(row["msg"], cve_linked)
        return decision, "heuristic"

    results = df_raw.apply(classify_row, axis=1, result_type="expand")
    df_raw["mono_decision"] = results[0]
    df_raw["classification_source"] = results[1]

    # ── 6. Language normalization ─────────────────────────────────────────────
    lang_map = {
        "c": "C", "c++": "C++", "cpp": "C++",
        "python": "Python", "java": "Java",
        "javascript": "JavaScript", "js": "JavaScript"
    }
    df_raw["language_normalized"] = (
        df_raw["programming_language"]
        .fillna("")
        .str.strip()
        .str.lower()
        .map(lambda x: lang_map.get(x, x.capitalize() if x else "Unknown"))
    )
    df_raw["in_target_languages"] = df_raw["language_normalized"].isin(TARGET_LANGS)

    # ── 7. Save report ────────────────────────────────────────────────────────
    log(f"Saving report to {REPORT_CSV} ...")
    df_out = df_raw[[
        "hash", "repo_url", "cve_id", "cwe_id",
        "programming_language", "language_normalized", "in_target_languages",
        "mono_decision", "classification_source"
    ]].copy()
    df_out.to_csv(REPORT_CSV, index=False, encoding="utf-8")

    # ── 8. Summary ────────────────────────────────────────────────────────────
    decision_counts = df_raw["mono_decision"].value_counts().to_dict()
    source_counts   = df_raw["classification_source"].value_counts().to_dict()
    lang_counts     = df_raw[df_raw["in_target_languages"]]["language_normalized"].value_counts().to_dict()

    security_df = df_raw[df_raw["mono_decision"] == "security_patch"]
    cwe_in_security = security_df["cwe_id"].value_counts().to_dict()

    summary = {
        "total_rows":           len(df_raw),
        "decision_counts":      decision_counts,
        "source_counts":        source_counts,
        "language_counts_target": lang_counts,
        "security_patch_cwe_counts": cwe_in_security,
        "monolens_verified_count": int((df_raw["classification_source"] == "monolens_verified").sum()),
        "heuristic_count":      int((df_raw["classification_source"] == "heuristic").sum()),
        "generated_at":         datetime.datetime.utcnow().isoformat()
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"Summary saved to {SUMMARY_JSON}")

    # ── 9. Checkpoint ─────────────────────────────────────────────────────────
    n_security = decision_counts.get("security_patch", 0)
    log(f"Decision counts: {decision_counts}")
    log(f"Security patches: {n_security}")

    if n_security == 0:
        log("CHECKPOINT FAIL: No security_patch commits found.")
        return False

    log("CHECKPOINT PASS: mono cleaning complete. Retain only security_patch in next step.")
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)