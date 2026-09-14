"""
Phase 1 - Step 1.1
Download CVEFixes dataset and record version, URL, and checksum
in dataset_manifest.json.
"""

import os
import json
import hashlib
import subprocess
import datetime
import argparse

# ── paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RAW_DIR        = os.path.join(PROJECT_ROOT, "data/raw/cvefixes")
CONFIGS_DIR    = os.path.join(PROJECT_ROOT, "configs")
MANIFEST_PATH  = os.path.join(CONFIGS_DIR, "dataset_manifest.json")
LOG_PATH       = os.path.join(PROJECT_ROOT, "logs/download_cvefixes.log")

# ── CVEFixes source ────────────────────────────────────────────────────────────
CVEFIXES_URL   = "https://github.com/secureIT-project/CVEfixes"
CVEFIXES_DB    = "CVEfixes.db"   # primary artifact inside the repo

os.makedirs(RAW_DIR,    exist_ok=True)
os.makedirs(CONFIGS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)


def log(msg: str):
    ts = datetime.datetime.utcnow().isoformat()
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clone_or_pull(url: str, dest: str) -> str:
    """Clone repo if not present, otherwise git pull. Returns HEAD commit hash."""
    if os.path.isdir(os.path.join(dest, ".git")):
        log(f"Repo already cloned at {dest}. Running git pull ...")
        subprocess.run(["git", "-C", dest, "pull"], check=True)
    else:
        log(f"Cloning {url} -> {dest} ...")
        subprocess.run(["git", "clone", "--depth", "1", url, dest], check=True)

    result = subprocess.run(
        ["git", "-C", dest, "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def load_manifest() -> dict:
    if os.path.isfile(MANIFEST_PATH):
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            m = json.load(f)
    else:
        m = {}
    m.setdefault("datasets", {})
    m.setdefault("deviations", [])
    m.setdefault("seeds", {})
    return m


def save_manifest(manifest: dict):
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    log(f"dataset_manifest.json updated at {MANIFEST_PATH}")


def main(skip_download: bool = False):
    log("=== Step 1.1 — Download CVEFixes ===")

    if not skip_download:
        commit_hash = clone_or_pull(CVEFIXES_URL, RAW_DIR)
    else:
        log("--skip-download flag set; assuming repo already present.")
        result = subprocess.run(
            ["git", "-C", RAW_DIR, "rev-parse", "HEAD"],
            capture_output=True, text=True
        )
        commit_hash = result.stdout.strip() if result.returncode == 0 else "unknown"

    # ── checksum of the primary DB artifact if present ────────────────────────
    db_path = os.path.join(RAW_DIR, CVEFIXES_DB)
    if os.path.isfile(db_path):
        checksum = sha256_file(db_path)
        db_size  = os.path.getsize(db_path)
        log(f"Found {CVEFIXES_DB}: size={db_size} bytes, sha256={checksum}")
    else:
        checksum = "artifact_not_present"
        db_size  = 0
        log(f"WARNING: {CVEFIXES_DB} not found in cloned repo. "
            f"Manual download may be required (see repo README).")

    # ── update manifest ───────────────────────────────────────────────────────
    manifest = load_manifest()
    manifest["datasets"]["cvefixes"] = {
        "url":          CVEFIXES_URL,
        "local_path":   RAW_DIR,
        "git_commit":   commit_hash,
        "primary_file": CVEFIXES_DB,
        "sha256":       checksum,
        "size_bytes":   db_size,
        "downloaded_at": datetime.datetime.utcnow().isoformat(),
        "status":       "present" if os.path.isfile(db_path) else "repo_only_db_missing"
    }
    save_manifest(manifest)

    # ── checkpoint ────────────────────────────────────────────────────────────
    if not os.path.isfile(db_path):
        log("CHECKPOINT FAIL: CVEFixes DB artifact not found. "
            "Download it manually per the repo README and rerun.")
        return False

    log("CHECKPOINT PASS: CVEFixes downloaded and recorded.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip git clone/pull; only re-record manifest.")
    args = parser.parse_args()
    success = main(skip_download=args.skip_download)
    exit(0 if success else 1)