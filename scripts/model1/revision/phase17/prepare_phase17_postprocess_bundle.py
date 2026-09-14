#!/usr/bin/env python3
"""Build the hash-manifested Colab bundle for Phase 17 scanning and HumanEval."""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
PHASE = ROOT / "revision/model1/phase17"
OUT = PHASE / "outputs"
BUNDLE = PHASE / "phase17_colab_postprocess_bundle.zip"
MANIFEST = OUT / "phase17_postprocess_bundle_manifest.json"


FILES = {
    OUT / "phase17_scan_transfer_manifest.json": "phase17_scan_transfer_manifest.json",
    OUT / "alwayson_security_seed42_scanner_input.json": "alwayson_security_seed42_scanner_input.json",
    ROOT / "phases/phase9/colab_scan_phase9.py": "colab_scan_phase9.py",
    PHASE / "scripts/scan_phase17_manifest.py": "scan_phase17_manifest.py",
    ROOT / "outputs/phase10/utility_baseline_humaneval.json": "utility_baseline_humaneval.json",
    OUT / "alwayson_utility_seed42_outputs.json": "alwayson_utility_seed42_outputs.json",
    PHASE / "scripts/evaluate_phase17_humaneval.py": "evaluate_phase17_humaneval.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    missing = [str(path) for path in FILES if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing bundle input(s): {missing}")
    with zipfile.ZipFile(BUNDLE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source, name in FILES.items():
            archive.write(source, name)
    manifest = {
        "schema_version": "phase17_postprocess_bundle_manifest_v1", "status": "READY_FOR_COLAB",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_path": str(BUNDLE.relative_to(ROOT)).replace("\\", "/"),
        "bundle_sha256": sha256(BUNDLE), "bundle_size_bytes": BUNDLE.stat().st_size,
        "file_count": len(FILES),
        "files": [{"archive_name": name, "source_path": str(source.relative_to(ROOT)).replace("\\", "/"),
                   "sha256": sha256(source), "size_bytes": source.stat().st_size}
                  for source, name in FILES.items()],
        "required_return_files": [
            "alwayson_security_seed42_icd.json", "phase17_scan_return_manifest.json",
            "humaneval_b0_task_results.json", "humaneval_alwayson_task_results.json",
            "phase17_humaneval_return_manifest.json",
        ],
        "generation_performed": False, "scanner_frozen": True, "human_eval_version": "1.0.3",
    }
    temporary = MANIFEST.with_name(f".{MANIFEST.name}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(MANIFEST)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
