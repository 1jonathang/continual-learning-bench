#!/usr/bin/env python3
"""Audit nined_energy_memory Cohort code for forbidden hidden-data access."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SYSTEM_DIR = ROOT / "src" / "systems" / "nined_energy_memory"

FORBIDDEN_PATTERNS = {
    "load_ground_truth": "task ground-truth loader",
    "FrozenDatasetMetadata": "frozen dataset metadata class",
    "GroundTruthEntry": "ground-truth entry class",
    "instance_references.json": "per-instance reference survival file",
    "ground_truth.json": "frozen ground-truth file",
    "ground_truth": "ground-truth symbol/path",
    "frozen_db": "frozen database internals",
    "data/cohort_studies/default": "direct cohort data path",
    "_study_info": "direct study-info table read",
}

ALLOWED_FILES = {
    "README.md",
}


def main() -> None:
    failures: list[dict[str, object]] = []
    scanned = []
    for path in sorted(SYSTEM_DIR.rglob("*")):
        if path.is_dir() or path.name in ALLOWED_FILES:
            continue
        if path.suffix not in {".py", ".md"}:
            continue
        text = path.read_text()
        scanned.append(str(path.relative_to(ROOT)))
        for pattern, reason in FORBIDDEN_PATTERNS.items():
            if pattern in text:
                failures.append(
                    {
                        "file": str(path.relative_to(ROOT)),
                        "pattern": pattern,
                        "reason": reason,
                    }
                )
    payload = {
        "system_dir": str(SYSTEM_DIR.relative_to(ROOT)),
        "scanned_files": scanned,
        "forbidden_patterns": FORBIDDEN_PATTERNS,
        "passed": not failures,
        "failures": failures,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
