#!/usr/bin/env python3
"""Audit Sales Prediction adapter for hidden-data references."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SYSTEM_DIR = ROOT / "src" / "systems" / "nined_energy_memory"

CHECK_FILES = [
    SYSTEM_DIR / "system.py",
    SYSTEM_DIR / "sales_runtime.py",
]

FORBIDDEN = [
    "ground_truth",
    "slice_annual_ground_truth",
    "FrozenSalesCorpus",
    "FrozenSalesCorpusMetadata",
    "load_sales_corpus",
    "resolve_corpus_paths",
    "_panel",
    "_furniture",
    "_locations",
    "sales_lifecycle_panel",
    "sales_lifecycle_metadata",
    "data/sales_prediction/frozen",
    "final_results",
]


def main() -> int:
    failures: list[str] = []
    for path in CHECK_FILES:
        if not path.exists():
            failures.append(f"missing file: {path}")
            continue
        text = path.read_text()
        for token in FORBIDDEN:
            if token in text:
                failures.append(f"{path.relative_to(ROOT)} contains forbidden token {token!r}")
    if failures:
        print("SALES PUBLIC-ONLY AUDIT FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("SALES PUBLIC-ONLY AUDIT PASSED")
    for path in CHECK_FILES:
        print(f"- scanned {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
