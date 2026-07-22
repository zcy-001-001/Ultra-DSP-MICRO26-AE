#!/usr/bin/env python3
"""Verify full-board and OOC FPGA power evidence scopes."""

from __future__ import annotations

import csv
import hashlib
import re
import runpy
from pathlib import Path, PurePosixPath


EVIDENCE = Path(__file__).resolve().parent
ROOT = EVIDENCE.parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def total_power(text: str) -> float:
    match = re.search(r"Total On-Chip Power \(W\)\s*\|\s*([0-9.]+)", text)
    if not match:
        raise AssertionError("missing total on-chip power")
    return float(match.group(1))


def confidence(text: str) -> str:
    match = re.search(r"Confidence Level\s*\|\s*([A-Za-z]+)", text)
    if not match:
        raise AssertionError("missing confidence level")
    return match.group(1)


def main() -> int:
    with (EVIDENCE / "raw_public_manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    evidence: dict[str, tuple[float, str]] = {}
    for row in rows:
        assert not PurePosixPath(row["source_relative_path"]).is_absolute()
        public = EVIDENCE / Path(*PurePosixPath(row["public_relative_path"]).parts)
        assert public.is_file()
        assert public.stat().st_size == int(row["public_size_bytes"])
        assert sha256(public) == row["public_sha256"]
        text = public.read_text(encoding="utf-8", errors="replace")
        assert "Vivado v.2023.2" in text
        evidence[row["kind"]] = total_power(text), confidence(text)
    assert evidence == {
        "full_board_power": (44.78, "Low"),
        "ooc_power_scope_reference": (6.485, "Medium"),
    }
    sanitizer = runpy.run_path(str(ROOT / "scripts/sanitize_paths.py"))
    assert not sanitizer["audit"](EVIDENCE)
    print("FPGA_POWER_EVIDENCE_PASS full_board_W=44.780 ooc_W=6.485 privacy=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
