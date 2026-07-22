#!/usr/bin/env python3
"""Verify sanitized phase-adaptivity reports and their provenance manifest."""

from __future__ import annotations

import csv
import hashlib
import runpy
from pathlib import Path, PurePosixPath


EXPERIMENT = Path(__file__).resolve().parent.parent
ROOT = EXPERIMENT.parents[1]
RESULTS = ROOT / "results" / "phase_adaptivity"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    with (RESULTS / "report_manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12
    assert sum(row["kind"].startswith("p_") for row in rows) == 4
    assert sum(row["kind"].startswith("d_") for row in rows) == 4
    assert sum(row["kind"].startswith("hybrid_") for row in rows) == 4
    for row in rows:
        assert not PurePosixPath(row["source_relative_path"]).is_absolute()
        public = RESULTS / Path(*PurePosixPath(row["public_relative_path"]).parts)
        assert public.is_file(), public
        assert public.stat().st_size == int(row["public_size_bytes"])
        assert sha256(public) == row["public_sha256"]
    power_row = next(row for row in rows if row["kind"] == "hybrid_full_board_power")
    assert power_row["public_relative_path"] == (
        "reports/hybrid/hw_bb_locked_power_routed.rpt"
    )
    sanitizer = runpy.run_path(str(ROOT / "scripts/sanitize_paths.py"))
    assert not sanitizer["audit"](RESULTS)
    print("PHASE_REPORT_MANIFEST_PASS reports=12 public_hashes=12 privacy=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
