#!/usr/bin/env python3
"""Verify archived Figure 18 reports against the extracted sweep tables."""

from __future__ import annotations

import csv
import hashlib
import runpy
from pathlib import Path, PurePosixPath


EXPERIMENT = Path(__file__).resolve().parent.parent
ROOT = EXPERIMENT.parents[1]
RESULTS = ROOT / "results" / "table3_figure18"
EVIDENCE = RESULTS / "evidence" / "figure18"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    manifest = rows(EVIDENCE / "raw_public_manifest.csv")
    sweep = rows(RESULTS / "figure18_frequency_sweep.csv")
    selected = rows(RESULTS / "figure18_post_implementation_summary.csv")
    assert len(sweep) == 106
    assert len(selected) == 7
    assert len(manifest) == 113
    timing_manifest = [row for row in manifest if row["kind"] == "timing_sweep"]
    util_manifest = [row for row in manifest if row["kind"] == "selected_utilization"]
    assert len(timing_manifest) == 106
    assert len(util_manifest) == 7
    assert {row["source_relative_path"] for row in timing_manifest} == {
        row["timing_report"] for row in sweep
    }
    assert {row["source_relative_path"] for row in util_manifest} == {
        row["utilization_report"] for row in selected
    }

    for row in manifest:
        assert not PurePosixPath(row["source_relative_path"]).is_absolute()
        assert not PurePosixPath(row["public_relative_path"]).is_absolute()
        public = EVIDENCE / Path(*PurePosixPath(row["public_relative_path"]).parts)
        assert public.is_file(), public
        assert public.stat().st_size == int(row["public_size_bytes"])
        assert sha256(public) == row["public_sha256"]

    extractor = runpy.run_path(str(EXPERIMENT / "scripts/extract_ooc_reports.py"))
    parse_wns = extractor["parse_wns"]
    parse_utilization = extractor["parse_utilization"]
    for table_row in sweep:
        evidence_row = next(
            row for row in timing_manifest
            if row["source_relative_path"] == table_row["timing_report"]
        )
        public = EVIDENCE / Path(*PurePosixPath(evidence_row["public_relative_path"]).parts)
        assert parse_wns(public.read_text(encoding="utf-8", errors="replace")) == float(
            table_row["wns_ns"]
        )
    for table_row in selected:
        evidence_row = next(
            row for row in util_manifest
            if row["source_relative_path"] == table_row["utilization_report"]
        )
        public = EVIDENCE / Path(*PurePosixPath(evidence_row["public_relative_path"]).parts)
        utilization = parse_utilization(public.read_text(encoding="utf-8", errors="replace"))
        assert utilization == {
            "lut": int(table_row["lut"]),
            "ff": int(table_row["ff"]),
            "dsp": int(table_row["routed_report_dsp"]),
        }

    sanitizer = runpy.run_path(str(ROOT / "scripts/sanitize_paths.py"))
    assert not sanitizer["audit"](EVIDENCE)
    print("FIGURE18_EVIDENCE_PASS timing=106 selected_utilization=7 hashes=113 privacy=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
