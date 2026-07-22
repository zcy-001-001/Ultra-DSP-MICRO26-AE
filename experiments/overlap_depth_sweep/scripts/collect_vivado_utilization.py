#!/usr/bin/env python3
"""Collect Vivado utilization and timing reports for overlap-depth sweep."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PACKAGE_ROOT / "results" / "overlap_depth_sweep"


RESOURCE_PATTERNS = {
    "clb_luts": re.compile(r"\|\s*CLB LUTs\*?\s*\|\s*([0-9,]+)"),
    "lut_as_logic": re.compile(r"\|\s*LUT as Logic\s*\|\s*([0-9,]+)"),
    "lut_as_memory": re.compile(r"\|\s*LUT as Memory\s*\|\s*([0-9,]+)"),
    "clb_registers": re.compile(r"\|\s*CLB Registers\s*\|\s*([0-9,]+)"),
    "dsp": re.compile(r"\|\s*DSPs\s*\|\s*([0-9,]+)"),
}


def parse_int(text: str, pattern: re.Pattern[str]) -> int | None:
    match = pattern.search(text)
    if not match:
        return None
    return int(match.group(1).replace(",", ""))


def parse_timing(path: Path) -> tuple[str, str]:
    if not path.exists():
        return "", ""
    lines = path.read_text(errors="ignore").splitlines()
    for idx, line in enumerate(lines):
        if "WNS(ns)" not in line:
            continue
        if idx + 2 >= len(lines):
            continue
        values = lines[idx + 2].split()
        if not values:
            continue
        wns = values[0]
        status = "NA" if wns.upper() == "NA" else ("MET" if float(wns) >= 0.0 else "VIOLATED")
        return wns, status
    return "", ""


def parse_report(path: Path) -> dict[str, object]:
    text = path.read_text(errors="ignore")
    row: dict[str, object] = {"top_module": path.name.replace("_utilization_synth.rpt", "")}
    for key, pattern in RESOURCE_PATTERNS.items():
        value = parse_int(text, pattern)
        row[key] = "" if value is None else value
    row["ff"] = row["clb_registers"]
    timing_path = path.with_name(path.name.replace("_utilization_synth.rpt", "_timing_synth.rpt"))
    wns, status = parse_timing(timing_path)
    row["wns_ns"] = wns
    row["timing_status"] = status
    try:
        row["utilization_report"] = path.resolve().relative_to(PACKAGE_ROOT).as_posix()
        row["timing_report"] = (
            timing_path.resolve().relative_to(PACKAGE_ROOT).as_posix()
            if timing_path.exists()
            else ""
        )
    except ValueError:
        row["utilization_report"] = path.name
        row["timing_report"] = timing_path.name if timing_path.exists() else ""
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, default=RESULTS_DIR / "vivado_resource")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "vivado_resource_summary.csv")
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    rows = [parse_report(path) for path in sorted(report_dir.glob("*_utilization_synth.rpt"))]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "top_module",
        "clb_luts",
        "lut_as_logic",
        "lut_as_memory",
        "ff",
        "clb_registers",
        "dsp",
        "wns_ns",
        "timing_status",
        "utilization_report",
        "timing_report",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out} with {len(rows)} rows")


if __name__ == "__main__":
    main()
