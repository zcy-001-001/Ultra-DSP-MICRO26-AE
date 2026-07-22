#!/usr/bin/env python3
"""Collect Vivado utilization report rows into a CSV summary."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PACKAGE_ROOT / "results" / "area_ablation"


METRICS = {
    "clb_luts": r"\|\s*CLB LUTs\*\s*\|\s*([0-9]+)",
    "lut_as_logic": r"\|\s*LUT as Logic\s*\|\s*([0-9]+)",
    "lut_as_memory": r"\|\s*LUT as Memory\s*\|\s*([0-9]+)",
    "clb_registers": r"\|\s*CLB Registers\s*\|\s*([0-9]+)",
    "dsp": r"\|\s*DSPs\s*\|\s*([0-9]+)",
}


def extract_metric(text: str, pattern: str) -> str:
    match = re.search(pattern, text)
    return match.group(1) if match else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, default=RESULTS_DIR / "vivado_resource")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "vivado_resource_summary.csv")
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    rows = []
    for report in sorted(report_dir.glob("*_utilization_synth.rpt")):
        text = report.read_text(encoding="utf-8", errors="ignore")
        top = report.name.replace("_utilization_synth.rpt", "")
        try:
            report_name = report.resolve().relative_to(PACKAGE_ROOT).as_posix()
        except ValueError:
            report_name = report.name
        row = {"top_module": top, "report": report_name}
        for name, pattern in METRICS.items():
            row[name] = extract_metric(text, pattern)
        rows.append(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["top_module", "report", *METRICS.keys()])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
