#!/usr/bin/env python3
"""Parse Vivado utilization reports produced by the active synthesis Tcl flow."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PACKAGE_ROOT / "results" / "component_ablation"


RESOURCE_PATTERNS = {
    "clb_luts": re.compile(r"\|\s*CLB LUTs\*\s*\|\s*([0-9,]+)"),
    "lut_as_logic": re.compile(r"\|\s*LUT as Logic\s*\|\s*([0-9,]+)"),
    "lut_as_memory": re.compile(r"\|\s*LUT as Memory\s*\|\s*([0-9,]+)"),
    "clb_registers": re.compile(r"\|\s*CLB Registers\s*\|\s*([0-9,]+)"),
    "dsp": re.compile(r"\|\s*DSPs\s*\|\s*([0-9,]+)"),
}


def parse_report(path: Path) -> dict[str, int | str]:
    text = path.read_text(errors="ignore")
    try:
        report_name = path.resolve().relative_to(PACKAGE_ROOT).as_posix()
    except ValueError:
        report_name = path.name
    row: dict[str, int | str] = {"report": report_name}
    for key, pat in RESOURCE_PATTERNS.items():
        m = pat.search(text)
        row[key] = int(m.group(1).replace(",", "")) if m else -1
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, default=RESULTS_DIR / "vivado_resource")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "vivado_resource_summary.csv")
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    rows = []
    for rpt in sorted(report_dir.glob("*_utilization_synth.rpt")):
        row = parse_report(rpt)
        row["top_module"] = rpt.name.replace("_utilization_synth.rpt", "")
        rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "top_module",
                "clb_luts",
                "lut_as_logic",
                "lut_as_memory",
                "clb_registers",
                "dsp",
                "report",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out} with {len(rows)} rows")


if __name__ == "__main__":
    main()
