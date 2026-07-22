#!/usr/bin/env python3
"""Extract the paper-compatible depth-3 64x64 OOC implementation point."""

from __future__ import annotations

import csv
import re
from pathlib import Path


EXPERIMENT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = EXPERIMENT.parents[1]
RESULT_DIR = PACKAGE_ROOT / "results" / "overlap_depth_sweep"
REPORT_DIR = RESULT_DIR / "depth3_64x64_ooc_210MHz"


def value(text: str, label: str) -> float:
    match = re.search(rf"(?m)^{re.escape(label)}\s*:\s*([0-9.]+)", text)
    if not match:
        raise AssertionError(f"missing {label}")
    return float(match.group(1))


def main() -> int:
    export = (REPORT_DIR / "gemv_kernel_export.rpt").read_text(encoding="utf-8", errors="replace")
    timing = (REPORT_DIR / "bd_0_wrapper_timing_summary_routed.rpt").read_text(
        encoding="utf-8", errors="replace"
    )
    match = re.search(
        r"WNS\(ns\).*?\n\s*-+.*?\n\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)",
        timing,
        re.DOTALL,
    )
    if not match:
        raise AssertionError("missing routed WNS/TNS")
    wns_ns = float(match.group(1))
    tns_ns = float(match.group(2))
    target_mhz = 210.0
    required_period_ns = value(export, "CP required")
    fmax_mhz = target_mhz * required_period_ns / (required_period_ns - wns_ns)

    with (RESULT_DIR / "layouts_w4a4_selected.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        layouts = {int(row["depth"]): row for row in csv.DictReader(handle)}
    layout = layouts[3]
    assert layout["prefill_schema"] == "3x3" and layout["decode_schema"] == "1x7"
    assert int(layout["prefill_T"]) == 9 and int(layout["decode_T"]) == 7

    row = {
        "depth": 3,
        "prefill_layout": "3x3",
        "decode_layout": "1x7",
        "prefill_T": 9,
        "decode_T": 7,
        "pe_count": 4096,
        "target_frequency_mhz": 210,
        "wns_ns": round(wns_ns, 3),
        "tns_ns": round(tns_ns, 3),
        "estimated_fmax_mhz": round(fmax_mhz, 3),
        "lut": int(value(export, "LUT")),
        "ff": int(value(export, "FF")),
        "dsp": int(value(export, "DSP")),
        "timing_met": True,
        "figure18_selected_point": True,
        "evidence_class": "RECOMPUTED_FROM_LOGS",
    }
    output = REPORT_DIR / "depth3_ooc_summary.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    assert row["wns_ns"] == 0.138 and row["dsp"] == 4096
    assert abs(row["estimated_fmax_mhz"] - 216.267) <= 0.001
    print("DEPTH3_OOC_PASS target_MHz=210 WNS_ns=0.138 Fmax_MHz=216.267 DSP=4096")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
