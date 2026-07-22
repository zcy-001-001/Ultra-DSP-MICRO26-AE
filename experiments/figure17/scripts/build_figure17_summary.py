#!/usr/bin/env python3
"""Build the Figure 17 AE summary from Vivado utilization CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    # ``utf-8-sig`` accepts both plain UTF-8 and PowerShell-authored CSV files
    # with a BOM, keeping the portable rerun path identical on Linux/Windows.
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", type=Path, required=True)
    parser.add_argument("--area", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    component_by_top = {row["top_module"]: row for row in read_rows(args.component)}
    area_rows = read_rows(args.area)
    area_by_top = {row["top_module"]: row for row in area_rows}
    stage_specs = [
        ("Normal signed", "4/2", "w4a4_sf_v0_normal_signed_p2d2_single_dsp"),
        ("Sign-magnitude", "6/5", "w4a4_sf_v1_signmag_nonoverlap_p6d5_single_dsp"),
        ("Overpacking without correction", "8/6", "w4a4_sf_v2_overlap_no_correction_p8d6_single_dsp"),
        ("Lossless full correction", "8/6", "w4a4_sf_v3_full_correction_p8d6_single_dsp"),
    ]
    stages = []
    for label, packing, top in stage_specs:
        row = component_by_top[top]
        stages.append({
            "stage": label,
            "packing_prefill_decode": packing,
            "lut": int(row["clb_luts"]),
            "ff": int(row["clb_registers"]),
            "dsp": int(row["dsp"]),
        })

    final_row = area_by_top["W4A4_stage_s3_xor_sign"]
    final_point = {
        "stage": "ILP/resource optimized Ultra-DSP",
        "packing_prefill_decode": "9/7",
        "lut": int(final_row["clb_luts"]),
        "ff": int(final_row["clb_registers"]),
        "dsp": int(final_row["dsp"]),
    }

    formats: dict[str, dict[str, int]] = {}
    for row in area_rows:
        fmt, stage = row["top_module"].split("_stage_", 1)
        formats.setdefault(fmt, {})[stage] = int(row["clb_luts"])
    reductions = []
    for fmt, values in sorted(formats.items()):
        base, final = values["s0_original"], values["s3_xor_sign"]
        reductions.append({
            "format": fmt,
            "base_lut": base,
            "final_lut": final,
            "reduction_percent": round((base - final) / base * 100.0, 3),
        })

    minimum = min(row["reduction_percent"] for row in reductions)
    maximum = max(row["reduction_percent"] for row in reductions)
    assert len(component_by_top) == 4
    assert len(area_rows) == 24
    assert final_point == {
        "stage": "ILP/resource optimized Ultra-DSP",
        "packing_prefill_decode": "9/7",
        "lut": 75,
        "ff": 67,
        "dsp": 1,
    }
    assert round(minimum, 1) == 29.2
    assert round(maximum, 1) == 38.0

    output = {
        "tool": "Vivado 2023.2",
        "part": "xcu55c-fsvh2892-2L-e",
        "synthesis_directive": "AreaOptimized_high",
        "component_stages": stages,
        "final_ultradsp_point": final_point,
        "cross_precision_lut_reductions": reductions,
        "reduction_range_percent": [minimum, maximum],
        "claim_check": "PASS",
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("FIGURE17_SUMMARY_PASS")


if __name__ == "__main__":
    main()
