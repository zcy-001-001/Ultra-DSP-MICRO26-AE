#!/usr/bin/env python3
"""Audit correction LUT counts against solver layouts and Vivado reports."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT.parents[1]
RESULTS_DIR = PACKAGE_ROOT / "results" / "table8"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in str(value).split() if item]


def adjacent_overlaps(starts: list[int], product_result_width: int) -> list[int]:
    return [
        max(0, product_result_width - (right - left))
        for left, right in zip(starts[:-1], starts[1:])
    ]


def single_output_lut6_equiv_estimate(overlaps: list[int]) -> int:
    return sum(overlap for overlap in overlaps if overlap > 0)


def physical_clb_lut_estimate(input_width: int, overlaps: list[int]) -> int:
    nonzero = [overlap for overlap in overlaps if overlap > 0]
    if not nonzero:
        return 0

    max_low_bits = max(nonzero)
    if input_width == 1 and max_low_bits == 1:
        return math.ceil(len(nonzero) / 2)
    if max_low_bits <= 2:
        return len(nonzero)
    return (max_low_bits - 1) * len(nonzero)


def parse_vivado_report(path: Path) -> dict[str, str]:
    out = {
        "vivado_clb_luts": "",
        "vivado_dsp": "",
        "vivado_lut6_cells": "0",
        "vivado_lut5_cells": "0",
    }
    if not path.exists():
        return out

    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith("| CLB LUTs"):
            out["vivado_clb_luts"] = line.split("|")[2].strip()
        elif line.startswith("| DSPs"):
            out["vivado_dsp"] = line.split("|")[2].strip()
        elif line.startswith("| LUT6") and "CLB" in line:
            out["vivado_lut6_cells"] = line.split("|")[2].strip()
        elif line.startswith("| LUT5") and "CLB" in line:
            out["vivado_lut5_cells"] = line.split("|")[2].strip()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="manifest.csv")
    parser.add_argument("--layouts", type=Path, default=RESULTS_DIR / "fp_layouts.csv")
    parser.add_argument("--report-dir", type=Path, default=RESULTS_DIR / "vivado_resource")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "fp_correction_lut_audit.csv")
    args = parser.parse_args()

    manifest = Path(args.manifest)
    layouts = Path(args.layouts)
    report_dir = Path(args.report_dir)
    out = Path(args.out)
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    if not layouts.is_absolute():
        layouts = ROOT / layouts
    if not report_dir.is_absolute():
        report_dir = ROOT / report_dir
    if not out.is_absolute():
        out = ROOT / out

    layout_rows = {
        (row["format"], row["design"]): row
        for row in read_csv(layouts)
    }

    rows: list[dict[str, str | int]] = []
    for row in read_csv(manifest):
        if row["design"] != "UltraDSP-Packing":
            continue

        input_width = int(row["product_width"])
        product_result_width = 2 * input_width
        starts = parse_int_list(row["result_starts"])
        overlaps = adjacent_overlaps(starts, product_result_width)
        nonzero = [overlap for overlap in overlaps if overlap > 0]
        max_low_bits = max(nonzero) if nonzero else 0
        single_output_lut6_equiv = single_output_lut6_equiv_estimate(overlaps)
        physical_clb_lut_est = physical_clb_lut_estimate(input_width, overlaps)

        layout = layout_rows.get((row["format"], row["design"]), {})
        solver_starts = parse_int_list(layout.get("result_starts", ""))
        solver_match = int(starts == solver_starts)

        report = report_dir / f"{row['top_module']}_utilization_synth.rpt"
        vivado = parse_vivado_report(report)
        rows.append(
            {
                "format": row["format"],
                "format_label": row["format_label"],
                "input_width": input_width,
                "products_per_cycle": row["products_per_dsp"],
                "manifest_x_pos": row["x_pos"],
                "manifest_y_pos": row["y_pos"],
                "solver_result_starts_match": solver_match,
                "result_starts": " ".join(map(str, starts)),
                "adjacent_overlaps": " ".join(map(str, overlaps)),
                "correction_terms": len(nonzero),
                "max_low_bits": max_low_bits,
                "single_output_lut6_equiv_est": single_output_lut6_equiv,
                "physical_lut6_2_clb_est": physical_clb_lut_est,
                **vivado,
            }
        )

    fieldnames = [
        "format",
        "format_label",
        "input_width",
        "products_per_cycle",
        "manifest_x_pos",
        "manifest_y_pos",
        "solver_result_starts_match",
        "result_starts",
        "adjacent_overlaps",
        "correction_terms",
        "max_low_bits",
        "single_output_lut6_equiv_est",
        "physical_lut6_2_clb_est",
        "vivado_clb_luts",
        "vivado_dsp",
        "vivado_lut6_cells",
        "vivado_lut5_cells",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote correction LUT audit to {out}")


if __name__ == "__main__":
    main()
