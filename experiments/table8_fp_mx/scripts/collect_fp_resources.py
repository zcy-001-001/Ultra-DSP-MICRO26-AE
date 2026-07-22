#!/usr/bin/env python3
"""Collect Vivado PE resources and derive per-product area metrics."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT.parents[1]
RESULTS_DIR = PACKAGE_ROOT / "results" / "table8"


UDP_GRID_NEW = [
    [12, 10, 8, 6, 6, 4, 4],
    [10, 8, 6, 6, 4, 4, 4],
    [8, 6, 6, 4, 4, 4, 4],
    [6, 6, 4, 4, 4, 4, 2],
    [6, 4, 4, 4, 2, 2, 2],
    [4, 4, 4, 4, 2, 2, 2],
    [4, 4, 4, 2, 2, 2, 2],
]


RESOURCE_PATTERNS = {
    "clb_luts": re.compile(r"\|\s*CLB LUTs\*\s*\|\s*([0-9,]+)"),
    "lut_as_logic": re.compile(r"\|\s*LUT as Logic\s*\|\s*([0-9,]+)"),
    "lut_as_memory": re.compile(r"\|\s*LUT as Memory\s*\|\s*([0-9,]+)"),
    "clb_registers": re.compile(r"\|\s*CLB Registers\s*\|\s*([0-9,]+)"),
    "dsp": re.compile(r"\|\s*DSPs\s*\|\s*([0-9,]+)"),
}


def parse_int(text: str) -> int:
    return int(text.replace(",", ""))


def parse_report(path: Path) -> dict[str, int | str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    try:
        report_name = path.resolve().relative_to(PACKAGE_ROOT).as_posix()
    except ValueError:
        report_name = path.name
    row: dict[str, int | str] = {"report": report_name}
    for key, pattern in RESOURCE_PATTERNS.items():
        match = pattern.search(text)
        row[key] = parse_int(match.group(1)) if match else -1
    return row


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def pct_saving(baseline: float, observed: float) -> str:
    if baseline <= 0:
        return ""
    return f"{100.0 * (baseline - observed) / baseline:.2f}"


def udp_packing_same_width(product_width: int) -> str:
    udp_bitwidth = product_width
    if udp_bitwidth < 2 or udp_bitwidth > 8:
        return "NAN"
    idx = udp_bitwidth - 2
    return str(UDP_GRID_NEW[idx][idx])


def collect(manifest: Path, report_dir: Path, out: Path) -> None:
    rows = []
    scalar_by_format: dict[str, dict[str, float]] = {}
    lut_parallel_by_format: dict[str, dict[str, float]] = {}
    for item in read_csv(manifest):
        top = item["top_module"]
        report = report_dir / f"{top}_utilization_synth.rpt"
        if not report.exists():
            raise FileNotFoundError(f"Missing utilization report for {top}: {report}")
        parsed = parse_report(report)
        products = int(item["products_per_dsp"])
        lut = int(parsed["clb_luts"])
        ff = int(parsed["clb_registers"])
        row = {
            **item,
            **parsed,
            "products_per_cycle": f"{products}",
            "UDP_packing_same_width": udp_packing_same_width(int(item["product_width"])),
            "LUT_per_product": f"{lut / products:.4f}",
            "FF_per_product": f"{ff / products:.4f}",
        }
        if item["design"] == "NoPackingScalar":
            scalar_by_format[item["format"]] = {"lut": float(lut), "ff": float(ff)}
        if item["design"] == "NoDSP_LUTParallel":
            lut_parallel_by_format[item["format"]] = {"lut": float(lut), "ff": float(ff)}
        rows.append(row)

    for row in rows:
        base = scalar_by_format[row["format"]]
        products = int(row["products_per_dsp"])
        matched_lut = base["lut"] * products
        matched_ff = base["ff"] * products
        matched_dsp = products
        lut = float(row["clb_luts"])
        ff = float(row["clb_registers"])
        row["matched_one_product_dsp_LUT"] = f"{matched_lut:.2f}"
        row["matched_one_product_dsp_FF"] = f"{matched_ff:.2f}"
        row["matched_one_product_dsp_DSP"] = f"{matched_dsp:.0f}"
        row["LUT_saving_vs_matched_one_product_dsp_pct"] = pct_saving(matched_lut, lut)
        row["FF_saving_vs_matched_one_product_dsp_pct"] = pct_saving(matched_ff, ff)
        lut_parallel = lut_parallel_by_format.get(row["format"], {"lut": 0.0, "ff": 0.0})
        row["same_throughput_lut_baseline_LUT"] = f"{lut_parallel['lut']:.2f}"
        row["same_throughput_lut_baseline_FF"] = f"{lut_parallel['ff']:.2f}"
        row["LUT_saving_vs_same_throughput_lut_pct"] = pct_saving(lut_parallel["lut"], lut)
        row["FF_saving_vs_same_throughput_lut_pct"] = pct_saving(lut_parallel["ff"], ff)

    fieldnames = [
        "format",
        "format_label",
        "product_kind",
        "design",
        "top_module",
        "rtl_file",
        "product_width",
        "nx",
        "ny",
        "products_per_dsp",
        "products_per_cycle",
        "UDP_packing_same_width",
        "x_pos",
        "y_pos",
        "result_starts",
        "max_result_overlap",
        "clb_luts",
        "lut_as_logic",
        "lut_as_memory",
        "clb_registers",
        "dsp",
        "LUT_per_product",
        "FF_per_product",
        "matched_one_product_dsp_LUT",
        "matched_one_product_dsp_FF",
        "matched_one_product_dsp_DSP",
        "LUT_saving_vs_matched_one_product_dsp_pct",
        "FF_saving_vs_matched_one_product_dsp_pct",
        "same_throughput_lut_baseline_LUT",
        "same_throughput_lut_baseline_FF",
        "LUT_saving_vs_same_throughput_lut_pct",
        "FF_saving_vs_same_throughput_lut_pct",
        "report",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    try:
        display_out = out.resolve().relative_to(PACKAGE_ROOT).as_posix()
    except ValueError:
        display_out = out.name
    print(f"Wrote resource summary to {display_out}")

    lut_out = out.with_name("fp_pe_lut_summary.csv")
    lut_fieldnames = [
        "format",
        "format_label",
        "product_kind",
        "design",
        "top_module",
        "product_width",
        "products_per_cycle",
        "UDP_packing_same_width",
        "clb_luts",
        "dsp",
        "LUT_per_product",
        "same_throughput_lut_baseline_LUT",
        "LUT_saving_vs_same_throughput_lut_pct",
        "report",
    ]
    with lut_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=lut_fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in lut_fieldnames} for row in rows)
    try:
        display_lut_out = lut_out.resolve().relative_to(PACKAGE_ROOT).as_posix()
    except ValueError:
        display_lut_out = lut_out.name
    print(f"Wrote LUT-only summary to {display_lut_out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="manifest.csv")
    parser.add_argument("--report-dir", type=Path, default=RESULTS_DIR / "vivado_resource")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "fp_pe_resource_summary.csv")
    args = parser.parse_args()

    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    report_dir = Path(args.report_dir)
    if not report_dir.is_absolute():
        report_dir = ROOT / report_dir
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    collect(manifest, report_dir, out)


if __name__ == "__main__":
    main()
