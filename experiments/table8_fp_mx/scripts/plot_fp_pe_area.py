#!/usr/bin/env python3
"""Plot FP4/FP8 PE area efficiency panels."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT.parents[1]
RESULTS_DIR = PACKAGE_ROOT / "results" / "table8"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value != "" else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=RESULTS_DIR / "fp_pe_lut_summary.csv")
    parser.add_argument("--out-prefix", type=Path, default=RESULTS_DIR / "fp_pe_area_efficiency")
    args = parser.parse_args()

    summary = Path(args.summary)
    if not summary.is_absolute():
        summary = ROOT / summary
    out_prefix = Path(args.out_prefix)
    if not out_prefix.is_absolute():
        out_prefix = ROOT / out_prefix

    rows = read_rows(summary)
    products_key = "products_per_dsp" if rows and "products_per_dsp" in rows[0] else "products_per_cycle"
    designs = ("NoPackingScalar", "UltraDSP-Packing", "NoDSP_LUTParallel")
    main_rows = [row for row in rows if row["design"] in designs]
    formats = []
    for row in main_rows:
        if row["format_label"] not in formats:
            formats.append(row["format_label"])

    by_key = {(row["format_label"], row["design"]): row for row in main_rows}
    x = np.arange(len(formats))
    width = 0.25
    colors = {
        "NoPackingScalar": "#4C78A8",
        "UltraDSP-Packing": "#F58518",
        "NoDSP_LUTParallel": "#54A24B",
    }
    offsets = {
        "NoPackingScalar": -width,
        "UltraDSP-Packing": 0.0,
        "NoDSP_LUTParallel": width,
    }

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    ax = axs[0, 0]
    for design in designs:
        offset = offsets[design]
        vals = [as_float(by_key[(fmt, design)], "clb_luts") for fmt in formats]
        ax.bar(x + offset, vals, width, label=f"{design} LUT", color=colors[design])
    ax.set_title("A. Raw LUT per PE")
    ax.set_ylabel("CLB LUTs")
    ax.set_xticks(x)
    ax.set_xticklabels(formats, rotation=20, ha="right")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    ax = axs[0, 1]
    for design in designs:
        offset = offsets[design]
        vals = [as_float(by_key[(fmt, design)], products_key) for fmt in formats]
        ax.bar(x + offset, vals, width, label=design, color=colors[design])
    ax.set_title("B. Products per Cycle")
    ax.set_ylabel("Products / cycle")
    ax.set_xticks(x)
    ax.set_xticklabels(formats, rotation=20, ha="right")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    ax = axs[1, 0]
    for design in designs:
        offset = offsets[design]
        vals = [as_float(by_key[(fmt, design)], "LUT_per_product") for fmt in formats]
        ax.bar(x + offset, vals, width, label=design, color=colors[design])
    ax.set_title("C. LUT per Product")
    ax.set_ylabel("CLB LUTs / product")
    ax.set_xticks(x)
    ax.set_xticklabels(formats, rotation=20, ha="right")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    ax = axs[1, 1]
    savings = [
        as_float(by_key[(fmt, "UltraDSP-Packing")], "LUT_saving_vs_same_throughput_lut_pct")
        for fmt in formats
    ]
    ax.bar(x, savings, 0.52, color="#54A24B")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("D. LUT Saving vs Same-Throughput LUT")
    ax.set_ylabel("Saving (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(formats, rotation=20, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.suptitle("Decoded FP4/FP8/MXFP4 Mantissa Product Backend Area Efficiency", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".png"), dpi=220)
    fig.savefig(out_prefix.with_suffix(".pdf"))
    print(f"Wrote {out_prefix.with_suffix('.png')} and {out_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
