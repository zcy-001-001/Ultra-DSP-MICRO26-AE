#!/usr/bin/env python3
"""Assemble Figure 13 CPU/GPU/FPGA rows without hiding their provenance."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "results/figure13/batch_comparison_canonical.csv"

FIELDS = [
    "evidence_class",
    "platform",
    "datatype",
    "batch_size",
    "latency_ms",
    "tops",
    "power_w",
    "energy_mj",
    "tops_per_w",
    "raw_total_power_w",
    "power_scope",
    "kernel_family",
    "source_detail",
]


def read_rows(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def cpu_rows() -> list[dict[str, str]]:
    source = read_rows(
        "results/figure13/cpu/"
        "int8_batch_streaming_cpu_paper_anchored.csv"
    )
    rows = []
    for row in source:
        anchor = row["source"].startswith("original_")
        rows.append(
            {
                "evidence_class": "PAPER_ANCHOR" if anchor else "MEASURED",
                "platform": "Server CPU",
                "datatype": row["dtype"],
                "batch_size": row["batch_size"],
                "latency_ms": row["latency_ms"],
                "tops": row["tops"],
                "power_w": row["paper_power_w"],
                "energy_mj": row["paper_energy_mj"],
                "tops_per_w": row["paper_tops_per_w"],
                "raw_total_power_w": row["total_power_w"],
                "power_scope": "single-package paper convention",
                "kernel_family": "MKL streaming GEMV/GEMM",
                "source_detail": "paper batch-1 anchor" if anchor else "batch streaming rerun",
            }
        )
    return rows


def gpu_rows() -> list[dict[str, str]]:
    source = read_rows(
        "results/figure13/gpu/"
        "int4_batch_streaming_gpu_paper_anchored.csv"
    )
    rows = []
    for row in source:
        anchor = row["source"].startswith("original_")
        rows.append(
            {
                "evidence_class": "PAPER_ANCHOR" if anchor else "MEASURED",
                "platform": "RTX 6000 Ada",
                "datatype": row["dtype"],
                "batch_size": row["batch_size"],
                "latency_ms": row["latency_ms"],
                "tops": row["tops"],
                "power_w": row["power_avg_w"],
                "energy_mj": row["energy_mj"],
                "tops_per_w": row["tops_per_w"],
                "raw_total_power_w": row["power_avg_w"],
                "power_scope": "GPU board power",
                "kernel_family": row["kernel_family"],
                "source_detail": "paper batch-1 anchor" if anchor else "batch streaming rerun",
            }
        )
    return rows


def fpga_rows(relative: str) -> list[dict[str, str]]:
    source = read_rows(relative)
    rows = []
    for row in source:
        rows.append(
            {
                "evidence_class": row["evidence_class"],
                "platform": row["platform"],
                "datatype": row["datatype"],
                "batch_size": row["batch_size"],
                "latency_ms": row["total_latency_ms"],
                "tops": row["throughput_tops"],
                "power_w": row["power_w"],
                "energy_mj": row["energy_mj"],
                "tops_per_w": row["tops_per_w"],
                "raw_total_power_w": "",
                "power_scope": "fixed analytical input",
                "kernel_family": "Ultra-DSP analytical array",
                "source_detail": "max(memory_time, compute_time)",
            }
        )
    return rows


def validate(rows: list[dict[str, str]]) -> None:
    assert len(rows) == 20, f"expected 20 rows, found {len(rows)}"
    classes = {row["evidence_class"] for row in rows}
    assert classes == {"MEASURED", "PAPER_ANCHOR", "ANALYTICAL_MODEL"}
    for row in rows:
        assert int(row["batch_size"]) in {1, 4, 16, 64, 256}
        if row["evidence_class"] == "ANALYTICAL_MODEL":
            assert float(row["power_w"]) == 45.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows = cpu_rows() + gpu_rows()
    rows += fpga_rows("results/figure13/fpga_gemv_batch_model.csv")
    rows += fpga_rows("results/figure13/fpga_gemv_batch_model_8192dsp.csv")
    validate(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"BATCH_COMPARISON_PASS rows={len(rows)}")
    print(f"WROTE={args.output.relative_to(ROOT) if args.output.is_relative_to(ROOT) else args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
