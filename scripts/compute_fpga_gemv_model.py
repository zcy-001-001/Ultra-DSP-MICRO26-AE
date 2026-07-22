#!/usr/bin/env python3
"""Recompute the FPGA analytical GEMV model used by Table 5 and Figure 13.

The default configuration follows the author-confirmed AE specification: a
W4A4 4096 x 4096 workload, 460 GB/s decimal memory bandwidth, 4096 DSPs, nine packed MACs per
DSP per cycle, 200 MHz, and fixed 45 W power.  The script only uses Python's
standard library and writes a CSV with the model inputs and intermediate terms.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable


DEFAULT_BATCHES = (1, 4, 16, 64, 256)
DEFAULT_K = 4096
DEFAULT_N = 4096
DEFAULT_WEIGHT_BITS = 4
# Author-confirmed Table 5 / Figure 13 convention.  The older 410 GB/s value
# is intentionally not used because it does not reproduce the paper's rounded
# 0.018 ms batch-1 weight-read latency.
DEFAULT_BANDWIDTH_GBPS = 460.0
DEFAULT_POWER_W = 45.0
DEFAULT_FREQUENCY_MHZ = 200.0
DEFAULT_DSP_COUNT = 4096
DEFAULT_PACKING_FACTOR = 9

FIELDNAMES = (
    "evidence_class",
    "paper_mapping",
    "platform",
    "datatype",
    "batch_size",
    "k",
    "n",
    "weight_bits_per_value",
    "total_weight_bits",
    "bandwidth_GBps_decimal",
    "power_w",
    "frequency_mhz",
    "dsp_count",
    "packed_macs_per_dsp_cycle",
    "macs",
    "ops",
    "memory_time_ms",
    "compute_cycles",
    "compute_time_ms",
    "total_latency_ms",
    "bottleneck",
    "throughput_tops",
    "energy_mj",
    "tops_per_w",
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be a positive finite number")
    return parsed


def parse_batches(value: str) -> tuple[int, ...]:
    try:
        batches = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("batches must be comma-separated integers") from exc
    if not batches or any(batch <= 0 for batch in batches):
        raise argparse.ArgumentTypeError("batches must contain positive integers")
    if len(set(batches)) != len(batches):
        raise argparse.ArgumentTypeError("batches must not contain duplicates")
    return batches


def default_output_path() -> Path:
    ae_root = Path(__file__).resolve().parent.parent
    return ae_root / "results" / "figure13" / "fpga_gemv_batch_model.csv"


def compute_rows(
    batches: Iterable[int],
    *,
    k: int,
    n: int,
    weight_bits: int,
    bandwidth_gbps: float,
    power_w: float,
    frequency_mhz: float,
    dsp_count: int,
    packing_factor: int,
) -> list[dict[str, object]]:
    """Return one auditable analytical-model row per batch size."""

    weight_total_bits = k * n * weight_bits
    bandwidth_bits_per_second = bandwidth_gbps * 1_000_000_000 * 8
    memory_time_s = weight_total_bits / bandwidth_bits_per_second
    cycle_time_s = 1.0 / (frequency_mhz * 1_000_000)

    rows: list[dict[str, object]] = []
    for batch in batches:
        macs = batch * k * n
        ops = 2 * macs
        compute_cycles = macs / (dsp_count * packing_factor)
        compute_time_s = compute_cycles * cycle_time_s
        total_time_s = max(memory_time_s, compute_time_s)
        bottleneck = "memory" if memory_time_s >= compute_time_s else "compute"
        throughput_tops = ops / total_time_s / 1_000_000_000_000
        latency_ms = total_time_s * 1_000

        rows.append(
            {
                "evidence_class": "ANALYTICAL_MODEL",
                "paper_mapping": "Table 5; Figure 13" if batch == 1 else "Figure 13",
                "platform": f"FPGA {dsp_count}-DSP analytical array",
                "datatype": "W4A4",
                "batch_size": batch,
                "k": k,
                "n": n,
                "weight_bits_per_value": weight_bits,
                "total_weight_bits": weight_total_bits,
                "bandwidth_GBps_decimal": bandwidth_gbps,
                "power_w": power_w,
                "frequency_mhz": frequency_mhz,
                "dsp_count": dsp_count,
                "packed_macs_per_dsp_cycle": packing_factor,
                "macs": macs,
                "ops": ops,
                "memory_time_ms": memory_time_s * 1_000,
                "compute_cycles": compute_cycles,
                "compute_time_ms": compute_time_s * 1_000,
                "total_latency_ms": latency_ms,
                "bottleneck": bottleneck,
                "throughput_tops": throughput_tops,
                "energy_mj": power_w * latency_ms,
                "tops_per_w": throughput_tops / power_w,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def run_self_test() -> None:
    rows = compute_rows(
        DEFAULT_BATCHES,
        k=DEFAULT_K,
        n=DEFAULT_N,
        weight_bits=DEFAULT_WEIGHT_BITS,
        bandwidth_gbps=DEFAULT_BANDWIDTH_GBPS,
        power_w=DEFAULT_POWER_W,
        frequency_mhz=DEFAULT_FREQUENCY_MHZ,
        dsp_count=DEFAULT_DSP_COUNT,
        packing_factor=DEFAULT_PACKING_FACTOR,
    )
    expected_memory_ms = (4096 * 4096 * 4) / (460e9 * 8) * 1e3
    expected_compute_ms_b1 = (4096 * 4096 / (4096 * 9)) * 5e-9 * 1e3
    assert len(rows) == len(DEFAULT_BATCHES)
    assert math.isclose(float(rows[0]["memory_time_ms"]), expected_memory_ms, rel_tol=1e-14)
    assert math.isclose(float(rows[0]["compute_time_ms"]), expected_compute_ms_b1, rel_tol=1e-14)
    assert rows[0]["bottleneck"] == "memory"
    assert rows[2]["bottleneck"] == "compute"
    assert all(float(row["power_w"]) == 45.0 for row in rows)
    assert all(
        math.isclose(
            float(row["total_latency_ms"]),
            max(float(row["memory_time_ms"]), float(row["compute_time_ms"])),
            rel_tol=1e-14,
        )
        for row in rows
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=parse_batches, default=DEFAULT_BATCHES)
    parser.add_argument("--k", type=positive_int, default=DEFAULT_K)
    parser.add_argument("--n", type=positive_int, default=DEFAULT_N)
    parser.add_argument("--weight-bits", type=positive_int, default=DEFAULT_WEIGHT_BITS)
    parser.add_argument("--bandwidth-gbps", type=positive_float, default=DEFAULT_BANDWIDTH_GBPS)
    parser.add_argument("--power-w", type=positive_float, default=DEFAULT_POWER_W)
    parser.add_argument("--frequency-mhz", type=positive_float, default=DEFAULT_FREQUENCY_MHZ)
    parser.add_argument("--dsp-count", type=positive_int, default=DEFAULT_DSP_COUNT)
    parser.add_argument("--packing-factor", type=positive_int, default=DEFAULT_PACKING_FACTOR)
    parser.add_argument("--output", type=Path, default=None, help="CSV path (default: results/figure13)")
    parser.add_argument("--self-test", action="store_true", help="run deterministic model assertions before writing")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        run_self_test()
        print("SELF_TEST=PASS")

    rows = compute_rows(
        args.batches,
        k=args.k,
        n=args.n,
        weight_bits=args.weight_bits,
        bandwidth_gbps=args.bandwidth_gbps,
        power_w=args.power_w,
        frequency_mhz=args.frequency_mhz,
        dsp_count=args.dsp_count,
        packing_factor=args.packing_factor,
    )
    output = args.output if args.output is not None else default_output_path()
    write_csv(output, rows)
    resolved_output = output.resolve()
    ae_root = Path(__file__).resolve().parent.parent
    try:
        display_output = resolved_output.relative_to(ae_root).as_posix()
    except ValueError:
        # A caller may deliberately select an external output. Print only its
        # filename so captured AE logs do not disclose a workstation path.
        display_output = resolved_output.name
    print(f"WROTE={display_output}")
    print(f"ROWS={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
