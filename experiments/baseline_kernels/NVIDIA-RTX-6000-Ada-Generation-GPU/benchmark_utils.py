"""
Shared helpers for GEMV latency / power / energy-efficiency benchmarking.
"""

import csv
import json
import os
import time

import torch

from power_monitor import PowerMonitor


def compute_ops(M: int, K: int, N: int) -> int:
    """Return integer MAC op count for an MxK times KxN GEMM/GEMV."""
    return 2 * M * K * N


def tops(ops: int, latency_sec: float) -> float:
    """Convert op count and latency to TOPS."""
    if latency_sec <= 0:
        return 0.0
    return ops / latency_sec / 1e12


def benchmark_single(
    gemv_fn,
    M: int,
    K: int,
    N: int,
    warmup_iters: int = 100,
    measure_sec: float = 10.0,
    gpu_index: int = 0,
):
    """Benchmark a single GEMV kernel in a sustained loop."""
    op_count = compute_ops(M, K, N)

    for _ in range(warmup_iters):
        gemv_fn()
    torch.cuda.synchronize()

    pm = PowerMonitor(gpu_index=gpu_index)
    pm.start()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    iters = 0

    wall_start = time.perf_counter()
    start_evt.record()
    while time.perf_counter() - wall_start < measure_sec:
        gemv_fn()
        iters += 1
    end_evt.record()

    torch.cuda.synchronize()
    power = pm.stop()

    total_ms = start_evt.elapsed_time(end_evt)
    per_iter_ms = total_ms / iters if iters else 0.0
    throughput = tops(op_count, per_iter_ms / 1000.0)
    tops_per_w = throughput / power["avg_w"] if power["avg_w"] > 0 else 0.0
    energy_mj = power["avg_w"] * per_iter_ms

    return {
        "iters": iters,
        "total_ms": round(total_ms, 3),
        "latency_ms": round(per_iter_ms, 4),
        "tops": round(throughput, 4),
        "power_avg_w": power["avg_w"],
        "power_min_w": power["min_w"],
        "power_max_w": power["max_w"],
        "energy_mj": round(energy_mj, 4),
        "tops_per_w": round(tops_per_w, 6),
        "power_samples": power["samples"],
    }


def save_results(rows: list[dict], out_dir: str, basename: str):
    """Write rows to CSV and JSON."""
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{basename}.csv")
    json_path = os.path.join(out_dir, f"{basename}.json")

    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"  -> saved {csv_path}")
    print(f"  -> saved {json_path}")
