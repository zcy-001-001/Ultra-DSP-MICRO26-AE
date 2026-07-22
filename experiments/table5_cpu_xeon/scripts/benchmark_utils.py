"""
Shared helpers for CPU GEMV latency / power / energy-efficiency benchmarking.
"""

from __future__ import annotations

import csv
import json
import os
import time

from power_monitor import LinuxRAPLMonitor


def compute_ops(M: int, K: int, N: int) -> int:
    return 2 * M * K * N


def tops(ops: int, latency_sec: float) -> float:
    if latency_sec <= 0:
        return 0.0
    return ops / latency_sec / 1e12


def benchmark_single(
    gemv_fn,
    M: int,
    K: int,
    N: int,
    warmup_sec: float = 3.0,
    measure_sec: float = 10.0,
    rapl_monitor: LinuxRAPLMonitor | None = None,
):
    op_count = compute_ops(M, K, N)

    warmup_deadline = time.perf_counter() + warmup_sec
    warmup_iters = 0
    while time.perf_counter() < warmup_deadline:
        gemv_fn()
        warmup_iters += 1

    if rapl_monitor is not None and rapl_monitor.available():
        energy_start = rapl_monitor.read_energy_uj()
    else:
        energy_start = None

    measure_start = time.perf_counter()
    deadline = measure_start + measure_sec
    iters = 0
    while time.perf_counter() < deadline:
        gemv_fn()
        iters += 1
    measure_end = time.perf_counter()

    total_time = measure_end - measure_start
    latency_sec = total_time / iters if iters else 0.0
    throughput = tops(op_count, latency_sec)

    power_total = None
    package0_w = None
    package1_w = None
    power_source = None
    if energy_start is not None:
        energy_end = rapl_monitor.read_energy_uj()
        power_info = rapl_monitor.measure_power(energy_start, energy_end, total_time)
        power_total = power_info["total_W"]
        package0_w = power_info["per_socket"].get("package-0")
        package1_w = power_info["per_socket"].get("package-1")
        power_source = power_info["source"]

    energy_mj = power_total * latency_sec * 1000.0 if power_total is not None else None
    tops_per_w = throughput / power_total if power_total not in (None, 0) else None

    return {
        "warmup_iters": warmup_iters,
        "iters": iters,
        "total_time_s": round(total_time, 4),
        "latency_ms": round(latency_sec * 1000.0, 4),
        "tops": round(throughput, 4),
        "power_total_w": round(power_total, 3) if power_total is not None else None,
        "package0_w": package0_w,
        "package1_w": package1_w,
        "energy_mj": round(energy_mj, 4) if energy_mj is not None else None,
        "tops_per_w": round(tops_per_w, 6) if tops_per_w is not None else None,
        "power_source": power_source,
    }


def save_results(rows: list[dict], out_dir: str, basename: str):
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
