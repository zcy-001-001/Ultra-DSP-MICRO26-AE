#!/usr/bin/env python3
"""Table 5 i7 measurement using the author's Windows MKL/RAPL method."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_sustained(engine, warmup_sec: float, measure_sec: float):
    warmup_deadline = time.perf_counter() + warmup_sec
    warmup_iterations = 0
    while time.perf_counter() < warmup_deadline:
        engine.run()
        warmup_iterations += 1

    latencies_s = []
    wall_start = time.time()
    perf_start = time.perf_counter()
    deadline = perf_start + measure_sec
    while time.perf_counter() < deadline:
        start = time.perf_counter()
        engine.run()
        latencies_s.append(time.perf_counter() - start)
    total_time_s = time.perf_counter() - perf_start
    wall_end = time.time()
    return warmup_iterations, np.asarray(latencies_s), total_time_s, wall_start, wall_end


def stats(values: list[float]) -> dict | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {
        "num_samples": int(arr.size),
        "mean_W": round(float(np.mean(arr)), 6),
        "median_W": round(float(np.median(arr)), 6),
        "min_W": round(float(np.min(arr)), 6),
        "max_W": round(float(np.max(arr)), 6),
        "std_W": round(float(np.std(arr)), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--rapl-monitor", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--idle-sec", type=float, default=10.0)
    parser.add_argument("--warmup-sec", type=float, default=5.0)
    parser.add_argument("--measure-sec", type=float, default=30.0)
    parser.add_argument("--sample-interval", type=int, default=1)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    backend = load_module("table5_i7_backend_reference", args.backend)
    rapl = load_module("table5_i7_rapl_reference", args.rapl_monitor)
    domains = ["PKG", "PP0", "DRAM"]
    paper = {
        2048: {"latency_ms": 0.173, "power_W": 72.0, "energy_mJ": 12.438},
        4096: {"latency_ms": 0.574, "power_W": 75.0, "energy_mJ": 42.850},
    }

    run_started_utc = datetime.now(timezone.utc).isoformat()
    idle_power = rapl.measure_idle_power(
        duration=args.idle_sec,
        sample_interval=args.sample_interval,
        domains=domains,
    )

    summaries = []
    power_rows = []
    latency_rows = []

    for size in (2048, 4096):
        engine = backend.MKLInt8GEMM(1, size, size)
        monitor = rapl.MultiDomainRAPLMonitor(
            sample_interval=args.sample_interval,
            domains=domains,
        )
        if not monitor.start():
            raise RuntimeError("Windows RAPL monitor failed to start")

        warmup_iterations, latencies_s, total_time_s, wall_start, wall_end = run_sustained(
            engine, args.warmup_sec, args.measure_sec
        )
        monitor.stop()

        loaded_stats = monitor.get_all_stats(start_time=wall_start, end_time=wall_end)
        power_breakdown = rapl.compute_power_breakdown(loaded_stats, idle_power)

        # Table 5 reports full package power, matching the historical
        # Intel-i7-CPU script. Idle-subtracted values are retained separately.
        pkg_raw_W = loaded_stats.get("PKG", {}).get("mean_power_W")
        if pkg_raw_W is None or pkg_raw_W <= 0:
            raise RuntimeError("No valid PKG power samples inside the measurement window")

        latency_ms = latencies_s * 1000.0
        median_latency_ms = float(np.median(latency_ms))
        mean_latency_ms = float(np.mean(latency_ms))
        ops = 2 * size * size
        throughput_tops = ops * len(latencies_s) / (total_time_s * 1e12)
        energy_mJ = pkg_raw_W * median_latency_ms

        for iteration, value in enumerate(latency_ms, start=1):
            latency_rows.append(
                {
                    "shape": f"1x{size}x{size}",
                    "iteration": iteration,
                    "latency_ms": f"{value:.9f}",
                }
            )

        for domain, readings in monitor.readings.items():
            for timestamp, watts in readings:
                power_rows.append(
                    {
                        "shape": f"1x{size}x{size}",
                        "domain": domain,
                        "timestamp_epoch_s": f"{timestamp:.6f}",
                        "relative_to_measure_start_s": f"{timestamp - wall_start:.6f}",
                        "power_W": f"{watts:.6f}",
                        "inside_measurement_window": int(wall_start <= timestamp <= wall_end),
                    }
                )

        anchor = paper[size]
        summaries.append(
            {
                "shape": f"1x{size}x{size}",
                "warmup_seconds": args.warmup_sec,
                "warmup_iterations": warmup_iterations,
                "measurement_seconds": round(total_time_s, 6),
                "measurement_iterations": int(latencies_s.size),
                "measurement_wall_start_epoch_s": wall_start,
                "measurement_wall_end_epoch_s": wall_end,
                "median_latency_ms": round(median_latency_ms, 6),
                "mean_latency_ms": round(mean_latency_ms, 6),
                "p99_latency_ms": round(float(np.percentile(latency_ms, 99)), 6),
                "throughput_TOPS": round(throughput_tops, 9),
                "package_power_raw_W": round(float(pkg_raw_W), 6),
                "package_power_idle_W": round(float(idle_power.get("PKG", 0.0)), 6),
                "package_power_delta_W": power_breakdown["total_power_W"],
                "pp0_power_raw_W": power_breakdown["pp0_raw_W"],
                "pp0_power_delta_W": power_breakdown["compute_power_W"],
                "dram_power_raw_W": power_breakdown["dram_raw_W"],
                "dram_power_delta_W": power_breakdown["memory_power_W"],
                "energy_raw_package_mJ": round(energy_mJ, 6),
                "paper_latency_ms": anchor["latency_ms"],
                "paper_power_W": anchor["power_W"],
                "paper_energy_mJ": anchor["energy_mJ"],
                "latency_deviation_percent": round((median_latency_ms / anchor["latency_ms"] - 1) * 100, 3),
                "power_deviation_percent": round((pkg_raw_W / anchor["power_W"] - 1) * 100, 3),
                "energy_deviation_percent": round((energy_mJ / anchor["energy_mJ"] - 1) * 100, 3),
                "power_scope": "RAPL_Package0_PKG_raw_mean_during_measurement_window",
                "loaded_domain_stats": loaded_stats,
            }
        )

    with (args.out_dir / "i7_table5_power_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(power_rows[0]))
        writer.writeheader()
        writer.writerows(power_rows)

    with (args.out_dir / "i7_table5_latency_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(latency_rows[0]))
        writer.writeheader()
        writer.writerows(latency_rows)

    with (args.out_dir / "i7_table5_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        flat_fields = [key for key in summaries[0] if key != "loaded_domain_stats"]
        writer = csv.DictWriter(handle, fieldnames=flat_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)

    metadata = {
        "run_started_utc": run_started_utc,
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "cpu": platform.processor(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "logical_cpu_count": os.cpu_count(),
        "backend_sha256": sha256(args.backend),
        "rapl_monitor_sha256": sha256(args.rapl_monitor),
        "idle_power_W": idle_power,
        "method": {
            "backend": "MKL gemm_s8u8s32 resident weight, batch 1",
            "warmup_seconds": args.warmup_sec,
            "measurement_seconds": args.measure_sec,
            "rapl_sample_interval_seconds": args.sample_interval,
            "paper_power_column": "raw package mean, not idle-subtracted",
        },
        "results": summaries,
    }
    (args.out_dir / "i7_table5_summary.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
