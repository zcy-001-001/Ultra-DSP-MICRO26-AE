#!/usr/bin/env python3
"""Memory-aware INT8 GEMV benchmark for the Intel Xeon Gold 6544Y."""

from __future__ import annotations

import argparse
import gc
import os

import numpy as np

from benchmark_utils import benchmark_single, save_results
from config import GEMV_CONFIGS
from mkl_int8 import (
    MKLInt8GEMVStream,
    choose_pool_size,
    print_system_info,
    set_mkl_threads,
)
from power_monitor import LinuxRAPLMonitor

MIN_POOL_SIZE = 8
MAX_POOL_SIZE = 1024


def verify_case(engine: MKLInt8GEMVStream):
    engine.run_bank_index(0)
    out = np.array(engine.C, copy=True)
    ref = engine.reference(0)
    np.testing.assert_array_equal(out, ref)


def run(args):
    os.makedirs(args.out_dir, exist_ok=True)

    set_mkl_threads(args.threads)
    rapl_monitor = LinuxRAPLMonitor()

    print("=== CPU GEMV benchmark (Intel Xeon Gold 6544Y, MKL INT8) ===")
    print_system_info()
    print(f"  Threads: {args.threads}")
    print(f"  Power mode: {'RAPL sysfs' if rapl_monitor.available() else 'none'}")
    print(f"  Streaming working set target: {args.streaming_mb} MB")
    print(f"  Pool bounds: min={MIN_POOL_SIZE}, max={MAX_POOL_SIZE}")

    rows = []
    for name, M, K, N in GEMV_CONFIGS:
        weight_bytes = K * N
        pool_size = choose_pool_size(weight_bytes, args.streaming_mb, MIN_POOL_SIZE, MAX_POOL_SIZE)

        print(f"\n  {name:16s} [{M},{K}]x[{K},{N}]  ", end="", flush=True)
        engine = MKLInt8GEMVStream(
            M=M,
            K=K,
            N=N,
            pool_size=pool_size,
            seed=args.seed,
            numa_first_touch=not args.disable_numa_first_touch,
        )

        if args.verify:
            verify_case(engine)

        result = benchmark_single(
            engine.run,
            M,
            K,
            N,
            warmup_sec=args.warmup_sec,
            measure_sec=args.measure_sec,
            rapl_monitor=rapl_monitor if rapl_monitor.available() else None,
        )

        result["config"] = name
        result["shape"] = f"[{M},{K}]x[{K},{N}]"
        result["dtype"] = "INT8"
        result["threads"] = args.threads
        result["streaming_weight_mb"] = engine.weight_bytes_mb
        result["working_set_mb"] = engine.working_set_mb
        result["pool_size"] = engine.pool_size
        result["streaming_target_mb"] = args.streaming_mb
        result["numa_first_touch"] = not args.disable_numa_first_touch
        rows.append(result)

        power_text = f"{result['power_total_w']:.1f}W" if result["power_total_w"] is not None else "N/A"
        energy_text = f"{result['energy_mj']:.4f}mJ" if result["energy_mj"] is not None else "N/A"
        topsw_text = f"{result['tops_per_w']:.6f}" if result["tops_per_w"] is not None else "N/A"
        print(
            f"lat={result['latency_ms']:.4f}ms  "
            f"power={power_text}  "
            f"energy={energy_text}  "
            f"TOPS/W={topsw_text}  "
            f"pool={engine.pool_size}  "
            f"working_set={engine.working_set_mb:.1f}MB"
        )

        del engine
        gc.collect()

    save_results(rows, args.out_dir, "int8_gemv_cpu_energy")
    print("\nFinished CPU GEMV benchmark run.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--warmup-sec", type=float, default=3.0)
    parser.add_argument("--measure-sec", type=float, default=10.0)
    parser.add_argument("--streaming-mb", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--disable-numa-first-touch", action="store_true")
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(__file__), "results"),
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
