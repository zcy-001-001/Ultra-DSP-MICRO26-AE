#!/usr/bin/env python3
"""Paper-matched INT8 CPU batch sweep for the Ultra-DSP rebuttal.

This keeps the original GEMV-CPU streaming weight-bank policy and extends the
shape from [1,4096]x[4096,4096] to [B,4096]x[4096,4096]. It reports both the
RAPL total-power columns and the single-package paper columns used by the
CPU/GPU baseline artifact.
"""

from __future__ import annotations

import argparse
import gc
import os

import numpy as np

from benchmark_utils import benchmark_single, save_results
from mkl_int8 import (
    MKLInt8GEMVStream,
    choose_pool_size,
    print_system_info,
    set_mkl_threads,
)
from power_monitor import LinuxRAPLMonitor

MIN_POOL_SIZE = 8
MAX_POOL_SIZE = 1024


def parse_batch_sizes(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def verify_small():
    engine = MKLInt8GEMVStream(M=4, K=256, N=256, pool_size=8, seed=123)
    engine.run_bank_index(0)
    out = np.array(engine.C, copy=True)
    ref = engine.reference(0)
    np.testing.assert_array_equal(out, ref)


def add_paper_power_columns(result: dict):
    package0_w = result.get("package0_w")
    latency_ms = result.get("latency_ms")
    tops = result.get("tops")

    if package0_w is None or latency_ms is None:
        result["paper_power_w"] = None
        result["paper_energy_mj"] = None
        result["paper_tops_per_w"] = None
        result["paper_power_note"] = "N/A"
        return

    paper_energy_mj = package0_w * latency_ms
    paper_tops_per_w = tops / package0_w if package0_w else None
    result["paper_power_w"] = round(package0_w, 3)
    result["paper_energy_mj"] = round(paper_energy_mj, 4)
    result["paper_tops_per_w"] = round(paper_tops_per_w, 6) if paper_tops_per_w is not None else None
    result["paper_power_note"] = "single_package_package0_matches_paper_table"


def run(args):
    os.makedirs(args.out_dir, exist_ok=True)
    set_mkl_threads(args.threads)
    if args.verify:
        verify_small()

    rapl_monitor = LinuxRAPLMonitor()

    print("=== Paper-matched Xeon Gold 6544Y INT8 batch sweep (MKL gemm_s8u8s32) ===")
    print_system_info()
    print(f"  Threads: {args.threads}")
    print(f"  Power mode: {'RAPL sysfs' if rapl_monitor.available() else 'none'}")
    print(f"  Shape family: [B,{args.k_dim}]x[{args.k_dim},{args.n_dim}]")
    print(f"  Batch sizes: {args.batch_sizes}")
    print(f"  Streaming working set target: {args.streaming_mb} MB")
    print("  Weight policy: original streaming weight bank, not one resident reused weight")

    rows = []
    for batch in parse_batch_sizes(args.batch_sizes):
        weight_bytes = args.k_dim * args.n_dim
        pool_size = choose_pool_size(weight_bytes, args.streaming_mb, MIN_POOL_SIZE, MAX_POOL_SIZE)
        engine = MKLInt8GEMVStream(
            M=batch,
            K=args.k_dim,
            N=args.n_dim,
            pool_size=pool_size,
            seed=args.seed + batch,
            numa_first_touch=not args.disable_numa_first_touch,
        )

        print(f"\n  B={batch:<3d} [{batch},{args.k_dim}]x[{args.k_dim},{args.n_dim}]  ", end="", flush=True)
        result = benchmark_single(
            engine.run,
            batch,
            args.k_dim,
            args.n_dim,
            warmup_sec=args.warmup_sec,
            measure_sec=args.measure_sec,
            rapl_monitor=rapl_monitor if rapl_monitor.available() else None,
        )
        add_paper_power_columns(result)
        result["config"] = f"PAPER_MATCHED_STREAM_INT8_B{batch}"
        result["shape"] = f"[{batch},{args.k_dim}]x[{args.k_dim},{args.n_dim}]"
        result["dtype"] = "INT8"
        result["batch_size"] = batch
        result["threads"] = args.threads
        result["weight_policy"] = "streaming_weight_bank_paper_matched"
        result["streaming_weight_mb"] = engine.weight_bytes_mb
        result["working_set_mb"] = engine.working_set_mb
        result["pool_size"] = engine.pool_size
        result["streaming_target_mb"] = args.streaming_mb
        result["numa_first_touch"] = not args.disable_numa_first_touch
        rows.append(result)

        power_text = f"{result['power_total_w']:.1f}W" if result["power_total_w"] is not None else "N/A"
        paper_power_text = (
            f"{result['paper_power_w']:.1f}W" if result["paper_power_w"] is not None else "N/A"
        )
        print(
            f"lat={result['latency_ms']:.4f}ms  "
            f"TOPS={result['tops']:.4f}  "
            f"total_power={power_text}  "
            f"paper_power={paper_power_text}  "
            f"paper_energy={result['paper_energy_mj']}mJ  "
            f"paper_TOPS/W={result['paper_tops_per_w']}  "
            f"pool={engine.pool_size}  "
            f"working_set={engine.working_set_mb:.1f}MB"
        )

        del engine
        gc.collect()

    save_results(rows, args.out_dir, "int8_batch_streaming_cpu_energy")
    print("\nFinished paper-matched Xeon INT8 batch sweep.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--k-dim", type=int, default=4096)
    parser.add_argument("--n-dim", type=int, default=4096)
    parser.add_argument("--batch-sizes", default="1,4,16,64,256")
    parser.add_argument("--streaming-mb", type=int, default=1024)
    parser.add_argument("--warmup-sec", type=float, default=3.0)
    parser.add_argument("--measure-sec", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--disable-numa-first-touch", action="store_true")
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(__file__), "results_batch_paper_matched"),
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
