#!/usr/bin/env python3
"""Paper-matched INT4 GPU batch sweep for the Ultra-DSP rebuttal.

This extends the original Table-5 GPU GEMV method to modest batch sizes while
keeping the original streaming weight-bank policy. Batches up to 32 use the
original conservative GEMV extension so B=1 matches the paper baseline. Larger
batches use the existing Ada6000 optimized INT4 GEMM extension.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import math
import os
from pathlib import Path

import torch

from benchmark_utils import benchmark_single, save_results
from build_ext import get_int4_ext

MIN_POOL_SIZE = 8
MAX_POOL_SIZE = 512


def parse_batch_sizes(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def pack_to_int4_row_major(tensor_int8: torch.Tensor) -> torch.Tensor:
    tensor_int8 = tensor_int8.contiguous()
    if tensor_int8.size(-1) % 2 != 0:
        raise ValueError("Last dimension must be even for INT4 packing")

    lo = tensor_int8[..., 0::2] & 0x0F
    hi = tensor_int8[..., 1::2] & 0x0F
    return ((hi << 4) | lo).to(torch.uint8).contiguous()


def tensor_bytes_mb(num_bytes: int) -> float:
    return num_bytes / (1024.0 * 1024.0)


def choose_pool_size(k_dim: int, n_dim: int, streaming_mb: int) -> int:
    weight_bytes = (k_dim * n_dim) // 2
    target_bytes = streaming_mb * 1024 * 1024
    pool_size = math.ceil(target_bytes / weight_bytes)
    return max(MIN_POOL_SIZE, min(MAX_POOL_SIZE, pool_size))


def load_ada_int4_ext(opt_dir: Path):
    build_ext_path = opt_dir / "build_ext.py"
    if not build_ext_path.exists():
        raise FileNotFoundError(f"Cannot find Ada6000 INT4 build_ext.py at {build_ext_path}")

    spec = importlib.util.spec_from_file_location("ada6000_int4_build_ext", build_ext_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {build_ext_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_ext()


def make_streaming_case(batch: int, k_dim: int, n_dim: int, device: str, streaming_mb: int, seed: int):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + batch)

    pool_size = choose_pool_size(k_dim, n_dim, streaming_mb)
    activations = torch.randint(-8, 8, (batch, k_dim), dtype=torch.int8, device=device, generator=gen)
    activation_packed = pack_to_int4_row_major(activations)

    weight_bank_raw = torch.randint(
        -8,
        8,
        (pool_size, n_dim, k_dim),
        dtype=torch.int8,
        device=device,
        generator=gen,
    )
    weight_bank = pack_to_int4_row_major(weight_bank_raw)
    weight_bank = tuple(weight_bank.unbind(0))

    return {
        "A_kernel": activation_packed,
        "B_bank": weight_bank,
        "pool_size": pool_size,
        "weight_bytes_mb": round(tensor_bytes_mb((k_dim * n_dim) // 2), 2),
        "working_set_mb": round(tensor_bytes_mb(pool_size * ((k_dim * n_dim) // 2)), 2),
    }


def make_streaming_fn(batch: int, gemv_ext, gemm_ext, case: dict, k_dim: int, n_dim: int):
    idx = 0
    pool_size = case["pool_size"]
    a_kernel = case["A_kernel"]
    b_bank = case["B_bank"]

    if batch <= 32:
        def fn():
            nonlocal idx
            out = gemv_ext.int4_gemv(a_kernel, b_bank[idx], batch, n_dim, k_dim)
            idx += 1
            if idx == pool_size:
                idx = 0
            return out

        return fn, "original GEMV CUTLASS INT4 small-M"

    def fn():
        nonlocal idx
        out = gemm_ext.int4_gemm(a_kernel, b_bank[idx], batch, n_dim, k_dim)
        idx += 1
        if idx == pool_size:
            idx = 0
        return out

    return fn, "Ada6000/opt CUTLASS INT4 GEMM large-M"


def verify(ext, gemm_ext, device: str):
    gen = torch.Generator(device=device)
    gen.manual_seed(123)

    for batch in (4, 64):
        k_dim, n_dim = 256, 256
        a = torch.randint(-8, 8, (batch, k_dim), dtype=torch.int8, device=device, generator=gen)
        b = torch.randint(-8, 8, (n_dim, k_dim), dtype=torch.int8, device=device, generator=gen)
        a_packed = pack_to_int4_row_major(a)
        b_packed = pack_to_int4_row_major(b)
        if batch <= 32:
            out = ext.int4_gemv(a_packed, b_packed, batch, n_dim, k_dim).cpu()
        else:
            out = gemm_ext.int4_gemm(a_packed, b_packed, batch, n_dim, k_dim).cpu()
        ref = torch.matmul(a.cpu().to(torch.int32), b.cpu().t().contiguous().to(torch.int32))
        torch.testing.assert_close(out, ref, rtol=0, atol=0)


def run(args):
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)

    here = Path(__file__).resolve().parent
    default_opt_dir = here.parent / "Ada6000" / "opt"
    opt_dir = Path(args.ada_opt_dir).resolve() if args.ada_opt_dir else default_opt_dir

    gemv_ext = get_int4_ext()
    gemm_ext = load_ada_int4_ext(opt_dir)
    if args.verify:
        verify(gemv_ext, gemm_ext, device)

    print(f"=== Paper-matched INT4 GPU batch sweep on {torch.cuda.get_device_name(args.gpu)} ===")
    print(f"  Shape family: [B,{args.k_dim}]x[{args.k_dim},{args.n_dim}]")
    print(f"  Batch sizes: {args.batch_sizes}")
    print(f"  Streaming working set target: {args.streaming_mb} MB")
    print("  B<=32 uses original GEMV extension; B>32 uses Ada6000/opt GEMM extension")

    rows = []
    for batch in parse_batch_sizes(args.batch_sizes):
        print(f"\n  B={batch:<3d} [{batch},{args.k_dim}]x[{args.k_dim},{args.n_dim}]  ", end="", flush=True)
        case = make_streaming_case(batch, args.k_dim, args.n_dim, device, args.streaming_mb, args.seed)
        fn, kernel_family = make_streaming_fn(batch, gemv_ext, gemm_ext, case, args.k_dim, args.n_dim)
        result = benchmark_single(
            fn,
            batch,
            args.k_dim,
            args.n_dim,
            warmup_iters=args.warmup,
            measure_sec=args.measure_sec,
            gpu_index=args.gpu,
        )
        result["config"] = f"PAPER_MATCHED_STREAM_INT4_B{batch}"
        result["shape"] = f"[{batch},{args.k_dim}]x[{args.k_dim},{args.n_dim}]"
        result["dtype"] = "INT4"
        result["batch_size"] = batch
        result["weight_policy"] = "streaming_weight_bank_paper_matched"
        result["streaming_weight_mb"] = case["weight_bytes_mb"]
        result["working_set_mb"] = case["working_set_mb"]
        result["pool_size"] = case["pool_size"]
        result["kernel_family"] = kernel_family
        rows.append(result)

        print(
            f"lat={result['latency_ms']:.4f}ms  "
            f"TOPS={result['tops']:.4f}  "
            f"power={result['power_avg_w']:.1f}W  "
            f"energy={result['energy_mj']:.4f}mJ  "
            f"TOPS/W={result['tops_per_w']:.6f}  "
            f"pool={case['pool_size']}  "
            f"working_set={case['working_set_mb']:.1f}MB"
        )

        del case
        del fn
        gc.collect()
        torch.cuda.empty_cache()

    save_results(rows, args.out_dir, "int4_batch_streaming_gpu_energy")
    print("\nFinished paper-matched INT4 GPU batch sweep.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--k-dim", type=int, default=4096)
    parser.add_argument("--n-dim", type=int, default=4096)
    parser.add_argument("--batch-sizes", default="1,4,16,64,256")
    parser.add_argument("--streaming-mb", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--measure-sec", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--ada-opt-dir", default="")
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(__file__), "results_batch_paper_matched"),
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
