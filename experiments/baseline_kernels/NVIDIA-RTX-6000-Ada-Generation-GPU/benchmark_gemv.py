#!/usr/bin/env python3
"""Benchmark conservative CUTLASS GEMV kernels for INT8 and INT4."""

import argparse
import gc
import math
import os

import torch

from benchmark_utils import benchmark_single, save_results
from build_ext import get_int4_ext, get_int8_ext
from config import GEMV_CONFIGS

MIN_POOL_SIZE = 8
MAX_POOL_SIZE = 512


def pack_to_int4_row_major(tensor_int8: torch.Tensor) -> torch.Tensor:
    """Pack pairs of signed INT4 values along the last dimension."""
    tensor_int8 = tensor_int8.contiguous()
    if tensor_int8.size(-1) % 2 != 0:
        raise ValueError("Last dimension must be even for INT4 packing")

    lo = tensor_int8[..., 0::2] & 0x0F
    hi = tensor_int8[..., 1::2] & 0x0F
    return ((hi << 4) | lo).to(torch.uint8).contiguous()


def tensor_bytes_mb(num_bytes: int) -> float:
    return num_bytes / (1024.0 * 1024.0)


def choose_pool_size(dtype: str, K: int, N: int, streaming_mb: int) -> int:
    weight_bytes = K * N if dtype == "int8" else (K * N) // 2
    target_bytes = streaming_mb * 1024 * 1024
    pool_size = math.ceil(target_bytes / weight_bytes)
    return max(MIN_POOL_SIZE, min(MAX_POOL_SIZE, pool_size))


def make_int8_verify_case(M: int, K: int, N: int, device: str):
    A = torch.randint(-128, 128, (M, K), dtype=torch.int8, device=device)
    B = torch.randint(-128, 128, (K, N), dtype=torch.int8, device=device)
    B_t = B.t().contiguous()

    return {
        "A": A,
        "B": B,
        "B_kernel": B_t,
    }


def make_int4_verify_case(M: int, K: int, N: int, device: str):
    A = torch.randint(-8, 8, (M, K), dtype=torch.int8, device=device)
    B = torch.randint(-8, 8, (K, N), dtype=torch.int8, device=device)
    A_packed = pack_to_int4_row_major(A)
    B_packed = pack_to_int4_row_major(B.t().contiguous())

    return {
        "A": A,
        "B": B,
        "A_kernel": A_packed,
        "B_kernel": B_packed,
    }


def verify_int8_case(ext, case: dict, M: int, K: int, N: int):
    out = ext.int8_gemv(case["A"], case["B_kernel"], M, N, K).cpu()
    ref = torch.matmul(case["A"].cpu().to(torch.int32), case["B"].cpu().to(torch.int32))
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def verify_int4_case(ext, case: dict, M: int, K: int, N: int):
    out = ext.int4_gemv(case["A_kernel"], case["B_kernel"], M, N, K).cpu()
    ref = torch.matmul(case["A"].cpu().to(torch.int32), case["B"].cpu().to(torch.int32))
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def make_int8_streaming_case(M: int, K: int, N: int, device: str, streaming_mb: int):
    pool_size = choose_pool_size("int8", K, N, streaming_mb)
    A = torch.randint(-128, 128, (M, K), dtype=torch.int8, device=device)
    B_bank = torch.randint(-128, 128, (pool_size, N, K), dtype=torch.int8, device=device)
    B_bank = tuple(B_bank.unbind(0))

    return {
        "A": A,
        "B_bank": B_bank,
        "pool_size": pool_size,
        "weight_bytes_mb": round(tensor_bytes_mb(K * N), 2),
        "working_set_mb": round(tensor_bytes_mb(pool_size * K * N), 2),
    }


def make_int4_streaming_case(M: int, K: int, N: int, device: str, streaming_mb: int):
    pool_size = choose_pool_size("int4", K, N, streaming_mb)
    A = torch.randint(-8, 8, (M, K), dtype=torch.int8, device=device)
    A_packed = pack_to_int4_row_major(A)
    B_raw_bank = torch.randint(-8, 8, (pool_size, N, K), dtype=torch.int8, device=device)
    B_packed_bank = pack_to_int4_row_major(B_raw_bank)
    B_packed_bank = tuple(B_packed_bank.unbind(0))

    return {
        "A": A,
        "A_kernel": A_packed,
        "B_bank": B_packed_bank,
        "pool_size": pool_size,
        "weight_bytes_mb": round(tensor_bytes_mb((K * N) // 2), 2),
        "working_set_mb": round(tensor_bytes_mb(pool_size * ((K * N) // 2)), 2),
    }


def make_streaming_fn(dtype: str, ext, case: dict, M: int, K: int, N: int):
    idx = 0
    pool_size = case["pool_size"]

    if dtype == "int8":
        A = case["A"]
        B_bank = case["B_bank"]

        def fn():
            nonlocal idx
            out = ext.int8_gemv(A, B_bank[idx], M, N, K)
            idx += 1
            if idx == pool_size:
                idx = 0
            return out

        return fn

    A_kernel = case["A_kernel"]
    B_bank = case["B_bank"]

    def fn():
        nonlocal idx
        out = ext.int4_gemv(A_kernel, B_bank[idx], M, N, K)
        idx += 1
        if idx == pool_size:
            idx = 0
        return out

    return fn


def run_dtype(dtype: str, ext, args) -> list[dict]:
    device = f"cuda:{args.gpu}"
    rows = []

    print(f"\n=== {dtype.upper()} GEMV benchmark on {torch.cuda.get_device_name(args.gpu)} (GPU {args.gpu}) ===")
    print(
        f"    streaming working set target: {args.streaming_mb} MB "
        f"(cycling distinct weight tensors to avoid L2-resident reuse)"
    )

    for name, M, K, N in GEMV_CONFIGS:
        print(f"  {name:16s} [{M},{K}]x[{K},{N}]  ", end="", flush=True)

        if dtype == "int8":
            if args.verify:
                verify_case = make_int8_verify_case(M, K, N, device)
                verify_int8_case(ext, verify_case, M, K, N)
                del verify_case
            case = make_int8_streaming_case(M, K, N, device, args.streaming_mb)
            fn = make_streaming_fn("int8", ext, case, M, K, N)
        else:
            if args.verify:
                verify_case = make_int4_verify_case(M, K, N, device)
                verify_int4_case(ext, verify_case, M, K, N)
                del verify_case
            case = make_int4_streaming_case(M, K, N, device, args.streaming_mb)
            fn = make_streaming_fn("int4", ext, case, M, K, N)

        result = benchmark_single(
            fn,
            M,
            K,
            N,
            warmup_iters=args.warmup,
            measure_sec=args.measure_sec,
            gpu_index=args.gpu,
        )
        result["config"] = name
        result["shape"] = f"[{M},{K}]x[{K},{N}]"
        result["dtype"] = dtype.upper()
        result["streaming_weight_mb"] = case["weight_bytes_mb"]
        result["working_set_mb"] = case["working_set_mb"]
        result["pool_size"] = case["pool_size"]
        rows.append(result)

        print(
            f"lat={result['latency_ms']:.4f}ms  "
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

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--measure-sec", type=float, default=10.0)
    parser.add_argument("--dtype", choices=["int8", "int4", "all"], default="all")
    parser.add_argument("--streaming-mb", type=int, default=1024)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(__file__), "results"),
    )
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)

    all_rows = []

    if args.dtype in {"int8", "all"}:
        int8_rows = run_dtype("int8", get_int8_ext(), args)
        save_results(int8_rows, args.out_dir, "int8_gemv_energy")
        all_rows.extend(int8_rows)

    if args.dtype in {"int4", "all"}:
        int4_rows = run_dtype("int4", get_int4_ext(), args)
        save_results(int4_rows, args.out_dir, "int4_gemv_energy")
        all_rows.extend(int4_rows)

    if len(all_rows) > 1:
        save_results(all_rows, args.out_dir, "gemv_energy_summary")

    print("\nFinished GEMV benchmark run.")


if __name__ == "__main__":
    main()
