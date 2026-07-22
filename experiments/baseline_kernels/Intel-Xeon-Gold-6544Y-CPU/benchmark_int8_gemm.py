"""
INT8 GEMM Benchmark for CPU Energy Efficiency Baseline
=======================================================
Target: Intel Xeon Gold 6544Y (4th Gen Sapphire Rapids, AVX-512 VNNI + AMX-INT8)
Context: CPU baseline for Ultra-DSP FPGA DSP packing evaluation

Backend: Intel MKL cblas_gemm_s8u8s32 via ctypes (Linux)
  - Automatically selects AMX-INT8 on Sapphire Rapids when available
  - Falls back to AVX-512 VNNI on older platforms

GEMM shapes derived from LLaMA-2-7B (hidden=4096, intermediate=11008):
  - Attention Q/K/V/O projection: [M, 4096] x [4096, 4096]
  - FFN gate/up projection:       [M, 4096] x [4096, 11008]
  - FFN down projection:          [M, 11008] x [11008, 4096]

Phases:
  - Prefilling (GEMM): M in {512, 1024, 1536}
  - Decoding   (GEMV): M = 1  (batch_size=1)

Data format: INT8 weight x INT8 activation (symmetric, zero_point=0)
Metric: TOPS = 2*M*K*N / (latency * 1e12)
"""

import os
import sys
import time
import json
import argparse
import csv
import ctypes
import ctypes.util
import numpy as np

# ====================== MKL INT8 GEMM Engine ======================

def _find_mkl():
    """Locate MKL runtime library on Linux."""
    search_paths = []
    env_path = os.environ.get("MKL_RT_PATH")
    if env_path:
        search_paths.append(env_path)
    for prefix in (os.environ.get("CONDA_PREFIX"), sys.prefix):
        if prefix:
            search_paths.append(os.path.join(prefix, "lib", "libmkl_rt.so"))
    found = ctypes.util.find_library("mkl_rt")
    if found:
        search_paths.append(found)
    search_paths.append("libmkl_rt.so")

    for path in search_paths:
        if os.path.isfile(path):
            return path
    return None


def _load_mkl():
    """Load MKL and configure gemm_s8u8s32 (Fortran interface, column-major)."""
    mkl_path = _find_mkl()
    if mkl_path is None:
        raise RuntimeError(
            "MKL not found. Install via: conda install mkl mkl-service\n"
            "Or set LD_LIBRARY_PATH to include MKL lib directory."
        )
    print(f"  Loading MKL from: {mkl_path}")
    mkl = ctypes.cdll.LoadLibrary(mkl_path)

    # Fortran interface: gemm_s8u8s32_ (all scalars passed by reference)
    # subroutine gemm_s8u8s32(transa, transb, offsetc,
    #   m, n, k, alpha, a, lda, ao, b, ldb, bo, beta, c, ldc, co)
    gemm = mkl.gemm_s8u8s32_
    gemm.restype = None
    gemm.argtypes = [
        ctypes.c_char_p,                  # transa
        ctypes.c_char_p,                  # transb
        ctypes.c_char_p,                  # offsetc
        ctypes.POINTER(ctypes.c_int),     # m
        ctypes.POINTER(ctypes.c_int),     # n
        ctypes.POINTER(ctypes.c_int),     # k
        ctypes.POINTER(ctypes.c_float),   # alpha
        ctypes.c_void_p,                  # a (INT8)
        ctypes.POINTER(ctypes.c_int),     # lda
        ctypes.POINTER(ctypes.c_int8),    # ao
        ctypes.c_void_p,                  # b (UINT8)
        ctypes.POINTER(ctypes.c_int),     # ldb
        ctypes.POINTER(ctypes.c_int8),    # bo
        ctypes.POINTER(ctypes.c_float),   # beta
        ctypes.c_void_p,                  # c (INT32)
        ctypes.POINTER(ctypes.c_int),     # ldc
        ctypes.c_void_p,                  # co
    ]

    # Configure MKL threading
    try:
        set_threads = mkl.MKL_Set_Num_Threads
        set_threads.restype = None
        set_threads.argtypes = [ctypes.c_int]
        n_threads = os.cpu_count() or 1
        set_threads(n_threads)
        print(f"  MKL threads: {n_threads}")
    except Exception as e:
        print(f"  [WARN] Could not set MKL threads: {e}")

    return gemm, mkl


_MKL_GEMM, _MKL_LIB = _load_mkl()


class MKLInt8GEMM:
    """INT8 GEMM via MKL gemm_s8u8s32 (Fortran interface, column-major).
    Computes C_int32 = A_int8 @ B_uint8 using:
      - A stored as INT8 (column-major)
      - B converted to UINT8 with bo=-128 offset compensation
    """

    def __init__(self, M, K, N):
        self.M, self.K, self.N = M, K, N
        # Activation: INT8, column-major (Fortran order)
        self.A = np.asfortranarray(
            np.random.randint(-128, 127, (M, K), dtype=np.int8)
        )
        # Weight: INT8 -> UINT8 (add 128), column-major
        B_s8 = np.random.randint(-128, 127, (K, N), dtype=np.int8)
        self.B_u8 = np.asfortranarray(
            (B_s8.astype(np.int16) + 128).astype(np.uint8)
        )
        # Output: INT32, column-major
        self.C = np.zeros((M, N), dtype=np.int32, order="F")
        self.co = np.zeros(1, dtype=np.int32)

        # Pre-build ctypes args (avoid per-call overhead)
        self._m = ctypes.c_int(M)
        self._n = ctypes.c_int(N)
        self._k = ctypes.c_int(K)
        self._alpha = ctypes.c_float(1.0)
        self._beta = ctypes.c_float(0.0)
        self._lda = ctypes.c_int(M)   # column-major: lda = nrows(A)
        self._ldb = ctypes.c_int(K)   # column-major: ldb = nrows(B)
        self._ldc = ctypes.c_int(M)   # column-major: ldc = nrows(C)
        self._ao = ctypes.c_int8(0)    # symmetric quantization
        self._bo = ctypes.c_int8(-128)  # compensate INT8->UINT8 shift

        self._a_ptr = self.A.ctypes.data_as(ctypes.c_void_p)
        self._b_ptr = self.B_u8.ctypes.data_as(ctypes.c_void_p)
        self._c_ptr = self.C.ctypes.data_as(ctypes.c_void_p)
        self._co_ptr = self.co.ctypes.data_as(ctypes.c_void_p)

        self._args = (
            b"N", b"N", b"F",
            ctypes.byref(self._m), ctypes.byref(self._n), ctypes.byref(self._k),
            ctypes.byref(self._alpha),
            self._a_ptr, ctypes.byref(self._lda), ctypes.byref(self._ao),
            self._b_ptr, ctypes.byref(self._ldb), ctypes.byref(self._bo),
            ctypes.byref(self._beta),
            self._c_ptr, ctypes.byref(self._ldc),
            self._co_ptr,
        )

    def run(self):
        """Execute one INT8 GEMM."""
        _MKL_GEMM(*self._args)

    def update_activation(self):
        """Refresh activation data (optional, for realistic benchmarking)."""
        self.A[:] = np.random.randint(-128, 127, (self.M, self.K), dtype=np.int8)


# ====================== Benchmark ======================

def benchmark_gemm(M, K, N, warmup=50, iterations=200):
    """Benchmark a single INT8 GEMM shape. Returns latency array."""
    engine = MKLInt8GEMM(M, K, N)

    # Warmup
    for _ in range(warmup):
        engine.run()

    # Timed iterations
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        engine.run()
        t1 = time.perf_counter()
        latencies.append(t1 - t0)

    return np.array(latencies)


def benchmark_sustained(M, K, N, warmup_sec=5, duration_sec=30):
    """Run sustained GEMM for a fixed duration (for energy measurement).
    Returns (n_iters, total_time, latencies, wall_start, wall_end).
    """
    engine = MKLInt8GEMM(M, K, N)

    # Warmup
    warmup_end = time.perf_counter() + warmup_sec
    warmup_count = 0
    while time.perf_counter() < warmup_end:
        engine.run()
        warmup_count += 1

    # Sustained measurement
    latencies = []
    wall_start = time.time()
    perf_start = time.perf_counter()
    deadline = perf_start + duration_sec
    while time.perf_counter() < deadline:
        t0 = time.perf_counter()
        engine.run()
        latencies.append(time.perf_counter() - t0)
    total_time = time.perf_counter() - perf_start
    wall_end = time.time()

    return len(latencies), total_time, np.array(latencies), wall_start, wall_end


# ====================== Decode Sequence Benchmark ======================

def benchmark_decode_sequence(K, N, decode_steps, warmup=50):
    """Simulate autoregressive decoding: run M=1 GEMV for `decode_steps` steps.
    Returns dict with per-token and total latency statistics.
    """
    engine = MKLInt8GEMM(1, K, N)

    # Warmup
    for _ in range(warmup):
        engine.run()

    # Timed decode sequence
    step_latencies = []
    seq_start = time.perf_counter()
    for _ in range(decode_steps):
        t0 = time.perf_counter()
        engine.run()
        step_latencies.append(time.perf_counter() - t0)
    total_time = time.perf_counter() - seq_start

    step_latencies = np.array(step_latencies)
    ops_per_step = 2 * 1 * K * N
    total_ops = ops_per_step * decode_steps

    return {
        "decode_steps": decode_steps,
        "K": K, "N": N,
        "ops_per_step": ops_per_step,
        "total_ops": total_ops,
        "total_latency_ms": round(total_time * 1000, 4),
        "avg_per_token_latency_ms": round(float(np.mean(step_latencies)) * 1000, 4),
        "median_per_token_latency_ms": round(float(np.median(step_latencies)) * 1000, 4),
        "min_per_token_latency_ms": round(float(np.min(step_latencies)) * 1000, 4),
        "p99_per_token_latency_ms": round(float(np.percentile(step_latencies, 99)) * 1000, 4),
        "std_per_token_latency_ms": round(float(np.std(step_latencies)) * 1000, 4),
        "throughput_TOPS": round(total_ops / (total_time * 1e12), 6),
        "median_TOPS": round(ops_per_step / (float(np.median(step_latencies)) * 1e12), 6),
    }


# ====================== Metrics ======================

def compute_metrics(M, K, N, latencies):
    """Compute throughput metrics from latency array."""
    ops_per_gemm = 2 * M * K * N  # each MAC = 2 ops (multiply + add)
    lat_ms = latencies * 1000.0

    median_lat_ms = float(np.median(lat_ms))
    mean_lat_ms = float(np.mean(lat_ms))
    min_lat_ms = float(np.min(lat_ms))
    p99_lat_ms = float(np.percentile(lat_ms, 99))
    std_lat_ms = float(np.std(lat_ms))

    median_tops = ops_per_gemm / (float(np.median(latencies)) * 1e12)
    mean_tops = ops_per_gemm / (float(np.mean(latencies)) * 1e12)
    peak_tops = ops_per_gemm / (float(np.min(latencies)) * 1e12)

    return {
        "M": M, "K": K, "N": N,
        "ops_per_gemm": ops_per_gemm,
        "num_iterations": len(latencies),
        "median_latency_ms": round(median_lat_ms, 4),
        "mean_latency_ms": round(mean_lat_ms, 4),
        "min_latency_ms": round(min_lat_ms, 4),
        "p99_latency_ms": round(p99_lat_ms, 4),
        "std_latency_ms": round(std_lat_ms, 4),
        "median_TOPS": round(median_tops, 6),
        "mean_TOPS": round(mean_tops, 6),
        "peak_TOPS": round(peak_tops, 6),
    }


# ====================== Configuration ======================

# LLaMA-2-7B dimensions
HIDDEN = 4096
INTER = 11008

PREFILL_SEQLENS = [512, 1024, 1536]
DECODE_BATCH = 1

def get_gemm_configs():
    """Return list of (name, phase, M, K, N) tuples for all GEMM shapes."""
    configs = []
    # Prefilling phase (GEMM)
    for seq_len in PREFILL_SEQLENS:
        configs.append((f"Attn_Proj_seq{seq_len}", "Prefilling", seq_len, HIDDEN, HIDDEN))
        configs.append((f"FFN_GateUp_seq{seq_len}", "Prefilling", seq_len, HIDDEN, INTER))
        configs.append((f"FFN_Down_seq{seq_len}", "Prefilling", seq_len, INTER, HIDDEN))
    # Decoding phase (GEMV, batch=1)
    configs.append(("Attn_Proj_dec", "Decoding", DECODE_BATCH, HIDDEN, HIDDEN))
    configs.append(("FFN_GateUp_dec", "Decoding", DECODE_BATCH, HIDDEN, INTER))
    configs.append(("FFN_Down_dec", "Decoding", DECODE_BATCH, INTER, HIDDEN))
    return configs


# ====================== System Info ======================

def print_system_info():
    """Print CPU and system information."""
    info = {}
    cpuinfo_path = os.path.join(os.sep, "proc", "cpuinfo")
    try:
        with open(cpuinfo_path) as f:
            for line in f:
                if "model name" in line:
                    info["cpu_model"] = line.split(":")[1].strip()
                    break
    except Exception:
        info["cpu_model"] = "Unknown"

    try:
        import subprocess
        result = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=5)
        for line in result.stdout.splitlines():
            if "Socket(s):" in line:
                info["sockets"] = line.split(":")[1].strip()
            elif "Core(s) per socket:" in line:
                info["cores_per_socket"] = line.split(":")[1].strip()
            elif "Thread(s) per core:" in line:
                info["threads_per_core"] = line.split(":")[1].strip()
            elif "CPU(s):" in line and "On-line" not in line and "NUMA" not in line:
                info["total_cpus"] = line.split(":")[1].strip()
    except Exception:
        pass

    print(f"  CPU: {info.get('cpu_model', 'N/A')}")
    print(f"  Sockets: {info.get('sockets', 'N/A')}, "
          f"Cores/Socket: {info.get('cores_per_socket', 'N/A')}, "
          f"Threads/Core: {info.get('threads_per_core', 'N/A')}, "
          f"Total CPUs: {info.get('total_cpus', 'N/A')}")

    # Check ISA features
    try:
        with open(cpuinfo_path) as f:
            content = f.read()
            flags = []
            if "avx512_vnni" in content:
                flags.append("AVX-512 VNNI")
            if "amx_int8" in content:
                flags.append("AMX-INT8")
            if "avx_vnni" in content:
                flags.append("AVX-VNNI")
            print(f"  ISA: {', '.join(flags) if flags else 'N/A'}")
    except Exception:
        pass

    return info


# ====================== Main ======================

def main():
    parser = argparse.ArgumentParser(
        description="INT8 GEMM Benchmark for CPU Energy Efficiency (LLaMA-2-7B shapes)"
    )
    parser.add_argument("--warmup", type=int, default=50,
                        help="Warmup iterations (default: 50)")
    parser.add_argument("--iterations", type=int, default=200,
                        help="Benchmark iterations (default: 200)")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory (default: results)")
    parser.add_argument("--sustained", action="store_true",
                        help="Run sustained mode (fixed duration, for energy measurement)")
    parser.add_argument("--duration", type=int, default=30,
                        help="Sustained mode duration in seconds (default: 30)")
    parser.add_argument("--decode-steps", type=str, default="",
                        help="Comma-separated decode sequence lengths, e.g. '512,1024,1536'. "
                             "Simulates autoregressive decoding with that many GEMV steps.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Print system info
    print("=" * 78)
    print("  INT8 GEMM Benchmark -- CPU Energy Efficiency Baseline")
    print("  Model: LLaMA-2-7B | Data: INT8 x INT8 (symmetric, zero_point=0)")
    print("  Backend: Intel MKL cblas_gemm_s8u8s32 (AMX-INT8 / AVX-512 VNNI)")
    print_system_info()
    print(f"  Warmup: {args.warmup} iters | Benchmark: {args.iterations} iters")
    if args.sustained:
        print(f"  Mode: Sustained ({args.duration}s per config)")
    print("=" * 78)

    configs = get_gemm_configs()
    all_results = []

    for name, phase, M, K, N in configs:
        print(f"\n{'─'*66}")
        print(f"  {name}  |  [{M}, {K}] x [{K}, {N}]  |  Phase: {phase}")
        print(f"{'─'*66}")

        if args.sustained:
            n_iters, total_time, latencies, _, _ = benchmark_sustained(
                M, K, N, warmup_sec=5, duration_sec=args.duration
            )
            print(f"  Sustained: {n_iters} iters in {total_time:.2f}s")
        else:
            latencies = benchmark_gemm(M, K, N, args.warmup, args.iterations)

        metrics = compute_metrics(M, K, N, latencies)
        metrics["name"] = name
        metrics["phase"] = phase
        metrics["backend"] = "MKL_INT8_AMX"
        all_results.append(metrics)

        print(f"  Median Latency : {metrics['median_latency_ms']:.4f} ms")
        print(f"  Mean Latency   : {metrics['mean_latency_ms']:.4f} ms  "
              f"(std={metrics['std_latency_ms']:.4f})")
        print(f"  Median TOPS    : {metrics['median_TOPS']:.6f}")
        print(f"  Peak TOPS      : {metrics['peak_TOPS']:.6f}")

    # ---- Save results ----
    csv_path = os.path.join(args.output_dir, "gemm_benchmark.csv")
    json_path = os.path.join(args.output_dir, "gemm_benchmark.json")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nCSV saved: {csv_path}")

    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"JSON saved: {json_path}")

    # ---- Print summary table ----
    print("\n" + "=" * 100)
    print(f"{'Name':<25} {'Phase':<12} {'Shape':<28} "
          f"{'TOPS':<10} {'GOPS':<10} {'Lat(ms)':<10}")
    print("-" * 100)
    for r in all_results:
        shape = f"[{r['M']},{r['K']}]x[{r['K']},{r['N']}]"
        gops = r['median_TOPS'] * 1000
        print(f"{r['name']:<25} {r['phase']:<12} {shape:<28} "
              f"{r['median_TOPS']:<10.4f} {gops:<10.1f} {r['median_latency_ms']:<10.4f}")
    print("=" * 100)

    # ====================== Decode Sequence Benchmark ======================
    if args.decode_steps:
        decode_step_list = [int(x.strip()) for x in args.decode_steps.split(",")]
        decode_gemm_shapes = [
            ("Attn_Proj",  HIDDEN, HIDDEN),
            ("FFN_GateUp", HIDDEN, INTER),
            ("FFN_Down",   INTER,  HIDDEN),
        ]

        decode_seq_results = []
        print("\n" + "=" * 78)
        print("  Decode Sequence Benchmark (autoregressive multi-step GEMV)")
        print("=" * 78)

        for steps in decode_step_list:
            for layer_name, K, N in decode_gemm_shapes:
                cfg_name = f"{layer_name}_dec{steps}"
                print(f"\n{'─'*66}")
                print(f"  {cfg_name}  |  [1,{K}]x[{K},{N}]  |  {steps} decode steps")
                print(f"{'─'*66}")

                dr = benchmark_decode_sequence(K, N, steps, warmup=args.warmup)
                dr["name"] = cfg_name
                dr["layer"] = layer_name
                decode_seq_results.append(dr)

                print(f"  Total Generation Latency   : {dr['total_latency_ms']:.2f} ms")
                print(f"  Avg Per-Token Latency      : {dr['avg_per_token_latency_ms']:.4f} ms")
                print(f"  Median Per-Token Latency   : {dr['median_per_token_latency_ms']:.4f} ms")
                print(f"  Throughput TOPS            : {dr['throughput_TOPS']:.6f}")

        # Save decode-sequence results
        ds_json = os.path.join(args.output_dir, "decode_sequence_benchmark.json")
        ds_csv = os.path.join(args.output_dir, "decode_sequence_benchmark.csv")
        with open(ds_json, "w") as f:
            json.dump(decode_seq_results, f, indent=2)
        print(f"\nDecode-sequence JSON saved: {ds_json}")

        csv_fields = ["name", "layer", "decode_steps", "K", "N",
                      "total_latency_ms", "avg_per_token_latency_ms",
                      "median_per_token_latency_ms", "throughput_TOPS", "median_TOPS"]
        with open(ds_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(decode_seq_results)
        print(f"Decode-sequence CSV saved: {ds_csv}")

        # Print decode-sequence summary
        print("\n" + "=" * 110)
        print(f"{'Name':<25} {'Steps':<8} {'Shape':<26} "
              f"{'TotalLat(ms)':<14} {'PerToken(ms)':<14} {'TOPS':<10}")
        print("-" * 110)
        for dr in decode_seq_results:
            shape = f"[1,{dr['K']}]x[{dr['K']},{dr['N']}]"
            print(f"{dr['name']:<25} {dr['decode_steps']:<8} {shape:<26} "
                  f"{dr['total_latency_ms']:<14.2f} {dr['avg_per_token_latency_ms']:<14.4f} "
                  f"{dr['throughput_TOPS']:<10.6f}")
        print("=" * 110)


if __name__ == "__main__":
    main()
