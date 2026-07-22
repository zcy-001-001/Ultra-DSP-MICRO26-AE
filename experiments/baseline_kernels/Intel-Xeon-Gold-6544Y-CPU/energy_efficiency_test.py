"""
Energy Efficiency Test (TOPS/W) for CPU INT8 GEMM — Linux
==========================================================
Target: Intel Xeon Gold 6544Y (Sapphire Rapids, AMX-INT8)
Context: CPU baseline for Ultra-DSP FPGA DSP packing evaluation

Power measurement strategies (tried in order):
  1. Linux RAPL sysfs package counters — need read permission
  2. perf stat -e power/energy-pkg/ wrapper — needs perf_event_paranoid <= 0
  3. Manual input — user enters average power from external tool

Flow per config:
  1. Warmup GEMM (5s)
  2. Read RAPL energy_start (or start perf)
  3. Run sustained GEMM for measurement_duration
  4. Read RAPL energy_end (or stop perf)
  5. Compute: Power = delta_energy / duration
  6. Compute: TOPS/W = TOPS / Power
"""

import os
import sys
import time
import json
import csv
import argparse
import subprocess
import threading
import re
import glob
import numpy as np

# ====================== Power Monitor Backends ======================

class LinuxRAPLMonitor:
    """Read CPU package power via Linux RAPL sysfs interface.
    Reads Linux RAPL package energy counters.
    Supports multi-socket systems (sums all packages).
    """

    def __init__(self):
        self.rapl_paths = []
        self.max_energy = {}
        self._discover_rapl()

    def _discover_rapl(self):
        """Find all RAPL package energy counters."""
        rapl_root = os.path.join(os.sep, "sys", "class", "powercap", "intel-rapl")
        pattern = os.path.join(rapl_root, "intel-rapl:*", "energy_uj")
        paths = sorted(glob.glob(pattern))
        for p in paths:
            try:
                with open(p) as f:
                    f.read()
                self.rapl_paths.append(p)
                # Read max_energy_range_uj for wraparound handling
                max_path = p.replace("energy_uj", "max_energy_range_uj")
                try:
                    with open(max_path) as f:
                        self.max_energy[p] = int(f.read().strip())
                except Exception:
                    self.max_energy[p] = 2**63
            except PermissionError:
                continue
        if self.rapl_paths:
            names = []
            for p in self.rapl_paths:
                name_path = p.replace("energy_uj", "name")
                try:
                    with open(name_path) as f:
                        names.append(f.read().strip())
                except Exception:
                    names.append("unknown")
            print(f"  RAPL sysfs available: {len(self.rapl_paths)} package(s): {names}")

    def available(self):
        return len(self.rapl_paths) > 0

    def read_energy_uj(self):
        """Read per-package energy in microjoules."""
        readings = {}
        for p in self.rapl_paths:
            with open(p) as f:
                readings[p] = int(f.read().strip())
        return readings

    def measure_power(self, start_energy_uj, end_energy_uj, duration_s):
        """Compute power from per-package energy deltas, handling wraparound.
        Returns dict: {"total_W": float, "per_socket": {name: float, ...}}
        """
        total_delta_uj = 0
        per_socket = {}
        for p in self.rapl_paths:
            start_val = start_energy_uj[p]
            end_val = end_energy_uj[p]
            delta = end_val - start_val
            if delta < 0:
                delta += self.max_energy[p]
            total_delta_uj += delta
            # Get socket name
            name_path = p.replace("energy_uj", "name")
            try:
                with open(name_path) as f:
                    sock_name = f.read().strip()
            except Exception:
                sock_name = p
            per_socket[sock_name] = round((delta / 1e6) / duration_s, 3)
        total_w = (total_delta_uj / 1e6) / duration_s
        return {"total_W": round(total_w, 3), "per_socket": per_socket}


class PerfStatMonitor:
    """Use perf stat to measure CPU package energy.
    Wraps the benchmark in a perf stat call.
    """

    @staticmethod
    def available():
        """Check if perf stat power events are accessible."""
        try:
            result = subprocess.run(
                ["perf", "stat", "-e", "power/energy-pkg/", "--", "sleep", "0.1"],
                capture_output=True, text=True, timeout=10
            )
            return "Joules" in result.stderr or "energy-pkg" in result.stderr
        except Exception:
            return False

    @staticmethod
    def parse_energy(stderr_text):
        """Parse energy in Joules from perf stat output."""
        for line in stderr_text.splitlines():
            if "energy-pkg" in line or "Joules" in line:
                # Format: "    123.45 Joules power/energy-pkg/"
                match = re.search(r'([\d.]+)\s+Joules', line)
                if match:
                    return float(match.group(1))
        return None


class ManualPowerInput:
    """Fallback: user provides average power reading from external tool."""

    def get_power(self, config_name):
        print(f"\n  [Manual Mode] Config: {config_name}")
        print("  Read power from an external tool (e.g., PDU, BMC IPMI, wall meter).")
        try:
            power = float(input("  Enter average CPU package power in Watts: "))
            return power
        except (ValueError, EOFError):
            return None


# ====================== GEMM Engine ======================

# Import from the benchmark module.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_int8_gemm import MKLInt8GEMM, print_system_info


def run_sustained_gemm(M, K, N, warmup_sec=5, measure_sec=30):
    """Run MKL INT8 GEMM in a tight loop for a fixed duration. Returns metrics dict."""
    engine = MKLInt8GEMM(M, K, N)

    # Warmup
    print(f"    Warming up ({warmup_sec}s)...")
    warmup_end = time.perf_counter() + warmup_sec
    warmup_iters = 0
    while time.perf_counter() < warmup_end:
        engine.run()
        warmup_iters += 1
    print(f"    Warmup done ({warmup_iters} iters)")

    # Sustained measurement
    print(f"    Measuring ({measure_sec}s)...")
    latencies = []
    measure_start = time.perf_counter()
    measure_start_wall = time.time()
    deadline = measure_start + measure_sec
    while time.perf_counter() < deadline:
        t0 = time.perf_counter()
        engine.run()
        latencies.append(time.perf_counter() - t0)
    measure_end = time.perf_counter()
    measure_end_wall = time.time()
    total_time = measure_end - measure_start

    latencies = np.array(latencies)
    n_iters = len(latencies)
    ops_per_gemm = 2 * M * K * N

    total_ops = ops_per_gemm * n_iters
    throughput_tops = total_ops / (total_time * 1e12)
    median_tops = ops_per_gemm / (float(np.median(latencies)) * 1e12)

    return {
        "n_iters": n_iters,
        "total_time_s": round(total_time, 3),
        "wall_start": measure_start_wall,
        "wall_end": measure_end_wall,
        "median_latency_ms": round(float(np.median(latencies)) * 1000, 4),
        "mean_latency_ms": round(float(np.mean(latencies)) * 1000, 4),
        "throughput_TOPS": round(throughput_tops, 6),
        "median_TOPS": round(median_tops, 6),
        "M": M, "K": K, "N": N,
        "ops_per_gemm": ops_per_gemm,
    }


# ====================== Decode Sequence Energy Measurement ======================

def run_decode_sequence_energy(K, N, decode_steps, warmup_sec=5, min_measure_sec=15):
    """Run decode sequences in a sustained loop for power measurement.

    Strategy:
      1. Warmup for warmup_sec
      2. Time a single decode sequence (decode_steps GEMV) for latency reporting
      3. Run sustained repetitions of decode_steps GEMV for min_measure_sec
         to give RAPL enough samples for accurate power reading
      4. Compute per-token energy from sustained power and single-sequence latency
    """
    engine = MKLInt8GEMM(1, K, N)

    # Warmup
    print(f"    Warming up ({warmup_sec}s)...")
    warmup_end = time.perf_counter() + warmup_sec
    warmup_iters = 0
    while time.perf_counter() < warmup_end:
        engine.run()
        warmup_iters += 1
    print(f"    Warmup done ({warmup_iters} iters)")

    # --- Single sequence timing (for latency) ---
    step_latencies = []
    seq_start = time.perf_counter()
    for _ in range(decode_steps):
        t0 = time.perf_counter()
        engine.run()
        step_latencies.append(time.perf_counter() - t0)
    single_seq_time = time.perf_counter() - seq_start
    step_latencies = np.array(step_latencies)

    # --- Sustained loop (for power measurement) ---
    print(f"    Sustained decode loop ({min_measure_sec}s for power)...")
    total_steps = 0
    wall_start = time.time()
    perf_start = time.perf_counter()
    deadline = perf_start + min_measure_sec
    while time.perf_counter() < deadline:
        for _ in range(decode_steps):
            engine.run()
        total_steps += decode_steps
    sustained_time = time.perf_counter() - perf_start
    wall_end = time.time()
    n_sequences = total_steps // decode_steps

    ops_per_step = 2 * 1 * K * N
    total_ops_sustained = ops_per_step * total_steps

    print(f"    Sustained: {n_sequences} full sequences, {total_steps} total steps in {sustained_time:.2f}s")

    return {
        "decode_steps": decode_steps,
        "K": K, "N": N,
        "ops_per_step": ops_per_step,
        # Single-sequence latency (for user-facing latency metrics)
        "total_latency_ms": round(single_seq_time * 1000, 4),
        "avg_per_token_latency_ms": round(float(np.mean(step_latencies)) * 1000, 4),
        "median_per_token_latency_ms": round(float(np.median(step_latencies)) * 1000, 4),
        # Sustained measurement (for power/energy)
        "sustained_time_s": round(sustained_time, 4),
        "sustained_total_steps": total_steps,
        "sustained_n_sequences": n_sequences,
        "throughput_TOPS": round(total_ops_sustained / (sustained_time * 1e12), 6),
        "median_TOPS": round(ops_per_step / (float(np.median(step_latencies)) * 1e12), 6),
        "wall_start": wall_start,
        "wall_end": wall_end,
    }


# ====================== RAPL-based Energy Measurement Wrappers ======================

def measure_with_rapl(rapl_monitor, run_fn, *args, **kwargs):
    """Wrap a benchmark function with RAPL energy sampling.
    Returns (result_dict, power_dict) where power_dict has
    {"total_W": float, "per_socket": {name: float, ...}}
    or (result_dict, None) on failure.
    """
    energy_start = rapl_monitor.read_energy_uj()
    t_start = time.perf_counter()

    result = run_fn(*args, **kwargs)

    t_end = time.perf_counter()
    energy_end = rapl_monitor.read_energy_uj()
    duration = t_end - t_start

    power_info = rapl_monitor.measure_power(energy_start, energy_end, duration)
    return result, power_info


def measure_with_perf(measure_sec, M, K, N, warmup_sec=5):
    """Run benchmark wrapped in perf stat for energy measurement.
    Uses a subprocess that calls this script in sustained mode.
    Returns (gemm_result_dict, power_watts) or None on failure.
    """
    script_path = os.path.abspath(__file__)
    # Run a sub-process benchmark under perf stat
    cmd = [
        "perf", "stat", "-e", "power/energy-pkg/", "-a", "--",
        sys.executable, "-c",
        f"import sys; sys.path.insert(0, '{os.path.dirname(script_path)}'); "
        f"from benchmark_int8_gemm import MKLInt8GEMM; "
        f"import time; "
        f"engine = MKLInt8GEMM({M}, {K}, {N}); "
        f"end = time.perf_counter() + {warmup_sec}; "
        f"c = 0\n"
        f"while time.perf_counter() < end: engine.run(); c += 1\n"
        f"end2 = time.perf_counter() + {measure_sec}\n"
        f"c2 = 0\n"
        f"while time.perf_counter() < end2: engine.run(); c2 += 1\n"
        f"print(f'iters={{c2}} time={{{measure_sec}}}')"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=warmup_sec + measure_sec + 30)
        energy_j = PerfStatMonitor.parse_energy(result.stderr)
        if energy_j is not None:
            total_time = warmup_sec + measure_sec
            power_w = energy_j / total_time
            return power_w
    except Exception as e:
        print(f"    [WARN] perf stat failed: {e}")
    return None


# ====================== Configuration ======================

# LLaMA-2-7B configs
HIDDEN = 4096
INTER = 11008

GEMM_CONFIGS = [
    # (name, phase, M, K, N)
    # --- Prefilling (GEMM) ---
    ("Attn_Proj_seq512",   "Prefilling",  512, HIDDEN, HIDDEN),
    ("FFN_GateUp_seq512",  "Prefilling",  512, HIDDEN, INTER),
    ("FFN_Down_seq512",    "Prefilling",  512, INTER,  HIDDEN),
    ("Attn_Proj_seq1024",  "Prefilling", 1024, HIDDEN, HIDDEN),
    ("FFN_GateUp_seq1024", "Prefilling", 1024, HIDDEN, INTER),
    ("FFN_Down_seq1024",   "Prefilling", 1024, INTER,  HIDDEN),
    ("Attn_Proj_seq1536",  "Prefilling", 1536, HIDDEN, HIDDEN),
    ("FFN_GateUp_seq1536", "Prefilling", 1536, HIDDEN, INTER),
    ("FFN_Down_seq1536",   "Prefilling", 1536, INTER,  HIDDEN),
    # --- Decoding (GEMV, batch=1) ---
    ("Attn_Proj_dec",   "Decoding", 1, HIDDEN, HIDDEN),
    ("FFN_GateUp_dec",  "Decoding", 1, HIDDEN, INTER),
    ("FFN_Down_dec",    "Decoding", 1, INTER,  HIDDEN),
]


def main():
    parser = argparse.ArgumentParser(
        description="CPU INT8 GEMM Energy Efficiency Test (TOPS/W) — Linux"
    )
    parser.add_argument("--power-mode", choices=["rapl", "perf", "manual", "none", "auto"],
                        default="auto",
                        help="Power measurement: rapl (sysfs), perf (perf stat), "
                             "manual (user input), none (skip power), "
                             "auto (try rapl -> perf -> none)")
    parser.add_argument("--warmup-sec", type=int, default=5,
                        help="Warmup duration per config (seconds)")
    parser.add_argument("--measure-sec", type=int, default=30,
                        help="Measurement duration per config (seconds)")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory")
    parser.add_argument("--configs", type=str, default="all",
                        help="Comma-separated config indices or 'all'")
    parser.add_argument("--decode-steps", type=str, default="",
                        help="Comma-separated decode sequence lengths, e.g. '512,1024,1536'. "
                             "Measures energy per-token (J/token) for multi-step decoding.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Select configs
    if args.configs == "all":
        configs = GEMM_CONFIGS
    else:
        indices = [int(x) for x in args.configs.split(",")]
        configs = [GEMM_CONFIGS[i] for i in indices]

    print("=" * 78)
    print("  CPU INT8 GEMM Energy Efficiency Test (TOPS/W) — Linux")
    print("  Backend: MKL INT8 (AMX-INT8 / AVX-512 VNNI)")
    print_system_info()
    print(f"  Warmup: {args.warmup_sec}s | Measure: {args.measure_sec}s per config")
    print(f"  Configs: {len(configs)} GEMM shapes")
    print("=" * 78)

    # ---- Select power measurement backend ----
    rapl_monitor = None
    use_perf = False
    use_manual = False

    if args.power_mode == "auto":
        # Try RAPL sysfs
        rapl_monitor = LinuxRAPLMonitor()
        if rapl_monitor.available():
            print("  Power mode: RAPL sysfs (auto-detected)")
        elif PerfStatMonitor.available():
            rapl_monitor = None
            use_perf = True
            print("  Power mode: perf stat (auto-detected)")
        else:
            rapl_monitor = None
            print("  Power mode: NONE (RAPL and perf unavailable, TOPS-only)")
            print("  TIP: Ask admin to enable read access to Linux RAPL energy counters.")
            print("  Or use --power-mode manual for interactive power input")
    elif args.power_mode == "rapl":
        rapl_monitor = LinuxRAPLMonitor()
        if not rapl_monitor.available():
            print("  [ERROR] RAPL sysfs counters are not readable.")
            print("  Falling back to manual mode.")
            rapl_monitor = None
            use_manual = True
    elif args.power_mode == "perf":
        if PerfStatMonitor.available():
            use_perf = True
        else:
            print("  [ERROR] perf stat power events not accessible (perf_event_paranoid > 0).")
            print("  Falling back to manual mode.")
            use_manual = True
    elif args.power_mode == "manual":
        use_manual = True
    elif args.power_mode == "none":
        pass  # No power measurement

    manual_input = ManualPowerInput() if use_manual else None

    all_results = []

    for idx, (name, phase, M, K, N) in enumerate(configs):
        print(f"\n{'━'*70}")
        print(f"  [{idx+1}/{len(configs)}] {name}  |  [{M}, {K}] x [{K}, {N}]  |  {phase}")
        print(f"{'━'*70}")

        power_info = None  # dict {"total_W", "per_socket"} or scalar or None

        if rapl_monitor and rapl_monitor.available():
            gemm_result, power_info = measure_with_rapl(
                rapl_monitor, run_sustained_gemm,
                M, K, N,
                warmup_sec=args.warmup_sec,
                measure_sec=args.measure_sec,
            )
        elif use_perf:
            gemm_result = run_sustained_gemm(
                M, K, N,
                warmup_sec=args.warmup_sec,
                measure_sec=args.measure_sec,
            )
            perf_w = measure_with_perf(args.measure_sec, M, K, N, warmup_sec=args.warmup_sec)
            if perf_w is not None:
                power_info = {"total_W": round(perf_w, 3), "per_socket": {}}
        else:
            gemm_result = run_sustained_gemm(
                M, K, N,
                warmup_sec=args.warmup_sec,
                measure_sec=args.measure_sec,
            )
            if manual_input:
                manual_w = manual_input.get_power(name)
                if manual_w is not None:
                    power_info = {"total_W": round(manual_w, 3), "per_socket": {}}

        # Compute energy efficiency
        result = {
            "name": name,
            "phase": phase,
            "backend": "MKL_INT8_AMX",
            **gemm_result,
        }

        if power_info is not None and power_info["total_W"] > 0:
            total_w = power_info["total_W"]
            per_socket = power_info.get("per_socket", {})
            tops_w = gemm_result["throughput_TOPS"] / total_w
            median_tops_w = gemm_result["median_TOPS"] / total_w
            source = "rapl_sysfs" if rapl_monitor else ("perf_stat" if use_perf else "manual")
            result["power"] = {
                "mean_power_W": total_w,
                "per_socket_W": per_socket,
                "source": source,
            }
            result["TOPS_per_W"] = round(tops_w, 6)
            result["median_TOPS_per_W"] = round(median_tops_w, 6)
            result["total_energy_J"] = round(total_w * gemm_result["total_time_s"], 3)

            print(f"    Throughput TOPS     : {gemm_result['throughput_TOPS']:.6f}")
            print(f"    Median TOPS        : {gemm_result['median_TOPS']:.6f}")
            print(f"    Total Package Power: {total_w:.2f} W")
            for sock_name, sock_w in sorted(per_socket.items()):
                print(f"      {sock_name:>12}   : {sock_w:.2f} W")
            print(f"    Energy Efficiency  : {tops_w:.6f} TOPS/W")
            print(f"    Median TOPS/W      : {median_tops_w:.6f}")
            print(f"    Total Energy       : {result['total_energy_J']:.3f} J")
        else:
            result["power"] = None
            result["TOPS_per_W"] = None
            result["median_TOPS_per_W"] = None
            print(f"    Throughput TOPS    : {gemm_result['throughput_TOPS']:.6f}")
            print(f"    Median TOPS       : {gemm_result['median_TOPS']:.6f}")
            print(f"    [WARN] No power data available.")

        all_results.append(result)

    # ---- Save results ----
    json_path = os.path.join(args.output_dir, "energy_efficiency.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {json_path}")

    csv_path = os.path.join(args.output_dir, "energy_efficiency.csv")
    csv_fields = ["name", "phase", "M", "K", "N", "backend",
                  "throughput_TOPS", "median_TOPS",
                  "median_latency_ms", "mean_latency_ms",
                  "TOPS_per_W", "median_TOPS_per_W", "total_energy_J"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
    print(f"CSV saved: {csv_path}")

    # ---- Print final summary ----
    print("\n" + "=" * 140)
    print(f"{'Name':<25} {'Phase':<11} {'Shape':<26} "
          f"{'TOPS':<10} {'Total(W)':<10} {'Sock0(W)':<10} {'Sock1(W)':<10} "
          f"{'TOPS/W':<10} {'Lat(ms)':<10}")
    print("-" * 140)
    for r in all_results:
        shape = f"[{r['M']},{r['K']}]x[{r['K']},{r['N']}]"
        pw = r.get("power") or {}
        total_str = f"{pw['mean_power_W']:.1f}" if pw and "mean_power_W" in pw else "N/A"
        ps = pw.get("per_socket_W", {}) if pw else {}
        s0_str = f"{ps.get('package-0', 0):.1f}" if ps.get("package-0") else "N/A"
        s1_str = f"{ps.get('package-1', 0):.1f}" if ps.get("package-1") else "N/A"
        topsw_str = f"{r['TOPS_per_W']:.6f}" if r.get("TOPS_per_W") else "N/A"
        print(f"{r['name']:<25} {r['phase']:<11} {shape:<26} "
              f"{r['median_TOPS']:<10.6f} {total_str:<10} {s0_str:<10} {s1_str:<10} "
              f"{topsw_str:<10} {r['median_latency_ms']:<10.4f}")
    print("=" * 140)

    # ====================== Decode Sequence Energy Measurement ======================
    if args.decode_steps:
        decode_step_list = [int(x.strip()) for x in args.decode_steps.split(",")]
        decode_gemm_shapes = [
            ("Attn_Proj",  HIDDEN, HIDDEN),
            ("FFN_GateUp", HIDDEN, INTER),
            ("FFN_Down",   INTER,  HIDDEN),
        ]

        decode_seq_results = []
        print("\n" + "=" * 78)
        print("  Decode Sequence Energy Test (multi-step GEMV with power)")
        print("=" * 78)

        for steps in decode_step_list:
            for layer_name, K, N in decode_gemm_shapes:
                cfg_name = f"{layer_name}_dec{steps}"
                print(f"\n{'━'*70}")
                print(f"  {cfg_name}  |  [1,{K}]x[{K},{N}]  |  {steps} decode steps")
                print(f"{'━'*70}")

                power_info = None

                if rapl_monitor and rapl_monitor.available():
                    dr, power_info = measure_with_rapl(
                        rapl_monitor, run_decode_sequence_energy,
                        K, N, steps, warmup_sec=args.warmup_sec,
                    )
                else:
                    dr = run_decode_sequence_energy(
                        K, N, steps, warmup_sec=args.warmup_sec,
                    )
                    if manual_input:
                        manual_w = manual_input.get_power(cfg_name)
                        if manual_w is not None:
                            power_info = {"total_W": round(manual_w, 3), "per_socket": {}}

                dr["name"] = cfg_name
                dr["layer"] = layer_name
                dr["backend"] = "MKL_INT8_AMX"

                if power_info is not None and power_info["total_W"] > 0:
                    total_w = power_info["total_W"]
                    per_socket = power_info.get("per_socket", {})
                    seq_energy_J = total_w * (dr["total_latency_ms"] / 1000.0)
                    energy_per_token_J = total_w * (dr["avg_per_token_latency_ms"] / 1000.0)
                    tops_w = dr["throughput_TOPS"] / total_w

                    dr["power"] = {
                        "mean_power_W": total_w,
                        "per_socket_W": per_socket,
                    }
                    dr["seq_energy_J"] = round(seq_energy_J, 4)
                    dr["energy_per_token_J"] = round(energy_per_token_J, 6)
                    dr["energy_per_token_mJ"] = round(energy_per_token_J * 1000, 4)
                    dr["TOPS_per_W"] = round(tops_w, 6)

                    print(f"    Total Generation Latency  : {dr['total_latency_ms']:.2f} ms")
                    print(f"    Avg Per-Token Latency     : {dr['avg_per_token_latency_ms']:.4f} ms")
                    print(f"    Total Package Power       : {total_w:.2f} W")
                    for sock_name, sock_w in sorted(per_socket.items()):
                        print(f"      {sock_name:>12}       : {sock_w:.2f} W")
                    print(f"    Sequence Energy           : {seq_energy_J:.4f} J ({steps} tokens)")
                    print(f"    Energy Per Token          : {energy_per_token_J*1000:.4f} mJ/token")
                    print(f"    TOPS/W                    : {tops_w:.6f}")
                else:
                    dr["power"] = None
                    dr["seq_energy_J"] = None
                    dr["energy_per_token_J"] = None
                    dr["energy_per_token_mJ"] = None
                    dr["TOPS_per_W"] = None
                    print(f"    Total Generation Latency  : {dr['total_latency_ms']:.2f} ms")
                    print(f"    Avg Per-Token Latency     : {dr['avg_per_token_latency_ms']:.4f} ms")
                    print(f"    Throughput TOPS           : {dr['throughput_TOPS']:.6f}")
                    print(f"    [WARN] No power data available.")

                decode_seq_results.append(dr)

        # Save decode-sequence energy results
        ds_json = os.path.join(args.output_dir, "decode_sequence_energy.json")
        ds_csv = os.path.join(args.output_dir, "decode_sequence_energy.csv")
        with open(ds_json, "w") as f:
            json.dump(decode_seq_results, f, indent=2, default=str)
        print(f"\nDecode-sequence energy JSON saved: {ds_json}")

        csv_fields = ["name", "layer", "decode_steps", "K", "N", "backend",
                      "total_latency_ms", "avg_per_token_latency_ms",
                      "median_per_token_latency_ms", "throughput_TOPS", "median_TOPS",
                      "TOPS_per_W", "seq_energy_J", "energy_per_token_J",
                      "energy_per_token_mJ"]
        with open(ds_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(decode_seq_results)
        print(f"Decode-sequence energy CSV saved: {ds_csv}")

        # Print decode-sequence summary
        print("\n" + "=" * 150)
        print(f"{'Name':<25} {'Steps':<7} {'Shape':<24} "
              f"{'TotalLat(ms)':<14} {'PerTok(ms)':<12} {'Total(W)':<10} "
              f"{'Sock0(W)':<10} {'Sock1(W)':<10} {'mJ/token':<10} {'TOPS/W':<10}")
        print("-" * 150)
        for dr in decode_seq_results:
            shape = f"[1,{dr['K']}]x[{dr['K']},{dr['N']}]"
            pw = dr.get("power") or {}
            total_str = f"{pw['mean_power_W']:.1f}" if pw and "mean_power_W" in pw else "N/A"
            ps = pw.get("per_socket_W", {}) if pw else {}
            s0_str = f"{ps.get('package-0', 0):.1f}" if ps.get("package-0") else "N/A"
            s1_str = f"{ps.get('package-1', 0):.1f}" if ps.get("package-1") else "N/A"
            mj_str = f"{dr['energy_per_token_mJ']:.4f}" if dr.get('energy_per_token_mJ') else "N/A"
            topsw_str = f"{dr['TOPS_per_W']:.6f}" if dr.get('TOPS_per_W') else "N/A"
            print(f"{dr['name']:<25} {dr['decode_steps']:<7} {shape:<24} "
                  f"{dr['total_latency_ms']:<14.2f} {dr['avg_per_token_latency_ms']:<12.4f} "
                  f"{total_str:<10} {s0_str:<10} {s1_str:<10} {mj_str:<10} {topsw_str:<10}")
        print("=" * 150)

    print("\nDone.")


if __name__ == "__main__":
    main()
