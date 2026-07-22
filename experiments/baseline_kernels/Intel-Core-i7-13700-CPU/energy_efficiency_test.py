"""
Energy Efficiency Test (TOPS/W) for CPU INT8 GEMM
===================================================
Combines INT8 GEMM benchmark with RAPL power measurement via Intel PCM.

Flow:
  1. Start Intel PCM power logging in background
  2. Warmup GEMM (5s)
  3. Record RAPL energy_start via PCM
  4. Run sustained GEMM for measurement_duration
  5. Record RAPL energy_end via PCM
  6. Compute: Power = (energy_end - energy_start) / duration
  7. Compute: TOPS/W = TOPS / Power

Requirements:
  - Intel PCM installed (see README.md)
  - Run as Administrator (required for RAPL access)
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
import signal
import numpy as np

# ====================== Windows RAPL Power Monitor ======================
# Reads Intel RAPL via Windows Performance Counter "Energy Meter"
# Counter: \Energy Meter(rapl_package0_pkg)\Power  (unit: milliwatts)
# Uses built-in 'typeperf' for continuous background sampling — no external
# tools required.

RAPL_COUNTER = r"\Energy Meter(rapl_package0_pkg)\Power"


class WindowsRAPLMonitor:
    """Read CPU package power via Windows Performance Counter (RAPL).
    Uses 'typeperf' for efficient continuous sampling.
    Power unit from the counter is milliwatts; converted to Watts internally.
    """

    def __init__(self, sample_interval=1):
        self.sample_interval = sample_interval
        self.process = None
        self.reader_thread = None
        self.power_readings = []  # list of (wall_timestamp, power_watts)
        self._stop_event = threading.Event()

    def start(self):
        """Start typeperf in background to log RAPL power."""
        cmd = [
            "typeperf", RAPL_COUNTER,
            "-si", str(self.sample_interval),
        ]
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        except Exception as e:
            print(f"  [ERROR] Failed to start typeperf: {e}")
            return False

        self._stop_event.clear()
        self.power_readings = []
        self.reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.reader_thread.start()
        time.sleep(self.sample_interval + 0.5)  # wait for first sample
        print(f"  RAPL monitor started (typeperf, {self.sample_interval}s interval)")
        return True

    def _read_loop(self):
        """Parse typeperf CSV output lines.
        Format:
          "(PDH-CSV 4.0)","\\machine\\Energy Meter(...)\\Power"
          "03/16/2026 15:30:00.000","52070.123"
        """
        try:
            for line in iter(self.process.stdout.readline, ""):
                if self._stop_event.is_set():
                    break
                line = line.strip()
                if not line or line.startswith('"(PDH'):
                    continue
                # Parse CSV: "timestamp","value"
                parts = line.split('","')
                if len(parts) >= 2:
                    try:
                        val_str = parts[-1].strip().strip('"')
                        if val_str and val_str != " ":
                            mw = float(val_str)
                            watts = mw / 1000.0  # milliwatts -> watts
                            self.power_readings.append((time.time(), watts))
                    except (ValueError, IndexError):
                        pass
        except Exception:
            pass

    def stop(self):
        """Stop typeperf process."""
        self._stop_event.set()
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
        if self.reader_thread:
            self.reader_thread.join(timeout=5)
        print(f"  RAPL monitor stopped ({len(self.power_readings)} samples collected)")

    def get_power_stats(self, start_time=None, end_time=None):
        """Get power statistics for a time window."""
        readings = self.power_readings
        if start_time is not None:
            readings = [(t, p) for t, p in readings if t >= start_time]
        if end_time is not None:
            readings = [(t, p) for t, p in readings if t <= end_time]
        if not readings:
            return None
        powers = [p for _, p in readings]
        return {
            "num_samples": len(powers),
            "mean_power_W": round(float(np.mean(powers)), 3),
            "median_power_W": round(float(np.median(powers)), 3),
            "min_power_W": round(float(np.min(powers)), 3),
            "max_power_W": round(float(np.max(powers)), 3),
            "std_power_W": round(float(np.std(powers)), 3),
        }


class ManualPowerInput:
    """Fallback: user provides average power reading from external tool."""

    def start(self):
        print("  [Manual Mode] Start your power monitoring tool (HWiNFO64, etc.)")
        input("  Press Enter when ready...")
        return True

    def stop(self):
        print("  [Manual Mode] Benchmark complete.")

    def get_power_stats(self, start_time=None, end_time=None):
        try:
            power = float(input("  Enter average CPU package power in Watts: "))
            return {
                "num_samples": 1,
                "mean_power_W": power,
                "median_power_W": power,
                "min_power_W": power,
                "max_power_W": power,
                "std_power_W": 0.0,
            }
        except (ValueError, EOFError):
            return None


# ====================== GEMM Engine (MKL INT8 VNNI) ======================

from benchmark_int8_gemm import MKLInt8GEMM


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
        description="CPU INT8 GEMM Energy Efficiency Test (TOPS/W)"
    )
    parser.add_argument("--power-mode", choices=["rapl", "manual"], default="rapl",
                        help="Power measurement: rapl (Windows counter) or manual")
    parser.add_argument("--warmup-sec", type=int, default=5,
                        help="Warmup duration per config (seconds)")
    parser.add_argument("--measure-sec", type=int, default=30,
                        help="Measurement duration per config (seconds)")
    parser.add_argument("--sample-interval", type=int, default=1,
                        help="RAPL sampling interval in seconds (default: 1)")
    parser.add_argument("--output", type=str, default="results/energy_efficiency.json")
    parser.add_argument("--configs", type=str, default="all",
                        help="Comma-separated config indices or 'all'")
    parser.add_argument("--decode-steps", type=str, default="",
                        help="Comma-separated decode sequence lengths, e.g. '512,1024,1536'. "
                             "Measures energy per-token (J/token) for multi-step decoding.")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    # Select configs
    if args.configs == "all":
        configs = GEMM_CONFIGS
    else:
        indices = [int(x) for x in args.configs.split(",")]
        configs = [GEMM_CONFIGS[i] for i in indices]

    print("=" * 78)
    print("  CPU INT8 GEMM Energy Efficiency Test (TOPS/W)")
    print(f"  Backend: MKL INT8 VNNI | Power: {args.power_mode}")
    print(f"  Warmup: {args.warmup_sec}s | Measure: {args.measure_sec}s per config")
    print(f"  Configs: {len(configs)} GEMM shapes")
    print("=" * 78)

    # Initialize power monitor
    if args.power_mode == "rapl":
        monitor = WindowsRAPLMonitor(sample_interval=args.sample_interval)
    else:
        monitor = ManualPowerInput()

    all_results = []

    for name, phase, M, K, N in configs:
        print(f"\n{'━'*70}")
        print(f"  {name}  |  [{M}, {K}] x [{K}, {N}]  |  {phase}")
        print(f"{'━'*70}")

        # Start power monitoring
        ok = monitor.start()
        if not ok and args.power_mode == "rapl":
            print("  [WARN] RAPL failed. Falling back to manual mode.")
            monitor = ManualPowerInput()
            monitor.start()

        # Run sustained GEMM
        gemm_result = run_sustained_gemm(
            M, K, N,
            warmup_sec=args.warmup_sec,
            measure_sec=args.measure_sec,
        )

        # Get power stats
        monitor.stop()
        power_stats = monitor.get_power_stats(
            start_time=gemm_result["wall_start"],
            end_time=gemm_result["wall_end"],
        )

        # Compute energy efficiency
        result = {
            "name": name,
            "phase": phase,
            "backend": "MKL_INT8_VNNI",
            **gemm_result,
        }

        if power_stats and power_stats["mean_power_W"] > 0:
            avg_power = power_stats["mean_power_W"]
            tops_w = gemm_result["throughput_TOPS"] / avg_power
            median_tops_w = gemm_result["median_TOPS"] / avg_power
            result["power"] = power_stats
            result["TOPS_per_W"] = round(tops_w, 6)
            result["median_TOPS_per_W"] = round(median_tops_w, 6)
            result["total_energy_J"] = round(avg_power * gemm_result["total_time_s"], 3)

            print(f"    Throughput TOPS     : {gemm_result['throughput_TOPS']:.6f}")
            print(f"    Median TOPS        : {gemm_result['median_TOPS']:.6f}")
            print(f"    Avg Package Power  : {avg_power:.2f} W")
            print(f"    Energy Efficiency  : {tops_w:.6f} TOPS/W")
            print(f"    Median TOPS/W      : {median_tops_w:.6f}")
            print(f"    Total Energy       : {result['total_energy_J']:.3f} J")
        else:
            result["power"] = None
            result["TOPS_per_W"] = None
            print(f"    Throughput TOPS    : {gemm_result['throughput_TOPS']:.6f}")
            print(f"    [WARN] No power data available.")

        all_results.append(result)

    # Save results
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {args.output}")

    # Also save CSV summary
    csv_path = args.output.replace(".json", ".csv")
    csv_fields = ["name", "phase", "M", "K", "N", "backend",
                  "throughput_TOPS", "median_TOPS",
                  "median_latency_ms", "mean_latency_ms",
                  "TOPS_per_W", "median_TOPS_per_W", "total_energy_J"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
    print(f"CSV saved: {csv_path}")

    # Print final summary
    print("\n" + "=" * 110)
    print(f"{'Name':<25} {'Phase':<11} {'Shape':<26} "
          f"{'TOPS':<10} {'Power(W)':<10} {'TOPS/W':<10} {'Lat(ms)':<10}")
    print("-" * 110)
    for r in all_results:
        shape = f"[{r['M']},{r['K']}]x[{r['K']},{r['N']}]"
        pw = r.get("power", {})
        power_str = f"{pw['mean_power_W']:.1f}" if pw else "N/A"
        topsw_str = f"{r['TOPS_per_W']:.6f}" if r.get("TOPS_per_W") else "N/A"
        print(f"{r['name']:<25} {r['phase']:<11} {shape:<26} "
              f"{r['median_TOPS']:<10.6f} {power_str:<10} {topsw_str:<10} "
              f"{r['median_latency_ms']:<10.4f}")
    print("=" * 110)

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
                print(f"\n{'\u2501'*70}")
                print(f"  {cfg_name}  |  [1,{K}]x[{K},{N}]  |  {steps} decode steps")
                print(f"{'\u2501'*70}")

                # Start power monitoring
                if args.power_mode == "rapl":
                    ds_monitor = WindowsRAPLMonitor(sample_interval=args.sample_interval)
                else:
                    ds_monitor = ManualPowerInput()
                ok = ds_monitor.start()
                if not ok and args.power_mode == "rapl":
                    ds_monitor = ManualPowerInput()
                    ds_monitor.start()

                # Run decode sequence
                dr = run_decode_sequence_energy(K, N, steps, warmup_sec=args.warmup_sec)
                dr["name"] = cfg_name
                dr["layer"] = layer_name
                dr["backend"] = "MKL_INT8_VNNI"

                # Get power stats
                ds_monitor.stop()
                power_stats = ds_monitor.get_power_stats(
                    start_time=dr["wall_start"],
                    end_time=dr["wall_end"],
                )

                if power_stats and power_stats["mean_power_W"] > 0:
                    avg_power = power_stats["mean_power_W"]
                    # Total energy for one decode sequence
                    seq_energy_J = avg_power * (dr["total_latency_ms"] / 1000.0)
                    # Per-token energy = power × per-token latency
                    energy_per_token_J = avg_power * (dr["avg_per_token_latency_ms"] / 1000.0)
                    tops_w = dr["throughput_TOPS"] / avg_power

                    dr["power"] = power_stats
                    dr["seq_energy_J"] = round(seq_energy_J, 4)
                    dr["energy_per_token_J"] = round(energy_per_token_J, 6)
                    dr["energy_per_token_mJ"] = round(energy_per_token_J * 1000, 4)
                    dr["TOPS_per_W"] = round(tops_w, 6)

                    print(f"    Total Generation Latency  : {dr['total_latency_ms']:.2f} ms")
                    print(f"    Avg Per-Token Latency     : {dr['avg_per_token_latency_ms']:.4f} ms")
                    print(f"    Avg Package Power         : {avg_power:.2f} W")
                    print(f"    Sequence Energy           : {seq_energy_J:.4f} J ({steps} tokens)")
                    print(f"    Energy Per Token          : {energy_per_token_J*1000:.4f} mJ/token")
                    print(f"    TOPS/W                    : {tops_w:.6f}")
                else:
                    dr["power"] = None
                    dr["total_energy_J"] = None
                    dr["energy_per_token_J"] = None
                    dr["energy_per_token_mJ"] = None
                    dr["TOPS_per_W"] = None
                    print(f"    Total Generation Latency  : {dr['total_latency_ms']:.2f} ms")
                    print(f"    [WARN] No power data available.")

                decode_seq_results.append(dr)

        # Save decode-sequence energy results
        ds_json = "results/decode_sequence_energy.json"
        ds_csv = "results/decode_sequence_energy.csv"
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
        print("\n" + "=" * 120)
        print(f"{'Name':<25} {'Steps':<7} {'Shape':<24} "
              f"{'TotalLat(ms)':<14} {'PerTok(ms)':<12} {'Power(W)':<10} "
              f"{'mJ/token':<10} {'TOPS/W':<10}")
        print("-" * 120)
        for dr in decode_seq_results:
            shape = f"[1,{dr['K']}]x[{dr['K']},{dr['N']}]"
            pw = dr.get("power", {})
            power_str = f"{pw['mean_power_W']:.1f}" if pw else "N/A"
            mj_str = f"{dr['energy_per_token_mJ']:.4f}" if dr.get('energy_per_token_mJ') else "N/A"
            topsw_str = f"{dr['TOPS_per_W']:.6f}" if dr.get('TOPS_per_W') else "N/A"
            print(f"{dr['name']:<25} {dr['decode_steps']:<7} {shape:<24} "
                  f"{dr['total_latency_ms']:<14.2f} {dr['avg_per_token_latency_ms']:<12.4f} "
                  f"{power_str:<10} {mj_str:<10} {topsw_str:<10}")
        print("=" * 120)

    print("\nDone.")


if __name__ == "__main__":
    main()
