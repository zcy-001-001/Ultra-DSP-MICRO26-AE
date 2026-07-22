"""
Improved RAPL Power Monitor for Windows
========================================
Reads multiple Intel RAPL domains via Windows Performance Counters:
  - PKG  (RAPL_Package0_PKG)  : Total CPU package power
  - PP0  (RAPL_Package0_PP0)  : CPU cores power (compute)
  - PP1  (RAPL_Package0_PP1)  : Integrated GPU / uncore
  - DRAM (RAPL_Package0_DRAM) : DRAM power (data movement)

Features:
  1. Multi-domain simultaneous sampling via typeperf
  2. Idle power baseline measurement (subtract background load)
  3. Compute vs data-movement power decoupling:
       compute_power = PP0_loaded - PP0_idle
       memory_power  = DRAM_loaded - DRAM_idle
       total_power   = PKG_loaded - PKG_idle
  4. Background sampling with configurable interval
"""

import subprocess
import threading
import time
import numpy as np

# RAPL counter names on Windows
RAPL_COUNTERS = {
    "PKG":  r"\Energy Meter(RAPL_Package0_PKG)\Power",
    "PP0":  r"\Energy Meter(RAPL_Package0_PP0)\Power",
    "PP1":  r"\Energy Meter(RAPL_Package0_PP1)\Power",
    "DRAM": r"\Energy Meter(RAPL_Package0_DRAM)\Power",
}


class MultiDomainRAPLMonitor:
    """Read CPU power from multiple RAPL domains via Windows typeperf.
    Power unit from the counter is milliwatts; converted to Watts internally.
    """

    def __init__(self, sample_interval=1, domains=None):
        self.sample_interval = sample_interval
        self.domains = domains or ["PKG", "PP0", "DRAM"]
        self.process = None
        self.reader_thread = None
        self._stop_event = threading.Event()
        # readings[domain] = list of (wall_timestamp, power_watts)
        self.readings = {d: [] for d in self.domains}
        self._counter_order = []  # order of counters in typeperf output

    def start(self):
        """Start typeperf in background to log RAPL power for all domains."""
        counters = []
        self._counter_order = []
        for d in self.domains:
            if d in RAPL_COUNTERS:
                counters.append(RAPL_COUNTERS[d])
                self._counter_order.append(d)

        if not counters:
            print("  [ERROR] No valid RAPL counters specified")
            return False

        cmd = ["typeperf"] + counters + ["-si", str(self.sample_interval)]
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
        self.readings = {d: [] for d in self.domains}
        self.reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.reader_thread.start()
        time.sleep(self.sample_interval + 0.5)
        print(f"  RAPL monitor started ({', '.join(self._counter_order)}, "
              f"{self.sample_interval}s interval)")
        return True

    def _read_loop(self):
        """Parse typeperf CSV output: multiple counter values per line."""
        try:
            for line in iter(self.process.stdout.readline, ""):
                if self._stop_event.is_set():
                    break
                line = line.strip()
                if not line or line.startswith('"(PDH'):
                    continue
                parts = line.split('","')
                if len(parts) >= 1 + len(self._counter_order):
                    ts = time.time()
                    for idx, domain in enumerate(self._counter_order):
                        try:
                            val_str = parts[1 + idx].strip().strip('"')
                            if val_str and val_str.strip() != "":
                                mw = float(val_str)
                                watts = mw / 1000.0
                                self.readings[domain].append((ts, watts))
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
        total = sum(len(v) for v in self.readings.values())
        print(f"  RAPL monitor stopped ({total} total samples)")

    def get_domain_stats(self, domain, start_time=None, end_time=None):
        """Get power statistics for a specific domain in a time window."""
        if domain not in self.readings:
            return None
        r = self.readings[domain]
        if start_time is not None:
            r = [(t, p) for t, p in r if t >= start_time]
        if end_time is not None:
            r = [(t, p) for t, p in r if t <= end_time]
        if not r:
            return None
        powers = [p for _, p in r]
        return {
            "num_samples": len(powers),
            "mean_power_W": round(float(np.mean(powers)), 3),
            "median_power_W": round(float(np.median(powers)), 3),
            "min_power_W": round(float(np.min(powers)), 3),
            "max_power_W": round(float(np.max(powers)), 3),
            "std_power_W": round(float(np.std(powers)), 3),
        }

    def get_all_stats(self, start_time=None, end_time=None):
        """Get power statistics for all domains."""
        result = {}
        for d in self.domains:
            stats = self.get_domain_stats(d, start_time, end_time)
            if stats:
                result[d] = stats
        return result


def measure_idle_power(duration=10, sample_interval=1, domains=None):
    """Measure idle CPU power for the specified duration.
    Returns dict of {domain: mean_power_W}.
    """
    domains = domains or ["PKG", "PP0", "DRAM"]
    print(f"\n  Measuring idle power ({duration}s, domains={domains})...")
    monitor = MultiDomainRAPLMonitor(
        sample_interval=sample_interval, domains=domains
    )
    ok = monitor.start()
    if not ok:
        return {d: 0.0 for d in domains}

    time.sleep(duration)
    monitor.stop()

    idle_power = {}
    for d in domains:
        stats = monitor.get_domain_stats(d)
        if stats:
            idle_power[d] = stats["mean_power_W"]
            print(f"    Idle {d}: {stats['mean_power_W']:.2f} W "
                  f"(±{stats['std_power_W']:.2f})")
        else:
            idle_power[d] = 0.0
            print(f"    Idle {d}: N/A")
    return idle_power


def compute_power_breakdown(loaded_stats, idle_power):
    """Compute power breakdown: total, compute, memory, other.

    Args:
        loaded_stats: dict from get_all_stats() during GEMM
        idle_power: dict from measure_idle_power()

    Returns:
        dict with power breakdown
    """
    result = {}

    # Total package power (GEMM-specific)
    pkg_loaded = loaded_stats.get("PKG", {}).get("mean_power_W", 0)
    pkg_idle = idle_power.get("PKG", 0)
    result["total_power_W"] = round(max(0, pkg_loaded - pkg_idle), 3)
    result["pkg_raw_W"] = round(pkg_loaded, 3)

    # Compute power (CPU cores, GEMM-specific)
    pp0_loaded = loaded_stats.get("PP0", {}).get("mean_power_W", 0)
    pp0_idle = idle_power.get("PP0", 0)
    result["compute_power_W"] = round(max(0, pp0_loaded - pp0_idle), 3)
    result["pp0_raw_W"] = round(pp0_loaded, 3)

    # Memory/data-movement power (DRAM, GEMM-specific)
    dram_loaded = loaded_stats.get("DRAM", {}).get("mean_power_W", 0)
    dram_idle = idle_power.get("DRAM", 0)
    result["memory_power_W"] = round(max(0, dram_loaded - dram_idle), 3)
    result["dram_raw_W"] = round(dram_loaded, 3)

    # Other (uncore, memory controller, etc.)
    result["other_power_W"] = round(
        max(0, result["total_power_W"] - result["compute_power_W"]
            - result["memory_power_W"]), 3
    )

    # Idle baselines
    result["idle_PKG_W"] = round(pkg_idle, 3)
    result["idle_PP0_W"] = round(pp0_idle, 3)
    result["idle_DRAM_W"] = round(dram_idle, 3)

    return result
