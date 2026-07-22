"""
CPU package power monitor using Linux RAPL sysfs counters.
"""

from __future__ import annotations

import glob


class LinuxRAPLMonitor:
    """Read per-package CPU package energy and convert it to power."""

    def __init__(self):
        self.rapl_paths: list[str] = []
        self.max_energy: dict[str, int] = {}
        self._discover()

    def _discover(self):
        pattern = "/sys/class/powercap/intel-rapl/intel-rapl:*/energy_uj"
        for path in sorted(glob.glob(pattern)):
            try:
                with open(path) as f:
                    f.read()
            except (FileNotFoundError, PermissionError):
                continue

            self.rapl_paths.append(path)
            max_path = path.replace("energy_uj", "max_energy_range_uj")
            try:
                with open(max_path) as f:
                    self.max_energy[path] = int(f.read().strip())
            except Exception:
                self.max_energy[path] = 2**63

    def available(self) -> bool:
        return len(self.rapl_paths) > 0

    def read_energy_uj(self) -> dict[str, int]:
        readings = {}
        for path in self.rapl_paths:
            with open(path) as f:
                readings[path] = int(f.read().strip())
        return readings

    def measure_power(self, start_energy: dict[str, int], end_energy: dict[str, int], duration_s: float) -> dict:
        total_delta_uj = 0
        per_socket = {}

        for path in self.rapl_paths:
            start_val = start_energy[path]
            end_val = end_energy[path]
            delta = end_val - start_val
            if delta < 0:
                delta += self.max_energy[path]
            total_delta_uj += delta

            name_path = path.replace("energy_uj", "name")
            try:
                with open(name_path) as f:
                    socket_name = f.read().strip()
            except Exception:
                socket_name = path

            per_socket[socket_name] = round((delta / 1e6) / duration_s, 3)

        total_w = (total_delta_uj / 1e6) / duration_s if duration_s > 0 else 0.0
        return {
            "total_W": round(total_w, 3),
            "per_socket": per_socket,
            "source": "rapl_sysfs",
        }
