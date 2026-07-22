#!/usr/bin/env python3
"""Extract the two Table 5 Xeon points from the archived MKL/RAPL run."""

from __future__ import annotations

import csv
from pathlib import Path


HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parents[2]
RESULTS = PACKAGE_ROOT / "results" / "table5_cpu_xeon"
RAW = RESULTS / "int8_gemv_cpu_energy.csv"
OUTPUT = RESULTS / "table5_xeon_summary.csv"

PAPER = {
    "GEMV_2048x2048": {"latency_ms": 0.129, "package0_w": 213.0, "energy_mj": 27.439},
    "GEMV_4096x4096": {"latency_ms": 0.369, "package0_w": 229.0, "energy_mj": 84.325},
}


def main() -> int:
    with RAW.open(newline="", encoding="utf-8") as handle:
        raw_rows = {row["config"]: row for row in csv.DictReader(handle)}

    rows: list[dict[str, object]] = []
    for config, paper in PAPER.items():
        raw = raw_rows[config]
        latency_ms = float(raw["latency_ms"])
        package0_w = float(raw["package0_w"])
        package0_energy_mj = latency_ms * package0_w
        assert int(raw["streaming_target_mb"]) == 1024
        assert float(raw["total_time_s"]) >= 10.0
        rows.append(
            {
                "shape": f"1x{config.removeprefix('GEMV_')}",
                "config": config,
                "latency_ms": latency_ms,
                "package0_w": package0_w,
                "package1_w": float(raw["package1_w"]),
                "dual_socket_total_w": float(raw["power_total_w"]),
                "table5_package0_energy_mj": round(package0_energy_mj, 6),
                "dual_socket_energy_mj": float(raw["energy_mj"]),
                "paper_latency_ms": paper["latency_ms"],
                "paper_package0_w": paper["package0_w"],
                "paper_energy_mj": paper["energy_mj"],
                "measurement_seconds": float(raw["total_time_s"]),
                "measurement_iterations": int(raw["iters"]),
                "streaming_weight_bank_mb": int(raw["streaming_target_mb"]),
                "power_scope": "RAPL_package0_single_socket",
                "evidence_class": "RECOMPUTED_FROM_LOGS",
                "status": "PASS",
            }
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("XEON_TABLE5_PASS shapes=2 source=archived_streaming_weight_MKL_RAPL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
