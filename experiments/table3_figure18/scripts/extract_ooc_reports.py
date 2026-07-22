#!/usr/bin/env python3
"""Extract sanitized Table 3 and Figure 18 summaries from existing Vivado reports.

The script is read-only with respect to the Vivado build tree.  It can read a
local report tree or use OpenSSH to read one on a remote host.  Only relative
report paths and parsed scalar values are written to the output directory.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS = PACKAGE_ROOT / "results" / "table3_figure18"


TABLE3_CONFIGS = (
    ("W4A4", "WP521", "Baseline/WP521", 4),
    ("W4A4", "DB-MixQ", "Baseline/DeepBurning", 6),
    ("W4A4", "DSP-Packing", "Baseline/DSP-Packing", 6),
    ("W4A4", "DuoQ", "Baseline/DuoQ", 4),
    ("W4A4", "UDP", "Baseline/UDP-general/INT4_INT4", 6),
    ("W4A4", "Ultra-DSP", "Baseline/Ultra-DSP-MAX/W4A4", 9),
    ("W5A5", "UDP", "Baseline/UDP-general/INT5_INT5", 4),
    ("W5A5", "Ultra-DSP", "Baseline/Ultra-DSP-MAX/W5A5", 6),
    ("W4A3", "UDP", "Baseline/UDP-general/INT3_INT4", 6),
    ("W4A3", "Ultra-DSP", "Baseline/Ultra-DSP-MAX/W3A4", 12),
    ("W4A5", "UDP", "Baseline/UDP-general/INT4_INT5", 4),
    ("W4A5", "Ultra-DSP", "Baseline/Ultra-DSP-MAX/W4A5", 8),
    ("W5A3", "UDP", "Baseline/UDP-general/INT3_INT5", 6),
    ("W5A3", "Ultra-DSP", "Baseline/Ultra-DSP-MAX/W3A5", 10),
)

FIGURE18_SIZES = (8, 16, 32, 48, 64, 80, 90)
TABLE3_OOC_VARIANTS = {"DSP-Packing": "ooc_implement1", "DuoQ": "ooc_implement1"}
UTIL_SUFFIX = "hls/impl/verilog/report/gemv_kernel_utilization_routed.rpt"
TIMING_SUFFIX = (
    "hls/impl/verilog/project.runs/impl_1/"
    "bd_0_wrapper_timing_summary_routed.rpt"
)
POWER_SUFFIX = (
    "hls/impl/verilog/project.runs/impl_1/bd_0_wrapper_power_routed.rpt"
)


@dataclass(frozen=True)
class ReportReader:
    source_root: str
    ssh_host: str | None = None

    def read_text(self, relative_path: str) -> str:
        _validate_relative(relative_path)
        if self.ssh_host:
            remote_path = str(PurePosixPath(self.source_root) / relative_path)
            result = subprocess.run(
                ["ssh", self.ssh_host, f"cat -- {shlex.quote(remote_path)}"],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if result.returncode != 0:
                raise FileNotFoundError(relative_path)
            return result.stdout
        return (Path(self.source_root) / Path(relative_path)).read_text(
            encoding="utf-8", errors="replace"
        )

    def list_frequency_dirs(self, relative_ooc_dir: str) -> list[int]:
        _validate_relative(relative_ooc_dir)
        if self.ssh_host:
            remote_path = str(PurePosixPath(self.source_root) / relative_ooc_dir)
            command = (
                f"find {shlex.quote(remote_path)} -mindepth 1 -maxdepth 1 "
                "-type d -name 'freq_*MHz' -printf '%f\\n'"
            )
            result = subprocess.run(
                ["ssh", self.ssh_host, command],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if result.returncode != 0:
                return []
            names = result.stdout.splitlines()
        else:
            names = [p.name for p in (Path(self.source_root) / relative_ooc_dir).glob("freq_*MHz")]
        frequencies = []
        for name in names:
            match = re.fullmatch(r"freq_(\d+)MHz", name.strip())
            if match:
                frequencies.append(int(match.group(1)))
        return sorted(set(frequencies))


def _validate_relative(path: str) -> None:
    posix = PurePosixPath(path)
    if posix.is_absolute() or ".." in posix.parts:
        raise ValueError(f"report path must be relative: {path!r}")


def _first_table_integer(text: str, labels: Iterable[str]) -> int:
    for label in labels:
        match = re.search(
            rf"^\|\s*{re.escape(label)}\s*\|\s*([\d,]+)\s*\|",
            text,
            flags=re.MULTILINE,
        )
        if match:
            return int(match.group(1).replace(",", ""))
    raise ValueError(f"none of the utilization labels were found: {tuple(labels)}")


def parse_utilization(text: str) -> dict[str, int]:
    return {
        "lut": _first_table_integer(text, ("CLB LUTs", "Slice LUTs")),
        "ff": _first_table_integer(text, ("CLB Registers", "Slice Registers")),
        "dsp": _first_table_integer(text, ("DSPs", "DSP48E2")),
    }


def parse_power(text: str) -> float:
    match = re.search(
        r"^\|\s*Total On-Chip Power \(W\)\s*\|\s*([0-9.]+)\s*\|",
        text,
        flags=re.MULTILINE,
    )
    if not match:
        raise ValueError("Total On-Chip Power was not found")
    return float(match.group(1))


def parse_wns(text: str) -> float:
    header = text.find("WNS(ns)")
    if header < 0:
        raise ValueError("WNS header was not found")
    for line in text[header:].splitlines()[1:12]:
        match = re.match(r"\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s+", line)
        if match:
            return float(match.group(1))
    raise ValueError("Design Timing Summary WNS value was not found")


def _report_paths(
    base: str, frequency_mhz: int, ooc_variant: str = "ooc_implement"
) -> tuple[str, str, str]:
    prefix = f"{base}/{ooc_variant}/freq_{frequency_mhz}MHz"
    return (
        f"{prefix}/{UTIL_SUFFIX}",
        f"{prefix}/{POWER_SUFFIX}",
        f"{prefix}/{TIMING_SUFFIX}",
    )


def _figure_report_paths(base: str, ooc_variant: str, frequency_mhz: int) -> tuple[str, str]:
    prefix = f"{base}/{ooc_variant}/freq_{frequency_mhz}MHz"
    return f"{prefix}/{UTIL_SUFFIX}", f"{prefix}/{TIMING_SUFFIX}"


def extract_table3(reader: ReportReader) -> list[dict[str, object]]:
    rows = []
    frequency_mhz = 200
    dsp_count = 4096
    for data_format, method, base, packing_count in TABLE3_CONFIGS:
        ooc_variant = TABLE3_OOC_VARIANTS.get(method, "ooc_implement")
        util_path, power_path, timing_path = _report_paths(base, frequency_mhz, ooc_variant)
        util = parse_utilization(reader.read_text(util_path))
        power_w = parse_power(reader.read_text(power_path))
        wns_ns = parse_wns(reader.read_text(timing_path))
        throughput_gops = dsp_count * packing_count * 2 * frequency_mhz / 1000.0
        rows.append(
            {
                "data_format": data_format,
                "method": method,
                "dsp_count": dsp_count,
                "routed_report_dsp": util["dsp"],
                "lut": util["lut"],
                "ff": util["ff"],
                "klut": round(util["lut"] / 1000.0, 3),
                "kff": round(util["ff"] / 1000.0, 3),
                "power_w": power_w,
                "packing_count": packing_count,
                "frequency_mhz": frequency_mhz,
                "throughput_gops": throughput_gops,
                "energy_efficiency_gops_per_w": round(throughput_gops / power_w, 2),
                "wns_ns": wns_ns,
                "timing_met": wns_ns >= 0,
                "utilization_report": util_path,
                "power_report": power_path,
                "timing_report": timing_path,
            }
        )
    return rows


def extract_figure18(reader: ReportReader) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    sweep_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for side in FIGURE18_SIZES:
        base = f"W4A4_PD_{side}_{side}"
        candidates = []
        for ooc_variant in ("ooc_implement", "ooc_implement1"):
            for frequency_mhz in reader.list_frequency_dirs(f"{base}/{ooc_variant}"):
                util_path, timing_path = _figure_report_paths(base, ooc_variant, frequency_mhz)
                try:
                    wns_ns = parse_wns(reader.read_text(timing_path))
                    util = parse_utilization(reader.read_text(util_path))
                except (FileNotFoundError, ValueError):
                    continue
                period_ns = 1000.0 / frequency_mhz
                critical_path_ns = period_ns - wns_ns
                estimated_fmax_mhz = 1000.0 / critical_path_ns if critical_path_ns > 0 else math.nan
                row = {
                    "pe_rows": side,
                    "pe_cols": side,
                    "pe_count": side * side,
                    "ooc_variant": ooc_variant,
                    "target_frequency_mhz": frequency_mhz,
                    "wns_ns": wns_ns,
                    "timing_met": wns_ns >= 0,
                    "estimated_fmax_mhz": round(estimated_fmax_mhz, 3),
                    "lut": util["lut"],
                    "ff": util["ff"],
                    "expected_pe_dsp": side * side,
                    "routed_report_dsp": util["dsp"],
                    "utilization_report": util_path,
                    "timing_report": timing_path,
                }
                sweep_rows.append(row)
                if wns_ns >= 0:
                    candidates.append(row)
        if not candidates:
            raise ValueError(f"no timing-clean routed point was found for {base}")
        selected = max(
            candidates,
            key=lambda row: (int(row["target_frequency_mhz"]), float(row["wns_ns"])),
        )
        summary_rows.append(
            {
                **selected,
                "selected_for_paper_figure": side >= 16,
                "selection_rule": "highest routed target frequency with WNS >= 0",
            }
        )
    return summary_rows, sweep_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_table3(table3: list[dict[str, object]]) -> None:
    """Validate Table 3 independently for the packaged 42-report evidence."""
    if len(table3) != len(TABLE3_CONFIGS):
        raise AssertionError("Table 3 row count mismatch")
    for row in table3:
        expected = int(row["dsp_count"]) * int(row["packing_count"]) * 2 * int(row["frequency_mhz"]) / 1000.0
        if not math.isclose(float(row["throughput_gops"]), expected, rel_tol=0, abs_tol=1e-9):
            raise AssertionError(f"throughput formula mismatch: {row}")
        if int(row["dsp_count"]) != 4096:
            raise AssertionError(f"Table 3 must use 4096 DSPs: {row}")
        if int(row["routed_report_dsp"]) < 4096:
            raise AssertionError(f"routed report has fewer than 4096 array DSPs: {row}")
        if not bool(row["timing_met"]):
            raise AssertionError(f"Table 3 routed report misses 200 MHz timing: {row}")
    paper_w4a4 = next(
        row for row in table3 if row["data_format"] == "W4A4" and row["method"] == "Ultra-DSP"
    )
    anchors = {"lut": 244939, "ff": 259987, "power_w": 6.218, "throughput_gops": 14745.6}
    for key, expected in anchors.items():
        if not math.isclose(float(paper_w4a4[key]), expected, rel_tol=0, abs_tol=1e-6):
            raise AssertionError(f"W4A4 Ultra-DSP anchor mismatch for {key}")


def validate(table3: list[dict[str, object]], figure18: list[dict[str, object]]) -> None:
    # Keep the original combined validation behavior while allowing the new
    # public Table 3 evidence bundle to be checked without Figure 18 reports.
    validate_table3(table3)
    if len(figure18) != len(FIGURE18_SIZES):
        raise AssertionError("Figure 18 row count mismatch")
    if [int(row["pe_count"]) for row in figure18 if row["selected_for_paper_figure"]] != [256, 1024, 2304, 4096, 6400, 8100]:
        raise AssertionError("Figure 18 paper PE sequence mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, help="Vivado report tree root")
    parser.add_argument("--ssh-host", help="Read reports through OpenSSH")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--check", action="store_true", help="Validate without writing CSV files")
    parser.add_argument(
        "--table3-only",
        action="store_true",
        help="Read and validate only the packaged Table 3 routed reports",
    )
    args = parser.parse_args()

    reader = ReportReader(args.source_root, args.ssh_host)
    table3 = extract_table3(reader)
    if args.table3_only:
        validate_table3(table3)
        if not args.check:
            write_csv(args.output_dir / "table3_ooc_summary.csv", table3)
        print(f"OOC_TABLE3_EXTRACT_PASS table3_rows={len(table3)} reports={len(table3) * 3}")
        return 0
    figure18, figure18_sweep = extract_figure18(reader)
    validate(table3, figure18)
    if not args.check:
        write_csv(args.output_dir / "table3_ooc_summary.csv", table3)
        write_csv(args.output_dir / "figure18_post_implementation_summary.csv", figure18)
        write_csv(args.output_dir / "figure18_frequency_sweep.csv", figure18_sweep)
    print(
        f"OOC_EXTRACT_PASS table3_rows={len(table3)} "
        f"figure18_rows={len(figure18)} sweep_rows={len(figure18_sweep)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
