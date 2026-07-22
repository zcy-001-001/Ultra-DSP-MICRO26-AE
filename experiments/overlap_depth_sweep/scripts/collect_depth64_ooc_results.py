#!/usr/bin/env python3
"""Collect Vitis/Vivado OOC implementation results for the 64x64 depth sweep.

Run this on <REMOTE_HOST>, or on any machine that can see the generated project
directories.  The authoritative OOC timing comes from Vivado's routed timing
summary under `hls/impl/verilog/project.runs/impl_1/`.

Vitis also emits `hls/impl/report/verilog/gemv_kernel_export.rpt`, but in this
flow that file only reports post-synthesis timing.  The parser keeps those
numbers in separate `post_synth_*` columns so they cannot be mistaken for
place-and-route timing closure.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


PROJECTS = (
    "W4A4_Depth4_64_64",
    "W4A4_Depth5_64_64",
    "W4A4_Depth6_64_64",
)


def parse_routed_timing(path: Path) -> dict[str, str]:
    row: dict[str, str] = {
        "impl_status": "",
        "impl_worst_path_slack_ns": "",
        "impl_tns_ns": "",
        "impl_setup_failing_endpoints": "",
        "impl_target_period_ns": "",
    }
    if not path.exists():
        return row

    lines = path.read_text(errors="ignore").splitlines()
    for line in lines:
        if "Timing constraints are not met" in line:
            row["impl_status"] = "not_met"
        elif "Timing constraints are met" in line:
            row["impl_status"] = "met"

        clock_match = re.search(r"^ap_clk\s+\{[^}]+\}\s+([0-9.]+)\s+([0-9.]+)", line)
        if clock_match:
            row["impl_target_period_ns"] = clock_match.group(1)

        setup_match = re.search(
            r"Setup\s*:\s*(\d+)\s+Failing Endpoints,\s+Worst Slack\s+(-?[0-9.]+)ns,\s+Total Violation\s+(-?[0-9.]+)ns",
            line,
        )
        if setup_match:
            row["impl_setup_failing_endpoints"] = setup_match.group(1)
            row["impl_worst_path_slack_ns"] = setup_match.group(2)
            row["impl_tns_ns"] = setup_match.group(3)

    # Some Vivado reports put the compact design timing summary before the
    # detailed "Setup :" line.  Use it as a fallback.
    if not row["impl_worst_path_slack_ns"]:
        for idx, line in enumerate(lines):
            if "WNS(ns)" in line and "TNS(ns)" in line:
                for subline in lines[idx + 1 : idx + 5]:
                    nums = re.findall(r"-?\d+\.\d+|-?\d+", subline)
                    if len(nums) >= 3:
                        row["impl_worst_path_slack_ns"] = nums[0]
                        row["impl_tns_ns"] = nums[1]
                        row["impl_setup_failing_endpoints"] = nums[2]
                        break
                if row["impl_worst_path_slack_ns"]:
                    break
    return row


def parse_placed_utilization(path: Path) -> dict[str, str]:
    row = {
        "impl_lut": "",
        "impl_ff": "",
        "impl_dsp": "",
        "impl_bram": "",
        "impl_uram": "",
    }
    if not path.exists():
        return row

    for line in path.read_text(errors="ignore").splitlines():
        nums = re.findall(r"\d+\.?\d*", line)
        if "| CLB LUTs" in line and nums and not row["impl_lut"]:
            row["impl_lut"] = str(int(float(nums[0])))
        elif "| CLB Registers" in line and nums and not row["impl_ff"]:
            row["impl_ff"] = str(int(float(nums[0])))
        elif "| DSPs" in line and nums and not row["impl_dsp"]:
            row["impl_dsp"] = str(int(float(nums[0])))
        elif "| Block RAM Tile" in line and nums and not row["impl_bram"]:
            row["impl_bram"] = nums[0]
        elif "| URAM" in line and nums and not row["impl_uram"]:
            row["impl_uram"] = nums[0]
    return row


def parse_impl_runme_failure(path: Path) -> dict[str, str]:
    row = {
        "impl_failure_reason": "",
        "impl_required_lut_as_logic": "",
        "impl_available_lut_as_logic": "",
        "impl_required_slice_luts": "",
        "impl_available_slice_luts": "",
    }
    if not path.exists():
        return row

    text = path.read_text(errors="ignore")
    for line in text.splitlines():
        match = re.search(
            r"LUT as Logic over-utilized.*requires\s+(\d+).*only\s+(\d+)",
            line,
        )
        if match:
            row["impl_failure_reason"] = "lut_as_logic_overutilized"
            row["impl_required_lut_as_logic"] = match.group(1)
            row["impl_available_lut_as_logic"] = match.group(2)
        match = re.search(
            r"Slice LUTs over-utilized.*requires\s+(\d+).*only\s+(\d+)",
            line,
        )
        if match:
            row["impl_failure_reason"] = "slice_luts_overutilized"
            row["impl_required_slice_luts"] = match.group(1)
            row["impl_available_slice_luts"] = match.group(2)

    if "place_design failed" in text and row["impl_failure_reason"]:
        row["impl_failure_reason"] = f"place_drc_{row['impl_failure_reason']}"
    elif "place_design failed" in text:
        row["impl_failure_reason"] = "place_design_failed"

    return row


def parse_vitis_export(path: Path) -> dict[str, str]:
    row = {
        "post_synth_status": "",
        "post_synth_cp_required_ns": "",
        "post_synth_cp_achieved_ns": "",
        "post_synth_lut": "",
        "post_synth_ff": "",
        "post_synth_dsp": "",
        "post_synth_bram": "",
    }
    if not path.exists():
        return row

    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith("LUT:"):
            row["post_synth_lut"] = line.split(":", 1)[1].strip()
        elif line.startswith("FF:"):
            row["post_synth_ff"] = line.split(":", 1)[1].strip()
        elif line.startswith("DSP:"):
            row["post_synth_dsp"] = line.split(":", 1)[1].strip()
        elif line.startswith("BRAM:"):
            row["post_synth_bram"] = line.split(":", 1)[1].strip()
        elif line.startswith("CP required:"):
            row["post_synth_cp_required_ns"] = line.split(":", 1)[1].strip()
        elif line.startswith("CP achieved post-synthesis:"):
            row["post_synth_cp_achieved_ns"] = line.split(":", 1)[1].strip()
        elif line.strip() == "Timing met":
            row["post_synth_status"] = "met"
        elif line.strip() == "Timing not met":
            row["post_synth_status"] = "not_met"
    return row


def parse_synth_reports(report_dir: Path) -> dict[str, str]:
    row = {
        "synth_wns_ns": "",
        "synth_lut": "",
        "synth_ff": "",
        "synth_dsp": "",
    }
    timing = report_dir / "gemv_kernel_timing_synth.rpt"
    if timing.exists():
        lines = timing.read_text(errors="ignore").splitlines()
        for idx, line in enumerate(lines):
            if "WNS(ns)" in line:
                for subline in lines[idx + 1 : idx + 6]:
                    nums = re.findall(r"-?\d+\.\d+|-?\d+", subline)
                    if nums:
                        row["synth_wns_ns"] = nums[0]
                        break
                break
    util = report_dir / "gemv_kernel_utilization_synth.rpt"
    if util.exists():
        for line in util.read_text(errors="ignore").splitlines():
            nums = re.findall(r"\d+\.?\d*", line)
            if "| CLB LUTs" in line and nums:
                row["synth_lut"] = str(int(float(nums[0])))
            elif "| CLB Registers" in line and nums:
                row["synth_ff"] = str(int(float(nums[0])))
            elif "| DSPs" in line and nums:
                row["synth_dsp"] = str(int(float(nums[0])))
    return row


def collect(base: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for project in PROJECTS:
        root = base / project / "ooc_implement"
        for freq_dir in sorted(root.glob("freq_*MHz")):
            match = re.search(r"freq_(\d+)MHz", freq_dir.name)
            if not match:
                continue
            report_dir = freq_dir / "hls" / "impl" / "verilog" / "report"
            row = {
                "project": project,
                "depth": re.search(r"Depth(\d+)", project).group(1),
                "freq_mhz": match.group(1),
            }
            row.update(parse_synth_reports(report_dir))
            impl_dir = freq_dir / "hls" / "impl" / "verilog" / "project.runs" / "impl_1"
            row.update(parse_routed_timing(impl_dir / "bd_0_wrapper_timing_summary_routed.rpt"))
            row.update(parse_placed_utilization(impl_dir / "bd_0_wrapper_utilization_placed.rpt"))
            # Routed reports do not exist when placement DRC fails.  Keep this
            # failure parser separate so successful routed rows are still driven
            # by Vivado timing/utilization reports.
            failure = parse_impl_runme_failure(impl_dir / "runme.log")
            row.update(failure)
            if not row.get("impl_status") and failure.get("impl_failure_reason"):
                row["impl_status"] = failure["impl_failure_reason"]
            row.update(parse_vitis_export(freq_dir / "hls" / "impl" / "report" / "verilog" / "gemv_kernel_export.rpt"))
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("<REMOTE_WORKSPACE>/A-MICRO26-DSP-Packing/Baseline/Ultra-DSP-MAX"),
    )
    parser.add_argument("--out", type=Path, default=Path("depth64_ooc_summary.csv"))
    args = parser.parse_args()

    rows = collect(args.base)
    fieldnames = [
        "project",
        "depth",
        "freq_mhz",
        "impl_status",
        "impl_worst_path_slack_ns",
        "impl_tns_ns",
        "impl_setup_failing_endpoints",
        "impl_target_period_ns",
        "impl_lut",
        "impl_ff",
        "impl_dsp",
        "impl_bram",
        "impl_uram",
        "impl_failure_reason",
        "impl_required_lut_as_logic",
        "impl_available_lut_as_logic",
        "impl_required_slice_luts",
        "impl_available_slice_luts",
        "synth_wns_ns",
        "synth_lut",
        "synth_ff",
        "synth_dsp",
        "post_synth_status",
        "post_synth_cp_required_ns",
        "post_synth_cp_achieved_ns",
        "post_synth_lut",
        "post_synth_ff",
        "post_synth_dsp",
        "post_synth_bram",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
