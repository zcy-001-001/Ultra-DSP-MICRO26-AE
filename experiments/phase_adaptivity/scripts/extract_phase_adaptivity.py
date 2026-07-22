#!/usr/bin/env python3
"""Extract P-only, D-only, and Hybrid evidence from archived Vivado reports."""

from __future__ import annotations

import csv
import re
from pathlib import Path


EXPERIMENT = Path(__file__).resolve().parent.parent
ROOT = EXPERIMENT.parents[1]
RESULTS = ROOT / "results" / "phase_adaptivity"
REPORTS = RESULTS / "reports"


def first_int_column(text: str, label: str) -> int:
    match = re.search(rf"(?m)^\| {re.escape(label)}\s+\|\s*(\d+)", text)
    if not match:
        raise AssertionError(f"missing {label}")
    return int(match.group(1))


def timing(text: str) -> tuple[float, float, str]:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if "WNS(ns)" not in line or "TNS(ns)" not in line:
            continue
        for candidate in lines[index + 1 : index + 6]:
            match = re.match(r"\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)", candidate)
            if match:
                wns, tns = map(float, match.groups())
                return wns, tns, "MET" if wns >= 0 else "NOT_MET"
    raise AssertionError("missing design timing summary")


def kernel_row(text: str) -> tuple[int, int, int, int, int]:
    line = next(line for line in text.splitlines() if re.match(r"^\| gemv_kernel\s+\|", line))
    fields = [field.strip() for field in line.split("|")[2:8]]
    values = [int(re.match(r"\d+", field).group()) for field in fields]
    lut, lut_as_memory, registers, bram, _uram, dsp = values
    return lut, lut_as_memory, registers, bram, dsp


def power(text: str) -> float:
    match = re.search(r"Total On-Chip Power \(W\)\s+\|\s*([0-9.]+)", text)
    if not match:
        raise AssertionError("missing total on-chip power")
    return float(match.group(1))


def power_confidence(text: str) -> str:
    match = re.search(r"Confidence Level\s+\|\s*([A-Za-z]+)", text)
    if not match:
        raise AssertionError("missing power confidence level")
    return match.group(1)


def main() -> int:
    configs = {
        "P-only 3x3": {
            "dir": REPORTS / "p_only",
            "full": "post_route_utilization.rpt",
            "timing": "post_route_timing_summary.rpt",
            "power": "post_route_power.rpt",
            "power_scope": "P_ONLY_FULL_DESIGN",
            "functional": "ReviewD P-only RTL single-PE smoke check passed",
        },
        "D-only 1x7": {
            "dir": REPORTS / "d_only",
            "full": "post_route_utilization.rpt",
            "timing": "post_route_timing_summary.rpt",
            "power": "post_route_power.rpt",
            "power_scope": "D_ONLY_FULL_DESIGN",
            "functional": "ReviewD D-only RTL single-PE check passed",
        },
        "Hybrid 3x3/1x7": {
            "dir": REPORTS / "hybrid",
            "full": "impl_1_full_util_routed.rpt",
            "timing": "impl_1_hw_bb_locked_timing_summary_routed.rpt",
            # This is the original PD/Hybrid full-design report.  It is
            # provenance for the rounded 45 W analytical input, not a P-only
            # or D-only independent measurement.
            "power": "hw_bb_locked_power_routed.rpt",
            "power_scope": "HYBRID_PD_FULL_DESIGN_PROVENANCE",
            "functional": None,
        },
    }

    rows: list[dict[str, object]] = []
    for name, config in configs.items():
        report_dir = config["dir"]
        full_text = (report_dir / config["full"]).read_text(encoding="utf-8", errors="replace")
        timing_text = (report_dir / config["timing"]).read_text(encoding="utf-8", errors="replace")
        kernel_text = (report_dir / "impl_1_kernel_util_routed.rpt").read_text(
            encoding="utf-8", errors="replace"
        )
        wns, tns, timing_status = timing(timing_text)
        kernel_lut, kernel_lut_mem, kernel_reg, kernel_bram, kernel_dsp = kernel_row(kernel_text)
        if config["functional"]:
            xsim = (report_dir / "xsim.log").read_text(encoding="utf-8", errors="replace")
            assert config["functional"] in xsim
            functional_status = "PASS"
            functional_evidence = f"reports/{report_dir.name}/xsim.log"
        else:
            suite = (ROOT / "results/rtl/rtl_six_case_vivado2023_2.md").read_text(
                encoding="utf-8"
            )
            assert "Cases passed: 6/6" in suite and "Hybrid" in suite
            functional_status = "PASS"
            functional_evidence = "results/rtl/rtl_six_case_vivado2023_2.md"
        power_w = ""
        power_evidence = ""
        confidence = ""
        if config["power"]:
            power_path = report_dir / config["power"]
            power_text = power_path.read_text(encoding="utf-8", errors="replace")
            power_w = power(power_text)
            confidence = power_confidence(power_text)
            power_evidence = f"reports/{report_dir.name}/{config['power']}"
        rows.append(
            {
                "configuration": name,
                "target_mhz": 200,
                "full_clb_lut": first_int_column(full_text, "CLB LUTs"),
                "full_clb_register": first_int_column(full_text, "CLB Registers"),
                "full_dsp": first_int_column(full_text, "DSPs"),
                "kernel_lut": kernel_lut,
                "kernel_lut_as_memory": kernel_lut_mem,
                "kernel_register": kernel_reg,
                "kernel_bram": kernel_bram,
                "kernel_dsp": kernel_dsp,
                "wns_ns": wns,
                "tns_ns": tns,
                "timing_status": timing_status,
                "power_w": power_w,
                "power_evidence": power_evidence,
                "power_scope": config["power_scope"],
                "power_method": "VIVADO_VECTORLESS_ESTIMATE",
                "power_confidence": confidence,
                "functional_status": functional_status,
                "functional_evidence": functional_evidence,
                "evidence_class": "RECOMPUTED_FROM_LOGS",
            }
        )

    output = RESULTS / "phase_adaptivity_summary.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert rows[0]["wns_ns"] == 0.003 and rows[1]["wns_ns"] == 0.003
    assert rows[2]["wns_ns"] == -0.421
    assert rows[2]["power_w"] == 44.78
    assert all(row["power_confidence"] == "Low" for row in rows)
    print(
        "PHASE_ADAPTIVITY_PASS configs=3 functional=3 timing_met=2 "
        "timing_disclosed_not_met=1 hybrid_power=full_design_provenance"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
