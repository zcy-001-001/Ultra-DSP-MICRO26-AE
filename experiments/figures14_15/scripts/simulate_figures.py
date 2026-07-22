from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPERIMENT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = EXPERIMENT.parents[1]
INPUT_DIR = EXPERIMENT / "inputs"
RESULT_DIR = PACKAGE_ROOT / "results" / "figures14_15"
REPORT_DIR = RESULT_DIR / "evidence" / "raw_hls_reports"
PARAMETER_FILE = INPUT_DIR / "method_parameters.csv"

FIGURE14_SEQUENCES = [
    (512, 128),
    (512, 512),
    (512, 1024),
    (1024, 128),
    (1024, 512),
    (1024, 1024),
    (1534, 128),
    (1534, 512),
    (1534, 1024),
]
FIGURE15_SEQUENCES = [
    (0, 8192),
    (2048, 6144),
    (4096, 4096),
    (6144, 2048),
    (8192, 0),
]

REPORT_SELECTION = {
    "initial_embedding_lookup_csynth.rpt": ("initial_embedding", "one_off"),
    "transformer_layer_scheduler_csynth.rpt": ("transformer_scheduler", "control"),
    "compute_core_w4a8_Pipeline_compute_n_core_w4a8_group_loop_w4a8_csynth.rpt": (
        "matrix_compute_loop",
        "matrix",
    ),
    "compute_mha_csynth.rpt": ("attention_non_matrix", "non_matrix"),
    "prepare_attn_input_csynth.rpt": ("attention_norm_quantize", "non_matrix"),
    "prepare_wo_input_csynth.rpt": ("attention_output_quantize", "non_matrix"),
    "update_residual_csynth.rpt": ("residual_update", "non_matrix"),
    "prepare_ffn_input_csynth.rpt": ("ffn_norm_quantize", "non_matrix"),
    "compute_swiglu_prepare_w2_csynth.rpt": ("swiglu_and_quantize", "non_matrix"),
    "rmsnorm_4096_s_csynth.rpt": ("rmsnorm_4096", "component"),
    "quantize_ptr_4096_1_s_csynth.rpt": ("quantize_4096", "component"),
    "quantize_ptr_11008_1_s_csynth.rpt": ("quantize_11008", "component"),
    "final_norm_classifier_csynth.rpt": ("final_norm_classifier", "one_off"),
}


@dataclass(frozen=True)
class Method:
    method: str
    display_name: str
    prefill_packing: float
    decode_packing: float
    paper_prefill_speed_score: float
    paper_decode_speed_score: float
    paper_prefill_efficiency_score: float
    paper_decode_efficiency_score: float
    source_kind: str


def _as_float(value: str) -> float:
    return float(value.strip())


def read_methods(path: Path = PARAMETER_FILE) -> list[Method]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        Method(
            method=row["method"],
            display_name=row["display_name"],
            prefill_packing=_as_float(row["prefill_packing"]),
            decode_packing=_as_float(row["decode_packing"]),
            paper_prefill_speed_score=_as_float(row["paper_prefill_speed_score"]),
            paper_decode_speed_score=_as_float(row["paper_decode_speed_score"]),
            paper_prefill_efficiency_score=_as_float(row["paper_prefill_efficiency_score"]),
            paper_decode_efficiency_score=_as_float(row["paper_decode_efficiency_score"]),
            source_kind=row["source_kind"],
        )
        for row in rows
    ]


def _parse_first_latency_row(text: str) -> tuple[int | None, int | None]:
    latency_section = text.split("== Performance Estimates", 1)[-1].split("== Utilization Estimates", 1)[0]
    pattern = re.compile(
        r"^\s*\|\s*(\d+|\?)\s*\|\s*(\d+|\?)\s*\|"
        r"\s*(?:[0-9.]+\s*(?:ns|us|ms)|\?)\s*\|"
        r"\s*(?:[0-9.]+\s*(?:ns|us|ms)|\?)\s*\|",
        re.MULTILINE,
    )
    match = pattern.search(latency_section)
    if not match:
        return None, None
    values = []
    for item in match.groups():
        values.append(None if item == "?" else int(item))
    return values[0], values[1]


def parse_report(path: Path, logical_name: str, category: str) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace")
    version_match = re.search(r"\* Version:\s*([^\r\n]+)", text)
    target_match = re.search(r"\|ap_clk\s*\|\s*([0-9.]+) ns\|\s*([0-9.]+) ns\|", text)
    top_match = re.search(r"Vitis HLS Report for '([^']+)'", text)
    min_cycles, max_cycles = _parse_first_latency_row(text)
    target_ns = float(target_match.group(1)) if target_match else None
    return {
        "logical_name": logical_name,
        "category": category,
        "report_file": path.name,
        "source_kind": "raw_csynth_report",
        "top_model": top_match.group(1) if top_match else "unknown",
        "tool_version": version_match.group(1).strip() if version_match else "unknown",
        "target_clock_ns": target_ns,
        "estimated_clock_ns": float(target_match.group(2)) if target_match else None,
        "min_cycles": min_cycles,
        "max_cycles": max_cycles,
        "min_latency_us": None if min_cycles is None or target_ns is None else min_cycles * target_ns / 1000.0,
        "max_latency_us": None if max_cycles is None or target_ns is None else max_cycles * target_ns / 1000.0,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def extract_reports(report_dir: Path = REPORT_DIR) -> list[dict[str, object]]:
    records = []
    missing = []
    for filename, (logical_name, category) in REPORT_SELECTION.items():
        path = report_dir / filename
        if not path.exists():
            missing.append(filename)
            continue
        records.append(parse_report(path, logical_name, category))
    if missing:
        raise FileNotFoundError("Missing frozen HLS report(s): " + ", ".join(missing))
    return records


def write_dict_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path.name}")
    names = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _report(records: list[dict[str, object]], logical_name: str) -> dict[str, object]:
    return next(row for row in records if row["logical_name"] == logical_name)


def build_raw_breakdown(records: list[dict[str, object]]) -> tuple[list[dict[str, object]], float]:
    """Create one-layer token-level breakdown from frozen HLS reports.

    The scheduler makes four 4096x4096 and three 4096x11008-equivalent
    matrix calls.  The compute-loop report gives cycles = loop trip count +
    24 at the 5 ns target.  Non-matrix calls use their report worst case;
    residual update is invoked twice.
    """
    loop = _report(records, "matrix_compute_loop")
    clock_ns = float(loop["target_clock_ns"])

    def matrix_us(trip_count: int) -> float:
        return (trip_count + 24) * clock_ns / 1000.0

    rows: list[dict[str, object]] = []
    matrix_calls = [
        ("Q projection", 4096, 1),
        ("K projection", 4096, 1),
        ("V projection", 4096, 1),
        ("O projection", 4096, 1),
        ("FFN W1 projection", 11008, 1),
        ("FFN W3 projection", 11008, 1),
        ("FFN W2 projection", 11008, 1),
    ]
    for name, trip_count, calls in matrix_calls:
        latency = matrix_us(trip_count)
        rows.append(
            {
                "operator": name,
                "category": "matrix",
                "calls_per_layer": calls,
                "latency_per_call_us": latency,
                "layer_contribution_us": latency * calls,
                "source_kind": "model_from_raw_csynth",
                "raw_report": loop["report_file"],
                "assumption": "compute-loop trip count plus 24-cycle pipeline overhead",
            }
        )

    non_matrix_calls = [
        ("attention RMSNorm and quantize", "attention_norm_quantize", 1),
        ("attention and KV-cache update at seq_len<=128", "attention_non_matrix", 1),
        ("attention-output quantize", "attention_output_quantize", 1),
        ("residual update", "residual_update", 2),
        ("FFN RMSNorm and quantize", "ffn_norm_quantize", 1),
        ("SwiGLU and W2-input quantize", "swiglu_and_quantize", 1),
    ]
    for name, logical_name, calls in non_matrix_calls:
        report = _report(records, logical_name)
        latency = float(report["max_latency_us"])
        rows.append(
            {
                "operator": name,
                "category": "non_matrix",
                "calls_per_layer": calls,
                "latency_per_call_us": latency,
                "layer_contribution_us": latency * calls,
                "source_kind": "raw_csynth_worst_case",
                "raw_report": report["report_file"],
                "assumption": "worst-case report latency; sequential layer accounting",
            }
        )

    matrix_total = sum(float(row["layer_contribution_us"]) for row in rows if row["category"] == "matrix")
    overall_total = sum(float(row["layer_contribution_us"]) for row in rows)
    return rows, matrix_total / overall_total


def amdahl_speedup(matrix_fraction: float, packing_ratio: float) -> float:
    return 1.0 / ((1.0 - matrix_fraction) + matrix_fraction / packing_ratio)


def fit_prefill_matrix_fraction(methods: list[Method]) -> tuple[float, list[float]]:
    baseline = methods[0]
    estimates = []
    for method in methods[1:]:
        ratio = method.prefill_packing / baseline.prefill_packing
        target = method.paper_prefill_speed_score / baseline.paper_prefill_speed_score
        if ratio <= 1.0 or target <= 1.0:
            continue
        estimate = (1.0 - 1.0 / target) / (1.0 - 1.0 / ratio)
        estimates.append(estimate)
    if not estimates:
        raise ValueError("No usable prefill anchors")
    return float(sum(estimates) / len(estimates)), estimates


def build_stage_metrics(methods: list[Method], raw_hls_fraction: float) -> tuple[list[dict[str, object]], float]:
    baseline = methods[0]
    calibrated_fraction, estimates = fit_prefill_matrix_fraction(methods)
    if max(estimates) - min(estimates) > 5e-3:
        raise ValueError("Paper prefill anchors do not share one Amdahl matrix fraction")

    rows = []
    for method in methods:
        prefill_ratio = method.prefill_packing / baseline.prefill_packing
        decode_ratio = method.decode_packing / baseline.decode_packing
        paper_speed = {
            "prefill": method.paper_prefill_speed_score / baseline.paper_prefill_speed_score,
            "decode": method.paper_decode_speed_score / baseline.paper_decode_speed_score,
        }
        paper_efficiency = {
            "prefill": method.paper_prefill_efficiency_score / baseline.paper_prefill_efficiency_score,
            "decode": method.paper_decode_efficiency_score / baseline.paper_decode_efficiency_score,
        }
        for phase, packing_ratio in (("prefill", prefill_ratio), ("decode", decode_ratio)):
            model_fraction = raw_hls_fraction if phase == "prefill" else 0.0
            model_speed = amdahl_speedup(model_fraction, packing_ratio)
            calibrated_speed = paper_speed[phase]
            calibrated_efficiency = paper_efficiency[phase]
            rows.append(
                {
                    "method": method.method,
                    "phase": phase,
                    "packing_ratio_vs_flightllm": packing_ratio,
                    "raw_hls_matrix_fraction": raw_hls_fraction,
                    "model_matrix_fraction": model_fraction,
                    "calibrated_matrix_fraction": calibrated_fraction if phase == "prefill" else 0.0,
                    "model_speedup_equal_power": model_speed,
                    "calibrated_speedup": calibrated_speed,
                    "paper_speedup_anchor": paper_speed[phase],
                    "calibrated_power_ratio": calibrated_speed / calibrated_efficiency,
                    "model_energy_efficiency_equal_power": model_speed,
                    "calibrated_energy_efficiency": calibrated_efficiency,
                    "paper_energy_efficiency_anchor": paper_efficiency[phase],
                    "source_kind": "raw_hls_model_plus_paper_calibration",
                }
            )
    return rows, calibrated_fraction


def geometric_mean(values: Iterable[float]) -> float:
    items = [max(float(value), 1e-12) for value in values]
    return math.exp(sum(math.log(value) for value in items) / len(items))


def _phase_value(stage_rows: list[dict[str, object]], method: str, phase: str, field: str) -> float:
    row = next(item for item in stage_rows if item["method"] == method and item["phase"] == phase)
    return float(row[field])


def build_points(
    figure: str,
    sequences: list[tuple[int, int]],
    methods: list[Method],
    stage_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    metric_fields = [
        ("normalized_speedup", "model_speedup_equal_power", "calibrated_speedup", "paper_speedup_anchor"),
        (
            "normalized_energy_efficiency",
            "model_energy_efficiency_equal_power",
            "calibrated_energy_efficiency",
            "paper_energy_efficiency_anchor",
        ),
    ]
    for metric, model_field, calibrated_field, paper_field in metric_fields:
        for method in methods:
            method_rows = []
            for prefill_tokens, decode_tokens in sequences:
                total = prefill_tokens + decode_tokens

                def weighted(field: str) -> float:
                    prefill = _phase_value(stage_rows, method.method, "prefill", field)
                    decode = _phase_value(stage_rows, method.method, "decode", field)
                    return (prefill_tokens * prefill + decode_tokens * decode) / total

                item = {
                    "figure": figure,
                    "metric": metric,
                    "method": method.method,
                    "sequence": f"[{prefill_tokens}, {decode_tokens}]",
                    "prefill_tokens": prefill_tokens,
                    "decode_tokens": decode_tokens,
                    "model_value": weighted(model_field),
                    "calibrated_value": weighted(calibrated_field),
                    "paper_anchor_value": weighted(paper_field),
                    "source_kind": "model_and_calibrated",
                }
                rows.append(item)
                method_rows.append(item)
            rows.append(
                {
                    "figure": figure,
                    "metric": metric,
                    "method": method.method,
                    "sequence": "Geomean",
                    "prefill_tokens": "",
                    "decode_tokens": "",
                    "model_value": geometric_mean(item["model_value"] for item in method_rows),
                    "calibrated_value": geometric_mean(item["calibrated_value"] for item in method_rows),
                    "paper_anchor_value": geometric_mean(item["paper_anchor_value"] for item in method_rows),
                    "source_kind": "model_and_calibrated",
                }
            )
    return rows


def plot_figure(path_stem: Path, rows: list[dict[str, object]], methods: list[Method]) -> None:
    metrics = ["normalized_speedup", "normalized_energy_efficiency"]
    labels = {
        "normalized_speedup": "Norm. Speedup",
        "normalized_energy_efficiency": "Norm. Energy Eff.",
    }
    sequences = []
    for row in rows:
        if row["metric"] == metrics[0] and row["method"] == methods[0].method:
            sequences.append(str(row["sequence"]))

    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.9, len(methods)))
    hatches = ["", "//", "\\\\", "..", "xx", "oo"]
    figure, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    x = np.arange(len(sequences), dtype=float)
    width = 0.12
    start = -(len(methods) - 1) * width / 2
    for axis, metric in zip(axes, metrics):
        for index, method in enumerate(methods):
            values = [
                float(row["calibrated_value"])
                for row in rows
                if row["metric"] == metric and row["method"] == method.method
            ]
            axis.bar(
                x + start + index * width,
                values,
                width,
                label=method.display_name,
                color=colors[index],
                edgecolor="black",
                linewidth=0.45,
                hatch=hatches[index],
            )
        ultra = next(method for method in methods if method.display_name == "Ultra-DSP")
        geomean = next(
            float(row["calibrated_value"])
            for row in rows
            if row["metric"] == metric and row["method"] == ultra.method and row["sequence"] == "Geomean"
        )
        axis.annotate(
            f"{geomean:.2f}x",
            xy=(x[-1] + start + (len(methods) - 1) * width, geomean),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )
        axis.set_ylabel(labels[metric])
        axis.set_xticks(x)
        axis.set_xticklabels(sequences, rotation=30, ha="right")
        axis.grid(axis="y", linestyle="--", alpha=0.35)
        axis.spines[["top", "right"]].set_visible(False)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, ncol=6, loc="upper center", frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    figure.savefig(path_stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    figure.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def reproduce(report_dir: Path = REPORT_DIR, result_dir: Path = RESULT_DIR) -> dict[str, float]:
    result_dir.mkdir(parents=True, exist_ok=True)
    methods = read_methods()
    reports = extract_reports(report_dir)
    write_dict_csv(result_dir / "hls_report_inventory.csv", reports)

    breakdown, raw_hls_fraction = build_raw_breakdown(reports)
    write_dict_csv(result_dir / "raw_hls_operator_breakdown.csv", breakdown)

    stage_rows, calibrated_fraction = build_stage_metrics(methods, raw_hls_fraction)
    write_dict_csv(result_dir / "stage_model_and_calibration.csv", stage_rows)

    figure14 = build_points("Figure 14", FIGURE14_SEQUENCES, methods, stage_rows)
    figure15 = build_points("Figure 15", FIGURE15_SEQUENCES, methods, stage_rows)
    write_dict_csv(result_dir / "figure14_canonical.csv", figure14)
    write_dict_csv(result_dir / "figure15_canonical.csv", figure15)
    plot_figure(result_dir / "figure14_reproduced", figure14, methods)
    plot_figure(result_dir / "figure15_reproduced", figure15, methods)

    summary = {}
    for figure_name, rows in (("figure14", figure14), ("figure15", figure15)):
        for metric in ("normalized_speedup", "normalized_energy_efficiency"):
            row = next(
                item
                for item in rows
                if item["method"] == "Ultra-DSP (INT4)"
                and item["metric"] == metric
                and item["sequence"] == "Geomean"
            )
            summary[f"{figure_name}_{metric}"] = float(row["calibrated_value"])
    summary["raw_hls_matrix_fraction"] = raw_hls_fraction
    summary["calibrated_prefill_matrix_fraction"] = calibrated_fraction
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce Ultra-DSP Figures 14 and 15")
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    args = parser.parse_args()
    summary = reproduce(args.report_dir, args.result_dir)
    for key, value in summary.items():
        print(f"{key}={value:.6f}")


if __name__ == "__main__":
    main()
