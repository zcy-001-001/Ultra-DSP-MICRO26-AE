from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from plot_long_context import (
    ANNOTATION_COLOR,
    BASELINE_METHOD,
    BAR_WIDTH,
    COLORS,
    EDGE_COLOR,
    ENERGY_STAGE_DATA,
    GRID_COLOR,
    HATCHES,
    METHOD_DISPLAY_NAMES,
    PLOT_METHODS,
    RESULTS_DIR,
    SPEEDUP_STAGE_DATA,
    Y_MIN,
    Y_TICKS,
    configure_paper_font,
    geometric_mean,
    get_sequence_weighted_energy,
    get_sequence_weighted_ratio,
)


K = 1024

# Reviewer-facing compact 8K sweep. This intentionally keeps only every other
# P/D split while recomputing Geomean over this reduced 2K-spaced set.
SEQUENCE_PAIRS_2K = [
    (0 * K, 8 * K),
    (2 * K, 6 * K),
    (4 * K, 4 * K),
    (6 * K, 2 * K),
    (8 * K, 0 * K),
]
X_LABELS_2K = [f"[{p // K}K, {d // K}K]" for p, d in SEQUENCE_PAIRS_2K] + ["Geomean"]

GROUP_SPACING_2K = 0.82
BAR_WIDTH_2K = BAR_WIDTH
FIGSIZE = (18, 17.2)

PANEL_LAYOUTS = [
    {
        "metric": "normalized_speedup",
        "ylabel": "Norm. Speedup",
        "stage_data": SPEEDUP_STAGE_DATA,
        "y_max": 3.0,
    },
    {
        "metric": "normalized_energy_efficiency",
        "ylabel": "Norm. Energy Eff.",
        "stage_data": ENERGY_STAGE_DATA,
        "y_max": 3.0,
    },
]


def build_metric_payload(stage_data: dict[str, dict[str, float]]) -> dict[str, list[float]]:
    normalized_data: dict[str, list[float]] = {}
    for method in PLOT_METHODS:
        normalized_values = [
            get_sequence_weighted_ratio(stage_data[method], stage_data[BASELINE_METHOD], prefill_len, decode_len)
            for prefill_len, decode_len in SEQUENCE_PAIRS_2K
        ]
        normalized_data[method] = normalized_values + [geometric_mean(normalized_values)]
    return normalized_data


def write_wide_csv(path: Path, data: dict[str, list[float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["method", *X_LABELS_2K])
        for method in PLOT_METHODS:
            writer.writerow([method, *[f"{value:.6f}" for value in data[method]]])


def write_long_csv(
    path: Path,
    speedup_data: dict[str, list[float]],
    energy_data: dict[str, list[float]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["metric", "method", "sequence", "prefill_tokens", "decode_tokens", "value"])
        for metric, data in [
            ("normalized_speedup", speedup_data),
            ("normalized_energy_efficiency", energy_data),
        ]:
            for method in PLOT_METHODS:
                for index, label in enumerate(X_LABELS_2K):
                    if label == "Geomean":
                        prefill_tokens = ""
                        decode_tokens = ""
                    else:
                        prefill_tokens, decode_tokens = SEQUENCE_PAIRS_2K[index]
                    writer.writerow([metric, method, label, prefill_tokens, decode_tokens, f"{data[method][index]:.6f}"])


def annotate_ultra_geomean(axis: plt.Axes, data: dict[str, list[float]], group_positions: np.ndarray, y_max: float) -> None:
    geomean_index = len(X_LABELS_2K) - 1
    ultra_dsp_geomean = data["Ultra-DSP (INT4)"][-1]
    offset_base = -(BAR_WIDTH_2K * (len(PLOT_METHODS) - 1)) / 2
    ultra_dsp_bar_x = group_positions[geomean_index] + offset_base + (len(PLOT_METHODS) - 1) * BAR_WIDTH_2K
    axis.annotate(
        f"{ultra_dsp_geomean:.2f}x",
        xy=(ultra_dsp_bar_x, ultra_dsp_geomean),
        xytext=(ultra_dsp_bar_x + 0.05, min(ultra_dsp_geomean + 0.05, y_max - 0.02)),
        fontsize=45,
        fontweight="bold",
        color=ANNOTATION_COLOR,
        ha="right",
        va="bottom",
    )


def plot_panel(axis: plt.Axes, normalized_data: dict[str, list[float]], layout: dict[str, object]) -> None:
    method_colors = {method: COLORS[index] for index, method in enumerate(PLOT_METHODS)}
    method_hatches = {method: HATCHES[index] for index, method in enumerate(PLOT_METHODS)}
    group_positions = np.arange(len(X_LABELS_2K)) * GROUP_SPACING_2K
    offset_base = -(BAR_WIDTH_2K * (len(PLOT_METHODS) - 1)) / 2

    for index, method in enumerate(PLOT_METHODS):
        bar_positions = group_positions + offset_base + index * BAR_WIDTH_2K
        values = np.asarray(normalized_data[method], dtype=float)
        axis.bar(
            bar_positions,
            values - Y_MIN,
            BAR_WIDTH_2K,
            bottom=Y_MIN,
            label=METHOD_DISPLAY_NAMES[method],
            color=method_colors[method],
            hatch=method_hatches[method],
            edgecolor=EDGE_COLOR,
            linewidth=0.75,
            zorder=3,
        )

    y_max = float(layout["y_max"])
    annotate_ultra_geomean(axis, normalized_data, group_positions, y_max)
    axis.set_ylabel(str(layout["ylabel"]), fontsize=48, fontweight="bold")
    axis.set_xticks(group_positions)
    axis.set_xticklabels(X_LABELS_2K, fontsize=35, fontweight="bold", rotation=22, ha="center")
    axis.set_yticks(Y_TICKS)
    axis.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    axis.set_ylim(Y_MIN, y_max)
    axis.set_xlim(group_positions[0] - 0.42, group_positions[-1] + 0.42)
    axis.margins(x=0.01)
    axis.tick_params(axis="x", labelsize=35)
    axis.tick_params(axis="y", labelsize=36)
    for tick_label in list(axis.get_xticklabels()) + list(axis.get_yticklabels()):
        tick_label.set_fontweight("bold")
    axis.yaxis.grid(True, linestyle="--", which="major", color=GRID_COLOR, alpha=0.9, linewidth=1.2)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#444444")
    axis.spines["bottom"].set_color("#444444")


def plot_column_figure(speedup_data: dict[str, list[float]], energy_data: dict[str, list[float]]) -> list[Path]:
    figure, axes = plt.subplots(nrows=2, ncols=1, figsize=FIGSIZE)
    plot_panel(axes[0], speedup_data, PANEL_LAYOUTS[0])
    plot_panel(axes[1], energy_data, PANEL_LAYOUTS[1])

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        ncol=3,
        loc="upper center",
        prop={"size": 34, "weight": "bold"},
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        columnspacing=1.0,
        handletextpad=0.45,
        handlelength=1.6,
        borderpad=0.2,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90), h_pad=1.8)

    output_paths = [
        RESULTS_DIR / "long_context_8k_2k_column_speedup_energy.png",
        RESULTS_DIR / "long_context_8k_2k_column_speedup_energy.svg",
        RESULTS_DIR / "long_context_8k_2k_column_speedup_energy.pdf",
    ]
    figure.savefig(output_paths[0], dpi=300, bbox_inches="tight")
    figure.savefig(output_paths[1], format="svg", bbox_inches="tight")
    figure.savefig(output_paths[2], format="pdf", bbox_inches="tight")
    plt.close(figure)
    return output_paths


def main() -> None:
    selected_font = configure_paper_font()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    speedup_data = build_metric_payload(SPEEDUP_STAGE_DATA)
    energy_data = build_metric_payload(ENERGY_STAGE_DATA)

    write_wide_csv(RESULTS_DIR / "long_context_8k_2k_normalized_speedup.csv", speedup_data)
    write_wide_csv(RESULTS_DIR / "long_context_8k_2k_normalized_energy_efficiency.csv", energy_data)
    write_long_csv(RESULTS_DIR / "long_context_8k_2k_points.csv", speedup_data, energy_data)
    output_paths = plot_column_figure(speedup_data, energy_data)

    print(f"Selected font: {selected_font}")
    print("Ultra-DSP 8K 2K-spaced Geomean:")
    print(f"  normalized speedup = {speedup_data['Ultra-DSP (INT4)'][-1]:.4f}x")
    print(f"  normalized energy efficiency = {energy_data['Ultra-DSP (INT4)'][-1]:.4f}x")
    print("Saved files:")
    for output_path in [
        RESULTS_DIR / "long_context_8k_2k_normalized_speedup.csv",
        RESULTS_DIR / "long_context_8k_2k_normalized_energy_efficiency.csv",
        RESULTS_DIR / "long_context_8k_2k_points.csv",
        *output_paths,
    ]:
        print(output_path)


if __name__ == "__main__":
    main()
