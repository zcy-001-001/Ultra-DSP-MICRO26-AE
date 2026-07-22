from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from plot_long_context_2k_column import (
    BAR_WIDTH_2K,
    GROUP_SPACING_2K,
    PANEL_LAYOUTS,
    X_LABELS_2K,
    build_metric_payload,
    write_long_csv,
    write_wide_csv,
)
from plot_long_context import (
    ANNOTATION_COLOR,
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
)


FIGSIZE = (24, 7.0)
PANEL_FONT = {
    "annotation": 38,
    "ylabel": 42,
    "xtick": 32,
    "ytick": 30,
    "legend": 36,
}


def annotate_ultra_geomean(axis: plt.Axes, data: dict[str, list[float]], group_positions: np.ndarray, y_max: float) -> None:
    geomean_index = len(X_LABELS_2K) - 1
    ultra_dsp_geomean = data["Ultra-DSP (INT4)"][-1]
    offset_base = -(BAR_WIDTH_2K * (len(PLOT_METHODS) - 1)) / 2
    ultra_dsp_bar_x = group_positions[geomean_index] + offset_base + (len(PLOT_METHODS) - 1) * BAR_WIDTH_2K
    axis.annotate(
        f"{ultra_dsp_geomean:.2f}x",
        xy=(ultra_dsp_bar_x, ultra_dsp_geomean),
        xytext=(ultra_dsp_bar_x + 0.04, min(ultra_dsp_geomean + 0.05, y_max - 0.02)),
        fontsize=PANEL_FONT["annotation"],
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
    axis.set_ylabel(str(layout["ylabel"]), fontsize=PANEL_FONT["ylabel"], fontweight="bold")
    axis.set_xticks(group_positions)
    axis.set_xticklabels(X_LABELS_2K, fontsize=PANEL_FONT["xtick"], fontweight="bold", rotation=24, ha="center")
    axis.set_yticks(Y_TICKS)
    axis.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    axis.set_ylim(Y_MIN, y_max)
    axis.set_xlim(group_positions[0] - 0.42, group_positions[-1] + 0.42)
    axis.margins(x=0.01)
    axis.tick_params(axis="x", labelsize=PANEL_FONT["xtick"])
    axis.tick_params(axis="y", labelsize=PANEL_FONT["ytick"])
    for tick_label in list(axis.get_xticklabels()) + list(axis.get_yticklabels()):
        tick_label.set_fontweight("bold")
    axis.yaxis.grid(True, linestyle="--", which="major", color=GRID_COLOR, alpha=0.9, linewidth=1.2)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#444444")
    axis.spines["bottom"].set_color("#444444")


def plot_row_figure(speedup_data: dict[str, list[float]], energy_data: dict[str, list[float]]) -> list[Path]:
    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=FIGSIZE)
    plot_panel(axes[0], speedup_data, PANEL_LAYOUTS[0])
    plot_panel(axes[1], energy_data, PANEL_LAYOUTS[1])

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        ncol=len(PLOT_METHODS),
        loc="lower left",
        prop={"size": PANEL_FONT["legend"], "weight": "bold"},
        frameon=False,
        bbox_to_anchor=(0.065, 0.745, 0.87, 0.12),
        mode="expand",
        columnspacing=0.8,
        handletextpad=0.35,
        handlelength=1.8,
        borderpad=0.2,
        borderaxespad=0.0,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.70), w_pad=2.2)

    output_paths = [
        RESULTS_DIR / "long_context_8k_2k_row_speedup_energy.png",
        RESULTS_DIR / "long_context_8k_2k_row_speedup_energy.svg",
        RESULTS_DIR / "long_context_8k_2k_row_speedup_energy.pdf",
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
    output_paths = plot_row_figure(speedup_data, energy_data)

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
