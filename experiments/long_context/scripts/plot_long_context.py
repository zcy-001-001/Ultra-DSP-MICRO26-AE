from __future__ import annotations

import argparse
import csv
from pathlib import Path
from zipfile import ZipFile

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


ARTIFACT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ARTIFACT_DIR.parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "long_context"
FIGURE_DIR = REPO_ROOT.parent / "Figure"
FONT_ZIP_PATH = FIGURE_DIR / "figs" / "Libertinus-7.051.zip"
FONT_CACHE_DIR = FIGURE_DIR / ".font_cache" / "libertinus"

K = 1024
SEQUENCE_PAIRS = [
    (0 * K, 8 * K),
    (1 * K, 7 * K),
    (2 * K, 6 * K),
    (3 * K, 5 * K),
    (4 * K, 4 * K),
    (5 * K, 3 * K),
    (6 * K, 2 * K),
    (7 * K, 1 * K),
    (8 * K, 0 * K),
]
X_LABELS = [f"[{p // K}K, {d // K}K]" for p, d in SEQUENCE_PAIRS] + ["Geomean"]

PLOT_METHODS = [
    "FlightLLM (INT8)",
    "WP521 (INT4)",
    "DuoQ (INT4)",
    "DSP-Packing (INT4)",
    "UDP (INT4)",
    "Ultra-DSP (INT4)",
]
METHOD_DISPLAY_NAMES = {method: method.split(" (")[0] for method in PLOT_METHODS}
BASELINE_METHOD = "FlightLLM (INT8)"

ENERGY_STAGE_DATA = {
    "FlightLLM (INT8)": {"P": 0.0683, "D": 0.0224},
    "WP521 (INT4)": {"P": 0.1209, "D": 0.0225},
    "DSP-Packing (INT4)": {"P": 0.1595, "D": 0.0222},
    "UDP (INT4)": {"P": 0.1571, "D": 0.0218},
    "DuoQ (INT4)": {"P": 0.1169, "D": 0.0219},
    "Ultra-DSP (INT4)": {"P": 0.1920, "D": 0.0255},
}

SPEEDUP_STAGE_DATA = {
    "FlightLLM (INT8)": {"P": 2.83, "D": 0.93},
    "WP521 (INT4)": {"P": 4.97, "D": 0.93},
    "DuoQ (INT4)": {"P": 4.97, "D": 0.93},
    "DSP-Packing (INT4)": {"P": 6.64, "D": 0.93},
    "UDP (INT4)": {"P": 6.64, "D": 0.93},
    "Ultra-DSP (INT4)": {"P": 8.57, "D": 0.93},
}

COLORS = ["#99D98C", "#76C893", "#52B69A", "#34A0A4", "#168AAD", "#1A759F"]
HATCHES = ["", "/", "\\", ".", "x", "o"]
EDGE_COLOR = "#2E2E2E"
GRID_COLOR = "#9A9A9A"
ANNOTATION_COLOR = "#B0171F"

Y_MIN = 0.5
Y_MAX = 3.0
Y_TICKS = np.arange(0.5, 3.1, 0.5)
GROUP_SPACING = 0.80
BAR_WIDTH = 0.10

ENERGY_LAYOUT = {
    "ylabel": "Norm. Energy Eff.",
    "show_top_legend": False,
    "y_max": 3.0,
    "figsize": (24, 9.7),
    "annotation_fontsize": 60,
    "ylabel_fontsize": 63,
    "xtick_fontsize": 48,
    "ytick_fontsize": 46,
    "legend_fontsize": 39,
    "stem": "long_context_8k_normalized_energy_efficiency",
}

SPEEDUP_LAYOUT = {
    "ylabel": "Norm. Speedup",
    "show_top_legend": True,
    # Keep the paper Figure 11 y-axis framing exactly. The 8K pure-prefill
    # Ultra-DSP value is slightly above 3.0x, so this preserves visual parity.
    "y_max": 3.0,
    "figsize": (24, 11.7),
    "annotation_fontsize": 60,
    "ylabel_fontsize": 63,
    "xtick_fontsize": 48,
    "ytick_fontsize": 46,
    "legend_fontsize": 40,
    "stem": "long_context_8k_normalized_speedup",
}


def extract_local_serif_fonts() -> list[Path]:
    if not FONT_ZIP_PATH.exists():
        return []

    FONT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target_files = {
        "LibertinusSerif-Regular.ttf",
        "LibertinusSerif-Bold.ttf",
        "LibertinusSerif-Italic.ttf",
        "LibertinusSerif-BoldItalic.ttf",
        "LibertinusSerif-Semibold.ttf",
        "LibertinusSerif-SemiboldItalic.ttf",
    }
    extracted_paths: list[Path] = []
    with ZipFile(FONT_ZIP_PATH) as zip_file:
        for member in zip_file.namelist():
            filename = Path(member).name
            if filename not in target_files:
                continue
            output_path = FONT_CACHE_DIR / filename
            if not output_path.exists():
                output_path.write_bytes(zip_file.read(member))
            extracted_paths.append(output_path)
    return extracted_paths


def configure_paper_font() -> str:
    extracted_font_names = []
    for font_path in extract_local_serif_fonts():
        try:
            fm.fontManager.addfont(str(font_path))
            extracted_font_names.append(fm.FontProperties(fname=str(font_path)).get_name())
        except RuntimeError:
            continue

    available_names = {font.name for font in fm.fontManager.ttflist}
    available_names.update(extracted_font_names)
    preferred_fonts = [
        "Libertinus Serif",
        "Linux Libertine O",
        "LinLibertine",
        "Times New Roman",
        "DejaVu Serif",
    ]
    selected_font = next((name for name in preferred_fonts if name in available_names), "DejaVu Serif")

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": preferred_fonts,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "hatch.linewidth": 0.75,
        }
    )
    return selected_font


def geometric_mean(values: list[float]) -> float:
    safe_values = np.clip(np.asarray(values, dtype=float), 1e-12, None)
    return float(np.exp(np.mean(np.log(safe_values))))


def get_sequence_weighted_energy(stage_values: dict[str, float], prefill_len: int, decode_len: int) -> float:
    return (stage_values["P"] * prefill_len + stage_values["D"] * decode_len) / (prefill_len + decode_len)


def get_sequence_weighted_ratio(
    method_values: dict[str, float],
    baseline_values: dict[str, float],
    prefill_len: int,
    decode_len: int,
) -> float:
    prefill_ratio = method_values["P"] / baseline_values["P"]
    decode_ratio = method_values["D"] / baseline_values["D"]
    return (prefill_ratio * prefill_len + decode_ratio * decode_len) / (prefill_len + decode_len)


def build_metric_payload(
    stage_data: dict[str, dict[str, float]],
) -> tuple[dict[str, dict[str, float]], dict[str, list[float]], dict[str, list[float]]]:
    stage_summary: dict[str, dict[str, float]] = {}
    raw_data: dict[str, list[float]] = {}
    normalized_data: dict[str, list[float]] = {}
    for method in PLOT_METHODS:
        raw_values = [
            get_sequence_weighted_energy(stage_data[method], prefill_len, decode_len)
            for prefill_len, decode_len in SEQUENCE_PAIRS
        ]
        normalized_values = [
            get_sequence_weighted_ratio(stage_data[method], stage_data[BASELINE_METHOD], prefill_len, decode_len)
            for prefill_len, decode_len in SEQUENCE_PAIRS
        ]
        stage_summary[method] = {
            "P_raw": stage_data[method]["P"],
            "D_raw": stage_data[method]["D"],
            "P_norm": stage_data[method]["P"] / stage_data[BASELINE_METHOD]["P"],
            "D_norm": stage_data[method]["D"] / stage_data[BASELINE_METHOD]["D"],
        }
        raw_data[method] = raw_values + [geometric_mean(raw_values)]
        normalized_data[method] = normalized_values + [geometric_mean(normalized_values)]
    return stage_summary, raw_data, normalized_data


def write_wide_csv(path: Path, data: dict[str, list[float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["method", *X_LABELS])
        for method in PLOT_METHODS:
            writer.writerow([method, *[f"{value:.6f}" for value in data[method]]])


def write_long_csv(path: Path, speedup_data: dict[str, list[float]], energy_data: dict[str, list[float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["metric", "method", "sequence", "prefill_tokens", "decode_tokens", "value"])
        for metric, data in [
            ("normalized_speedup", speedup_data),
            ("normalized_energy_efficiency", energy_data),
        ]:
            for method in PLOT_METHODS:
                for index, label in enumerate(X_LABELS):
                    if label == "Geomean":
                        prefill_tokens = ""
                        decode_tokens = ""
                    else:
                        prefill_tokens, decode_tokens = SEQUENCE_PAIRS[index]
                    writer.writerow([metric, method, label, prefill_tokens, decode_tokens, f"{data[method][index]:.6f}"])


def plot_metric(normalized_data: dict[str, list[float]], layout: dict[str, object]) -> list[Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    method_colors = {method: COLORS[index] for index, method in enumerate(PLOT_METHODS)}
    method_hatches = {method: HATCHES[index] for index, method in enumerate(PLOT_METHODS)}
    group_positions = np.arange(len(X_LABELS)) * GROUP_SPACING
    offset_base = -(BAR_WIDTH * (len(PLOT_METHODS) - 1)) / 2

    figure, axis = plt.subplots(figsize=layout["figsize"])
    y_max = float(layout.get("y_max", Y_MAX))
    for index, method in enumerate(PLOT_METHODS):
        bar_positions = group_positions + offset_base + index * BAR_WIDTH
        values = np.asarray(normalized_data[method], dtype=float)
        axis.bar(
            bar_positions,
            values - Y_MIN,
            BAR_WIDTH,
            bottom=Y_MIN,
            label=METHOD_DISPLAY_NAMES[method],
            color=method_colors[method],
            hatch=method_hatches[method],
            edgecolor=EDGE_COLOR,
            linewidth=0.75,
            zorder=3,
        )

    geomean_index = len(X_LABELS) - 1
    ultra_dsp_geomean = normalized_data["Ultra-DSP (INT4)"][-1]
    ultra_dsp_bar_x = group_positions[geomean_index] + offset_base + (len(PLOT_METHODS) - 1) * BAR_WIDTH
    axis.annotate(
        f"{ultra_dsp_geomean:.2f}x",
        xy=(ultra_dsp_bar_x, ultra_dsp_geomean),
        xytext=(ultra_dsp_bar_x + 0.06, min(ultra_dsp_geomean + 0.06, y_max - 0.02)),
        fontsize=layout["annotation_fontsize"],
        fontweight="bold",
        color=ANNOTATION_COLOR,
        ha="right",
        va="bottom",
    )

    axis.set_ylabel(layout["ylabel"], fontsize=layout["ylabel_fontsize"], fontweight="bold")
    axis.set_xticks(group_positions)
    axis.set_xticklabels(X_LABELS, fontsize=layout["xtick_fontsize"], fontweight="bold", rotation=26, ha="center")
    axis.set_yticks(Y_TICKS)
    axis.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    axis.set_ylim(Y_MIN, y_max)
    axis.set_xlim(group_positions[0] - 0.45, group_positions[-1] + 0.45)
    axis.margins(x=0.01)
    axis.tick_params(axis="x", labelsize=layout["xtick_fontsize"])
    axis.tick_params(axis="y", labelsize=layout["ytick_fontsize"])
    for tick_label in list(axis.get_xticklabels()) + list(axis.get_yticklabels()):
        tick_label.set_fontweight("bold")
    axis.yaxis.grid(True, linestyle="--", which="major", color=GRID_COLOR, alpha=0.9, linewidth=1.2)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#444444")
    axis.spines["bottom"].set_color("#444444")

    if layout["show_top_legend"]:
        axis.legend(
            ncol=len(PLOT_METHODS),
            loc="lower left",
            prop={"size": layout["legend_fontsize"], "weight": "bold"},
            frameon=False,
            bbox_to_anchor=(0.01, 1.02, 0.98, 0.16),
            mode="expand",
            columnspacing=0.9,
            handletextpad=0.4,
            handlelength=1.7,
            borderpad=0.2,
            borderaxespad=0.0,
        )

    layout_top = 0.84 if layout["show_top_legend"] else 0.97
    figure.tight_layout(rect=(0, 0, 1, layout_top))
    output_paths = [
        RESULTS_DIR / f"{layout['stem']}.png",
        RESULTS_DIR / f"{layout['stem']}.svg",
        RESULTS_DIR / f"{layout['stem']}.pdf",
    ]
    figure.savefig(output_paths[0], dpi=300, bbox_inches="tight")
    figure.savefig(output_paths[1], format="svg", bbox_inches="tight")
    figure.savefig(output_paths[2], format="pdf", bbox_inches="tight")
    plt.close(figure)
    return output_paths


def main() -> None:
    global RESULTS_DIR
    parser = argparse.ArgumentParser(description="Generate long-context support tables and plots.")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    RESULTS_DIR = args.out_dir

    selected_font = configure_paper_font()
    _, _, normalized_energy = build_metric_payload(ENERGY_STAGE_DATA)
    _, _, normalized_speedup = build_metric_payload(SPEEDUP_STAGE_DATA)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_wide_csv(RESULTS_DIR / "long_context_8k_normalized_speedup.csv", normalized_speedup)
    write_wide_csv(RESULTS_DIR / "long_context_8k_normalized_energy_efficiency.csv", normalized_energy)
    write_long_csv(RESULTS_DIR / "long_context_8k_points.csv", normalized_speedup, normalized_energy)

    speedup_paths = plot_metric(normalized_speedup, SPEEDUP_LAYOUT)
    energy_paths = plot_metric(normalized_energy, ENERGY_LAYOUT)

    print(f"Selected font: {selected_font}")
    print("Ultra-DSP 8K Geomean:")
    print(f"  normalized speedup = {normalized_speedup['Ultra-DSP (INT4)'][-1]:.4f}x")
    print(f"  normalized energy efficiency = {normalized_energy['Ultra-DSP (INT4)'][-1]:.4f}x")
    print("Saved files:")
    for output_path in [
        RESULTS_DIR / "long_context_8k_normalized_speedup.csv",
        RESULTS_DIR / "long_context_8k_normalized_energy_efficiency.csv",
        RESULTS_DIR / "long_context_8k_points.csv",
        *speedup_paths,
        *energy_paths,
    ]:
        print(output_path)


if __name__ == "__main__":
    main()
