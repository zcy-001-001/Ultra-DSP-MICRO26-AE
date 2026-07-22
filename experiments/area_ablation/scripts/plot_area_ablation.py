#!/usr/bin/env python3
"""Plot 2x3 Ultra-DSP area-ablation figures."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
PACKAGE_ROOT = ROOT_DIR.parents[1]
RESULTS_DIR = PACKAGE_ROOT / "results" / "area_ablation"
DEFAULT_OUT_DIR = RESULTS_DIR / "figures"
DEFAULT_RESOURCE_CSV = RESULTS_DIR / "vivado_resource_summary.csv"
DEFAULT_OPT_CSV = RESULTS_DIR / "optimization_stage_lut_matrix.csv"
# 旧版默认读取 lsb_depth_lut_matrix.csv。该文件来自简单截断 partial-product
# RTL，Vivado 会把低位乘法强优化成近似线性的 LUT 序列，不符合论文 4.2.3
# 中 W4A4 depth 1-6 = 1/2/3/6/11/17 的 LSB generator 口径。
DEFAULT_LSB_CSV = RESULTS_DIR / "lsb_depth_lut_matrix_paper_aligned.csv"

PRECISIONS = ["W3A4", "W3A5", "W4A4", "W4A5", "W5A4", "W5A5"]

FONT_FAMILY = "serif"
FONT_SERIF = ["Times New Roman", "DejaVu Serif"]
FIG_DPI = 300
FIG_FACE_COLOR = "white"
AX_FACE_COLOR = "white"

GRID_LINESTYLE = (0, (4, 3))
GRID_LINEWIDTH = 0.85
# 旧版网格对比度偏高：GRID_ALPHA = 0.75, GRID_COLOR = "#8E8E8E"
GRID_ALPHA = 0.55
GRID_COLOR = "#B0B0B0"
# 旧版坐标轴和柱边框使用 "#2F2F2F"，论文图中整体线条更柔和，这里改为浅一些的深灰。
SPINE_COLOR = "#4A4A4A"
SPINE_LINEWIDTH = 1.0

BAR_EDGE_COLOR = "#4A4A4A"
BAR_EDGE_WIDTH = 1.05
VALUE_LABEL_COLOR = "#3F3F3F"
FF_HATCH = "////"

# 旧版配色为 ["#4C78A8", "#9ECAE9", "#72B7B2", "#ECA82C"]。
# 当前按用户指定的浅绿色渐变色板绘制四个累计优化阶段。
# OLD_OPT_COLORS = ["#98D890", "#50B898", "#30A0A0", "#1888B0"]
OPT_COLORS = ["#D9ED92", "#B5E48C", "#99D98C", "#76C893"]
OPT_STAGE_KEYS = [
    "s0_original",
    "s1_weight_offline",
    "s2_registered_c",
    "s3_xor_sign",
]
OPT_STAGE_LABELS = ["S0", "S1", "S2", "S3"]
# 旧版标签使用较多缩写；按 rebuttal 图示需求改为“累计添加优化项”的短标签。
# OLD_OPT_STAGE_PAPER_LABELS = ["Original", "Offline\nW", "Reg.\nC-port", "XOR-only\nSign"]
# OLD_OPT_STAGE_PAPER_LABELS = ["Baseline", "Offline\nWeight", "Registered\nC-corr.", "XOR-only\nSign"]
# OLD_OPT_STAGE_PAPER_LABELS = ["Base", "+OPT1", "+OPT1\n+OPT2", "+OPT1\n+OPT2\n+OPT3"]
OPT_STAGE_PAPER_LABELS = [
    "Base",
    "+Offline\nWeight",
    "+Registered\nC-port",
    "+XOR\nSign",
]
# 旧版在图底部解释 A/B/C；用户希望在图标题或 caption 中解释 OPT1/OPT2/OPT3，因此不再绘制底部说明。
# OLD_OPT_STAGE_EXPLAINER = "A: Offline weight    B: Registered C-correction    C: XOR-only sign"
OPT_STAGE_LEGEND = [
    "S0 Runtime W",
    "S1 Offline W",
    "S2 Reg-C",
    "S3 XOR sign",
]

LSB_COLOR = "#C7D4EA"
LSB_HATCH = "--"
LSB_DEPTH_GROUPS = [
    ("W3", ("W3A4", "W3A5")),
    ("W4", ("W4A4", "W4A5")),
    ("W5", ("W5A4", "W5A5")),
]
LSB_LINE_COLORS = {
    "A4": "#76C893",
    "A5": "#1888B0",
}
LSB_LINE_MARKERS = {
    "A4": "o",
    "A5": "s",
}
# W3 的 magnitude 维度不足以支持 depth=6；旧版直接画出数值点，
# 这里保留数据文件记录，但在图中显式标成 unsupported。
LSB_UNSUPPORTED_POINTS = {
    ("W3A4", 6),
}


def configure_matplotlib() -> None:
    plt.rcParams["font.family"] = FONT_FAMILY
    plt.rcParams["font.serif"] = FONT_SERIF
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["hatch.linewidth"] = 1.0


def read_matrix(path: Path) -> dict[str, dict[str, int]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    matrix: dict[str, dict[str, int]] = {}
    for row in rows:
        precision = row["precision"]
        matrix[precision] = {}
        for key, value in row.items():
            if key == "precision" or value == "":
                continue
            if value.upper() in {"N/A", "NA"}:
                # Unsupported design points are omitted from the numeric matrix
                # and rendered explicitly by plot_lsb_depth().
                continue
            matrix[precision][key] = int(value)
    return matrix


def read_resource_summary(path: Path) -> dict[str, dict[str, int]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    summary: dict[str, dict[str, int]] = {}
    for row in rows:
        top = row["top_module"]
        summary[top] = {
            "lut": int(row["clb_luts"]),
            "ff": int(row["clb_registers"]),
        }
    return summary


def style_axis(
    ax,
    show_ylabel: bool,
    ylabel: str,
    ylim: tuple[int, int],
    yticks: np.ndarray,
    tick_label_size: int = 12,
) -> None:
    ax.set_facecolor(AX_FACE_COLOR)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.grid(
        axis="y",
        linestyle=GRID_LINESTYLE,
        linewidth=GRID_LINEWIDTH,
        alpha=GRID_ALPHA,
        color=GRID_COLOR,
        zorder=0,
    )
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)
        spine.set_linewidth(SPINE_LINEWIDTH)
    ax.tick_params(axis="x", labelsize=tick_label_size, pad=5)
    ax.tick_params(axis="y", labelsize=tick_label_size)
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=14, fontweight="bold")
    else:
        ax.set_ylabel("")


def annotate_bars(ax, bars, offset: float, font_size: int = 10) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=font_size,
            fontweight="semibold",
            color=VALUE_LABEL_COLOR,
        )


def plot_optimization_lut_ff(summary: dict[str, dict[str, int]], out_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(
        2,
        3,
        # 旧版高度为 6.25；这里压缩高度以贴近论文中的紧凑 2x3 子图版式。
        # OLD_FIGSIZE = (13.2, 5.85)
        # 当前版本放大论文插图中的可读字号，同时增加画布避免标签重叠。
        figsize=(15.6, 7.1),
        dpi=FIG_DPI,
        sharey=True,
    )
    fig.patch.set_facecolor(FIG_FACE_COLOR)

    x = np.arange(len(OPT_STAGE_KEYS))
    bar_width = 0.18
    offset = 0.10

    for idx, (ax, precision) in enumerate(zip(axes.flat, PRECISIONS)):
        lut_values = []
        ff_values = []
        for stage in OPT_STAGE_KEYS:
            top = f"{precision}_stage_{stage}"
            lut_values.append(summary[top]["lut"])
            ff_values.append(summary[top]["ff"])

        lut_bars = ax.bar(
            x - offset,
            lut_values,
            width=bar_width,
            color=OPT_COLORS,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH,
            zorder=3,
        )
        ff_bars = ax.bar(
            x + offset,
            ff_values,
            width=bar_width,
            color=OPT_COLORS,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH,
            hatch=FF_HATCH,
            zorder=3,
        )

        style_axis(
            ax,
            show_ylabel=False,
            ylabel="Resource Usage Per DSP",
            ylim=(0, 160),
            # 旧版每 20 标一个刻度，2x3 小图中偏密；改为每 40 标一个刻度。
            yticks=np.arange(0, 161, 40),
            tick_label_size=18,
        )
        if idx % 3 != 0:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_xticks(x)
        if idx < 3:
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xticklabels(OPT_STAGE_PAPER_LABELS, fontsize=20, fontweight="normal")
        ax.set_title(precision, fontsize=25, fontweight="bold", pad=9)
        annotate_bars(ax, lut_bars, offset=2.4, font_size=16)
        annotate_bars(ax, ff_bars, offset=2.4, font_size=16)

    legend_handles = [
        Patch(
            facecolor="white",
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH,
            label="LUT",
        ),
        Patch(
            facecolor="white",
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH,
            hatch=FF_HATCH,
            label="FF",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        # 旧版图例贴近画布顶部；下移一些以便论文中裁剪/缩放后更稳。
        bbox_to_anchor=(0.5, 0.965),
        frameon=False,
        fontsize=23,
        handlelength=2.4,
        handleheight=1.2,
        columnspacing=2.0,
    )
    fig.supylabel("Resource Usage Per DSP", fontsize=28, fontweight="bold", x=0.012)
    # 旧版这里用 fig.text 绘制 A/B/C 说明；当前版本留给图标题或 caption 解释。
    fig.subplots_adjust(left=0.066, right=0.995, bottom=0.24, top=0.80, wspace=0.14, hspace=0.48)

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / "optimization_stage_lut_ff_2x3.pdf"
    png = out_dir / "optimization_stage_lut_ff_2x3.png"
    fig.savefig(pdf, bbox_inches="tight", facecolor=FIG_FACE_COLOR)
    fig.savefig(png, bbox_inches="tight", facecolor=FIG_FACE_COLOR)
    plt.close(fig)
    return [pdf, png]


def plot_optimization(matrix: dict[str, dict[str, int]], out_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(13.2, 7.2),
        dpi=FIG_DPI,
        sharey=True,
    )
    fig.patch.set_facecolor(FIG_FACE_COLOR)

    x = np.arange(len(OPT_STAGE_KEYS))
    for idx, (ax, precision) in enumerate(zip(axes.flat, PRECISIONS)):
        values = [matrix[precision][key] for key in OPT_STAGE_KEYS]
        bars = ax.bar(
            x,
            values,
            width=0.58,
            color=OPT_COLORS,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH,
            zorder=3,
        )
        style_axis(
            ax,
            show_ylabel=(idx % 3 == 0),
            ylabel="CLB LUTs",
            ylim=(0, 150),
            yticks=np.arange(0, 151, 30),
        )
        ax.set_xticks(x)
        ax.set_xticklabels(OPT_STAGE_LABELS, fontsize=12, fontweight="bold")
        ax.set_title(precision, fontsize=15, fontweight="bold", pad=6)
        annotate_bars(ax, bars, offset=2.0)

    legend_handles = [
        Patch(
            facecolor=color,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH,
            label=label,
        )
        for color, label in zip(OPT_COLORS, OPT_STAGE_LEGEND)
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.01),
        frameon=True,
        edgecolor="#000000",
        fontsize=13,
    )
    fig.supxlabel("Optimization Stage", fontsize=15, fontweight="bold", y=0.035)
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.11, top=0.88, wspace=0.14, hspace=0.35)

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / "optimization_stage_2x3.pdf"
    png = out_dir / "optimization_stage_2x3.png"
    fig.savefig(pdf, bbox_inches="tight", facecolor=FIG_FACE_COLOR)
    fig.savefig(png, bbox_inches="tight", facecolor=FIG_FACE_COLOR)
    plt.close(fig)
    return [pdf, png]


def plot_lsb_depth_2x3_legacy(matrix: dict[str, dict[str, int]], out_dir: Path) -> list[Path]:
    """Legacy 2x3 bar+line LSB plot kept for reference.

    The current rebuttal figure uses plot_lsb_depth(), which groups W3/W4/W5
    into a 1x3 layout and compares A4/A5 as two lines.
    """
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(13.2, 7.2),
        dpi=FIG_DPI,
        sharey=True,
    )
    fig.patch.set_facecolor(FIG_FACE_COLOR)

    depth_keys = [f"depth{i}" for i in range(1, 7)]
    depth_labels = [str(i) for i in range(1, 7)]
    x = np.arange(len(depth_keys))
    for idx, (ax, precision) in enumerate(zip(axes.flat, PRECISIONS)):
        values = [matrix[precision][key] for key in depth_keys]
        bars = ax.bar(
            x,
            values,
            width=0.55,
            color=LSB_COLOR,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH,
            hatch=LSB_HATCH,
            zorder=3,
        )
        ax.plot(
            x,
            values,
            color="#4C78A8",
            marker="o",
            markersize=4.5,
            linewidth=1.7,
            zorder=4,
        )
        style_axis(
            ax,
            show_ylabel=(idx % 3 == 0),
            ylabel="CLB LUTs",
            ylim=(0, 12),
            yticks=np.arange(0, 13, 2),
        )
        ax.set_xticks(x)
        ax.set_xticklabels(depth_labels, fontsize=12, fontweight="bold")
        ax.set_title(precision, fontsize=15, fontweight="bold", pad=6)
        annotate_bars(ax, bars, offset=0.18)

    legend_handles = [
        Patch(
            facecolor=LSB_COLOR,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH,
            hatch=LSB_HATCH,
            label="Truncated LSB logic",
        )
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=1,
        bbox_to_anchor=(0.5, 1.01),
        frameon=True,
        edgecolor="#000000",
        fontsize=13,
    )
    fig.supxlabel("Precomputed-LSB Depth", fontsize=15, fontweight="bold", y=0.035)
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.11, top=0.88, wspace=0.14, hspace=0.35)

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / "lsb_depth_2x3.pdf"
    png = out_dir / "lsb_depth_2x3.png"
    fig.savefig(pdf, bbox_inches="tight", facecolor=FIG_FACE_COLOR)
    fig.savefig(png, bbox_inches="tight", facecolor=FIG_FACE_COLOR)
    plt.close(fig)
    return [pdf, png]


def plot_lsb_depth(matrix: dict[str, dict[str, int]], out_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(
        1,
        3,
        # 放大字号时同步加宽、加高画布，避免标题、图例和坐标轴文字重叠。
        # OLD_FIGSIZE = (12.6, 3.65)
        figsize=(13.8, 4.35),
        dpi=FIG_DPI,
        sharey=True,
    )
    fig.patch.set_facecolor(FIG_FACE_COLOR)

    depth_keys = [f"depth{i}" for i in range(1, 7)]
    depth_values = np.arange(1, 7)

    for idx, (ax, (w_label, precisions)) in enumerate(zip(axes, LSB_DEPTH_GROUPS)):
        style_axis(
            ax,
            show_ylabel=False,
            ylabel="LSB Generator LUTs",
            ylim=(0, 27),
            yticks=np.arange(0, 26, 5),
            tick_label_size=17,
        )
        if idx != 0:
            ax.tick_params(axis="y", labelleft=False)

        for precision in precisions:
            a_label = precision[-2:]
            values = [matrix[precision].get(key) for key in depth_keys]
            supported_points = [
                (x_pos, y_pos)
                for x_pos, y_pos in zip(depth_values, values)
                if y_pos is not None and (precision, int(x_pos)) not in LSB_UNSUPPORTED_POINTS
            ]
            unsupported_points = [
                (x_pos, y_pos)
                for x_pos, y_pos in zip(depth_values, values)
                if y_pos is None or (precision, int(x_pos)) in LSB_UNSUPPORTED_POINTS
            ]
            line_x = [point[0] for point in supported_points]
            line_y = [point[1] for point in supported_points]
            ax.plot(
                line_x,
                line_y,
                color=LSB_LINE_COLORS[a_label],
                marker=LSB_LINE_MARKERS[a_label],
                markersize=5.5,
                linewidth=2.0,
                label=a_label,
                zorder=4,
            )
            for x_pos, _ in unsupported_points:
                marker_y = line_y[-1] if line_y else 0
                ax.scatter(
                    [x_pos],
                    [marker_y],
                    color=LSB_LINE_COLORS[a_label],
                    marker="x",
                    s=88,
                    linewidths=2.4,
                    zorder=6,
                )

            for x_pos, y_pos in supported_points:
                label_offset = -0.68 if a_label == "A4" else 0.75
                ax.text(
                    x_pos,
                    y_pos + label_offset,
                    f"{y_pos}",
                    ha="center",
                    va="top" if a_label == "A4" else "bottom",
                    fontsize=13,
                    fontweight="semibold",
                    color=VALUE_LABEL_COLOR,
                    zorder=5,
                )

        ax.set_title(w_label, fontsize=22, fontweight="bold", pad=9)
        ax.set_xticks(depth_values)
        ax.set_xticklabels([str(v) for v in depth_values], fontsize=17, fontweight="bold")
        ax.set_xlabel("Precomputed-LSB Depth", fontsize=18, fontweight="bold", labelpad=8)

    legend_handles = []
    for label in ["A4", "A5"]:
        handle = plt.Line2D(
            [0],
            [0],
            color=LSB_LINE_COLORS[label],
            marker=LSB_LINE_MARKERS[label],
            markersize=6,
            linewidth=2.0,
            label=label,
        )
        legend_handles.append(handle)

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.05),
        frameon=True,
        edgecolor=BAR_EDGE_COLOR,
        fontsize=17,
        columnspacing=1.8,
    )
    fig.supylabel("LSB Generator LUTs", fontsize=22, fontweight="bold", x=0.012)
    fig.subplots_adjust(left=0.066, right=0.995, bottom=0.24, top=0.78, wspace=0.16)

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / "lsb_depth_1x3.pdf"
    png = out_dir / "lsb_depth_1x3.png"
    fig.savefig(pdf, bbox_inches="tight", facecolor=FIG_FACE_COLOR)
    fig.savefig(png, bbox_inches="tight", facecolor=FIG_FACE_COLOR)
    plt.close(fig)
    return [pdf, png]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-csv", type=Path, default=DEFAULT_RESOURCE_CSV)
    parser.add_argument("--optimization-csv", type=Path, default=DEFAULT_OPT_CSV)
    parser.add_argument("--lsb-csv", type=Path, default=DEFAULT_LSB_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--only",
        choices=["all", "optimization", "lsb"],
        default="all",
        help="Select which figure family to regenerate.",
    )
    args = parser.parse_args()

    configure_matplotlib()
    resource_summary = read_resource_summary(args.resource_csv)
    opt_matrix = read_matrix(args.optimization_csv)
    lsb_matrix = read_matrix(args.lsb_csv)

    written = []
    if args.only in ("all", "optimization"):
        written.extend(plot_optimization_lut_ff(resource_summary, args.out_dir))
    if args.only in ("all", "lsb"):
        written.extend(plot_lsb_depth(lsb_matrix, args.out_dir))

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
