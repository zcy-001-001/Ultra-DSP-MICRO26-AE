#!/usr/bin/env python3
"""绘制 W4A4 组件消融的 1x2 rebuttal 图。"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PACKAGE_ROOT / "results" / "component_ablation"


# =========================
# 可调参数区
# =========================

# Variant 顺序和横坐标标签。当前图把 raw overlap 与 correction 合并成
# Lossless Overpacking，同时把 ILP Layout 与 Resource Opt. 拆成两列。
ORDER = ["V0", "V1", "V3", "V4_ILP", "V4"]
XLABELS = [
    "Normal\nSigned\nPacking",
    "+ Sign-\nMagnitude",
    "+ Lossless\nOverpacking",
    "+ ILP\nLayout",
    "+ Resource\nOpt.",
]

# 图像尺寸和导出分辨率。
FIG_SIZE = (14.2, 4.9)
FIG_DPI = 300
SUBPLOT_ADJUST = {
    "left": 0.085,
    "right": 0.995,
    "bottom": 0.32,
    "top": 0.78,
    "wspace": 0.36,
}

# 字体设置。论文图建议使用 serif；如果没有 Times New Roman，会回退到 DejaVu Serif。
FONT_FAMILY = "serif"
FONT_SERIF = ["Times New Roman", "DejaVu Serif"]
FONT_SIZE = {
    "axis_label": 19,
    "tick_label": 14,
    "legend": 19,
    "bar_label": 14,
    "resource_label": 13,
}

# 颜色、边框和 hatch。两张子图使用低饱和、浅对比 palette：
# P/D 和 LUT/FF 两组颜色互相区分，避免不同指标被误读为同一含义。
PACKING_P_COLOR = "#A8DADC"      # 浅蓝：P packing
PACKING_D_COLOR = "#CDB4DB"      # 浅紫：D packing
RESOURCE_LUT_COLOR = "#B7E4A8"   # 浅绿：LUT
RESOURCE_FF_COLOR = "#76C7B7"    # 青绿：FF
EDGE_COLOR = "#2F2F2F"
LEFT_HATCH = "xx"
RIGHT_HATCH = "////"
BAR_EDGE_WIDTH = 1.2

# 柱宽与数值标注偏移。
GROUPED_BAR_WIDTH = 0.34
PACKING_LABEL_OFFSET = 0.15
RESOURCE_LABEL_OFFSET = 3.0
RESOURCE_TBD_Y = 5.0

# Legend 样式。P/D 和 LUT/FF 两个 legend 共用这一套位置参数。
LEGEND_CONFIG = {
    "loc": "upper center",
    "bbox_to_anchor": (0.5, 1.23),
    "ncol": 2,
    "frameon": True,
    "handlelength": 1.2,
    "handletextpad": 0.5,
    "borderpad": 0.25,
    "columnspacing": 1.0,
}
LEGEND_EDGE_COLOR = "#000000"
LEGEND_EDGE_WIDTH = 0.9

# 网格线样式。
GRID_LINESTYLE = (0, (4, 3))
GRID_COLOR = "#000000"


def read_csv(path: Path) -> list[dict[str, str]]:
    """读取 CSV 并返回字典列表。"""
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(value: str) -> float | None:
    """把 CSV 字段转成 float；TBD/NA 保留为空值，绘图时单独标注。"""
    if value in {"", "TBD", "NA", "N/A"}:
        return None
    return float(value)


def scale_optional(value: float | None, scale: int) -> float | None:
    """按共享 P+D datapath 数量放大资源；空值仍保持为空。"""
    if value is None:
        return None
    return value * scale


def nice_step(value: float) -> float:
    """生成比较规整的 y 轴刻度间隔。"""
    if value <= 0:
        return 1
    exponent = 10 ** math.floor(math.log10(value))
    fraction = value / exponent
    for step in [1, 1.5, 2, 2.5, 4, 5, 10]:
        if fraction <= step:
            return step * exponent
    return 10 * exponent


def apply_style(ax) -> None:
    """统一两个子图的网格线、边框和 tick 样式。"""
    ax.grid(axis="y", linestyle=GRID_LINESTYLE, linewidth=0.9, alpha=0.8, color=GRID_COLOR, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(EDGE_COLOR)
    ax.spines["bottom"].set_color(EDGE_COLOR)
    ax.tick_params(axis="x", length=0, labelsize=FONT_SIZE["tick_label"])
    ax.tick_params(axis="y", labelsize=FONT_SIZE["tick_label"])


def set_zero_based_ticks(ax, values: list[float], tick_count: int = 5, top_padding: float = 0.0) -> None:
    """让 y 轴从 0 开始，并根据最大值自动选择上界和刻度。"""
    vmax = max(values) if values else 1
    step = nice_step(vmax / tick_count)
    upper = step * math.ceil(max(vmax + top_padding, 1e-9) / step)
    ax.set_ylim(0, upper)
    ax.set_yticks(np.arange(0, upper + 0.5 * step, step))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accuracy", type=Path, default=RESULTS_DIR / "component_ablation_w4a4.csv")
    parser.add_argument("--resources", default="data/resource_notes_w4a4.csv")
    parser.add_argument("--out-prefix", type=Path, default=RESULTS_DIR / "component_ablation_w4a4")
    args = parser.parse_args()

    resource_rows = {r["variant"]: r for r in read_csv(Path(args.resources))}

    # 子图 1：P/D 两阶段 packing number，分别画柱。
    # 这里从 resource_notes_w4a4.csv 读取，便于加入只用于绘图的 V4_ILP 中间点。
    prefill_t = [float(resource_rows[v]["prefill_t"]) for v in ORDER]
    decode_t = [float(resource_rows[v]["decode_t"]) for v in ORDER]

    # LUT/FF 资源。若某方案需要两套 P/D 硬件，则用 shared_pd_datapaths 放大。
    lut_raw = [to_float(resource_rows[v]["lut_per_dsp"]) for v in ORDER]
    ff_raw = [to_float(resource_rows[v]["ff_per_dsp"]) for v in ORDER]
    shared_pd_datapaths = [int(resource_rows[v].get("shared_pd_datapaths", "1")) for v in ORDER]
    lut = [scale_optional(value, count) for value, count in zip(lut_raw, shared_pd_datapaths)]
    ff = [scale_optional(value, count) for value, count in zip(ff_raw, shared_pd_datapaths)]

    plt.rcParams["font.family"] = FONT_FAMILY
    plt.rcParams["font.serif"] = FONT_SERIF
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["hatch.linewidth"] = 0.9

    x = np.arange(len(ORDER))
    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE, dpi=FIG_DPI)

    # 子图 1：Packing number。
    ax = axes[0]
    width = GROUPED_BAR_WIDTH
    ax.bar(x - width / 2, prefill_t, width=width, color=PACKING_P_COLOR, edgecolor=EDGE_COLOR, linewidth=BAR_EDGE_WIDTH, hatch=LEFT_HATCH, label="P", zorder=3)
    ax.bar(x + width / 2, decode_t, width=width, color=PACKING_D_COLOR, edgecolor=EDGE_COLOR, linewidth=BAR_EDGE_WIDTH, hatch=RIGHT_HATCH, label="D", zorder=3)
    for i, value in enumerate(prefill_t):
        ax.text(i - width / 2, value + PACKING_LABEL_OFFSET, f"{value:.0f}", ha="center", va="bottom", fontsize=FONT_SIZE["bar_label"])
    for i, value in enumerate(decode_t):
        ax.text(i + width / 2, value + PACKING_LABEL_OFFSET, f"{value:.0f}", ha="center", va="bottom", fontsize=FONT_SIZE["bar_label"])
    ax.set_ylabel("Packing Number Per DSP", fontsize=FONT_SIZE["axis_label"], fontweight="bold")
    ax.set_xticks(x, XLABELS, fontsize=FONT_SIZE["tick_label"], fontweight="bold")
    set_zero_based_ticks(ax, prefill_t + decode_t)
    apply_style(ax)
    legend = ax.legend(**LEGEND_CONFIG, fontsize=FONT_SIZE["legend"])
    legend.get_frame().set_edgecolor(LEGEND_EDGE_COLOR)
    legend.get_frame().set_linewidth(LEGEND_EDGE_WIDTH)

    # 子图 2：同时支持 P/D 的 LUT/FF 开销。
    ax = axes[1]
    width = GROUPED_BAR_WIDTH
    lut_plot = [0 if v is None else v for v in lut]
    ff_plot = [0 if v is None else v for v in ff]
    ax.bar(x - width / 2, lut_plot, width=width, color=RESOURCE_LUT_COLOR, edgecolor=EDGE_COLOR, linewidth=BAR_EDGE_WIDTH, hatch=LEFT_HATCH, label="LUT", zorder=3)
    ax.bar(x + width / 2, ff_plot, width=width, color=RESOURCE_FF_COLOR, edgecolor=EDGE_COLOR, linewidth=BAR_EDGE_WIDTH, hatch=RIGHT_HATCH, label="FF", zorder=3)
    for i, (lv, fv) in enumerate(zip(lut, ff)):
        if lv is not None:
            ax.text(i - width / 2, lv + RESOURCE_LABEL_OFFSET, f"{lv:.0f}", ha="center", va="bottom", fontsize=FONT_SIZE["resource_label"])
        if fv is not None:
            ax.text(i + width / 2, fv + RESOURCE_LABEL_OFFSET, f"{fv:.0f}", ha="center", va="bottom", fontsize=FONT_SIZE["resource_label"])
        else:
            ax.text(i + width / 2, RESOURCE_TBD_Y, "TBD", ha="center", va="bottom", fontsize=FONT_SIZE["resource_label"])
    ax.set_ylabel("Resource Usage per DSP", fontsize=FONT_SIZE["axis_label"], fontweight="bold")
    ax.set_xticks(x, XLABELS, fontsize=FONT_SIZE["tick_label"], fontweight="bold")
    set_zero_based_ticks(ax, [v for v in lut_plot + ff_plot if v is not None], top_padding=RESOURCE_LABEL_OFFSET + 8)
    apply_style(ax)
    legend = ax.legend(**LEGEND_CONFIG, fontsize=FONT_SIZE["legend"])
    legend.get_frame().set_edgecolor(LEGEND_EDGE_COLOR)
    legend.get_frame().set_linewidth(LEGEND_EDGE_WIDTH)

    fig.subplots_adjust(**SUBPLOT_ADJUST)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".png"), dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08, facecolor="white")
    print(f"Wrote {out_prefix.with_suffix('.png')} and {out_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
