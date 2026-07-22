#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

TOTAL_LUT = 1_304_000
TOTAL_FF = 2_607_000
DSP_BUDGET = 4096
FREQ_MHZ = 200
ULTRA_LLAMA2_AVG = 65.96
BUDGET_RATIOS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]

METHODS = {
    "WP521": {
        "acc": None,
        "T_prefill": 4,
        "T_decode": 2,
        "lut_per_dsp": 0,
        "ff_per_dsp": 0,
        "color": "#8A9693",
    },
    "DB-MixQ": {
        "acc": 32.85 / ULTRA_LLAMA2_AVG * 100.0,
        "T_prefill": 6,
        "T_decode": 2,
        "lut_per_dsp": 44,
        "ff_per_dsp": 50,
        "color": "#C97C7C",
    },
    "DSP-Packing": {
        "acc": 31.97 / ULTRA_LLAMA2_AVG * 100.0,
        "T_prefill": 6,
        "T_decode": 2,
        "lut_per_dsp": 27,
        "ff_per_dsp": 32,
        "color": "#D79A57",
    },
    "DuoQ": {
        "acc": 100.0,
        "T_prefill": 4,
        "T_decode": 4,
        "lut_per_dsp": 21,
        "ff_per_dsp": 30,
        "color": "#77A66F",
    },
    "UDP": {
        "acc": 100.0,
        "T_prefill": 6,
        "T_decode": 3,
        "lut_per_dsp": 33,
        "ff_per_dsp": 12,
        "color": "#5C8EAD",
    },
    "Ultra-DSP": {
        "acc": 100.0,
        "T_prefill": 9,
        "T_decode": 7,
        "lut_per_dsp": 75,
        "ff_per_dsp": 67,
        "color": "#2F5368",
    },
}

LEGEND_METHOD_ORDER = ["WP521", "DSP-Packing", "DB-MixQ", "DuoQ", "UDP", "Ultra-DSP"]

LABEL_OFFSETS = {
    "WP521": (5, -13, "left"),
    "DB-MixQ": (5, 7, "left"),
    "DSP-Packing": (-8, 7, "right"),
    "DuoQ": (-8, 7, "right"),
    "UDP": (-8, 7, "right"),
    "Ultra-DSP": (-8, 7, "right"),
}

ZOOM_LOW_XLIM = (47.9, 50.15)
ZOOM_HIGH_XLIM = (99.50, 100.95)
ZOOM_LOW_TICKS = [48.0, 48.5, 49.0, 49.5, 50.0]
ZOOM_HIGH_TICKS = [100.0]
ZOOM_LABEL_OFFSETS = {
    "WP521": (0, 16, "center"),
    "DB-MixQ": (-12, 22, "right"),
    "DSP-Packing": (14, 2, "left"),
    "DuoQ": (14, -2, "left"),
    "UDP": (14, 3, "left"),
    "Ultra-DSP": (-14, 1, "right"),
}


def load_wp521_accuracy(summary_path: Path) -> float:
    rows = json.loads(summary_path.read_text())
    for row in rows:
        if row.get("model") == "llama2_7b" and row.get("method") == "WP521":
            return float(row["avg"]) / ULTRA_LLAMA2_AVG * 100.0
    raise RuntimeError(f"Missing llama2_7b WP521 row in {summary_path}")


def effective_dsp_count(dsp_budget: int, lut_budget: int, ff_budget: int, lut_per_dsp: int, ff_per_dsp: int) -> int:
    lut_limit = 10**18 if lut_per_dsp == 0 else math.floor(lut_budget / lut_per_dsp)
    ff_limit = 10**18 if ff_per_dsp == 0 else math.floor(ff_budget / ff_per_dsp)
    return min(dsp_budget, lut_limit, ff_limit)


def throughput_gops(n_eff: int, t_value: int) -> float:
    return n_eff * t_value * 2 * FREQ_MHZ / 1000.0


def build_points(wp521_acc: float) -> list[dict[str, object]]:
    methods = {name: dict(values) for name, values in METHODS.items()}
    methods["WP521"]["acc"] = wp521_acc
    points: list[dict[str, object]] = []
    for ratio in BUDGET_RATIOS:
        lut_budget = round(ratio * TOTAL_LUT)
        ff_budget = round(ratio * TOTAL_FF)
        for method, info in methods.items():
            n_eff = effective_dsp_count(
                DSP_BUDGET,
                lut_budget,
                ff_budget,
                int(info["lut_per_dsp"]),
                int(info["ff_per_dsp"]),
            )
            points.append(
                {
                    "budget_ratio": ratio,
                    "budget_lut": lut_budget,
                    "budget_ff": ff_budget,
                    "method": method,
                    "relative_accuracy": float(info["acc"]),
                    "n_eff": n_eff,
                    "prefill_gops": throughput_gops(n_eff, int(info["T_prefill"])),
                    "decode_gops": throughput_gops(n_eff, int(info["T_decode"])),
                    "is_lut_limited": int(info["lut_per_dsp"]) > 0 and n_eff < DSP_BUDGET and n_eff == lut_budget // int(info["lut_per_dsp"]),
                    "is_ff_limited": int(info["ff_per_dsp"]) > 0 and n_eff < DSP_BUDGET and n_eff == ff_budget // int(info["ff_per_dsp"]),
                }
            )
    return points


def write_points(points: list[dict[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "budget_ratio",
        "budget_lut",
        "budget_ff",
        "method",
        "relative_accuracy",
        "n_eff",
        "prefill_gops",
        "decode_gops",
        "is_lut_limited",
        "is_ff_limited",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for point in points:
            writer.writerow(point)


def plot_points_legacy(points: list[dict[str, object]], out_prefix: Path) -> None:
    # Legacy full-range plot retained for auditability. The active plot_points
    # below uses a broken x-axis because the 50%-100% blank region compressed
    # the WP521/DSP-Packing/DB-MixQ accuracy differences too much.
    by_budget: dict[float, list[dict[str, object]]] = {}
    for point in points:
        by_budget.setdefault(float(point["budget_ratio"]), []).append(point)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    for ax, ratio in zip(axes_flat, BUDGET_RATIOS):
        rows = by_budget[ratio]
        for row in rows:
            method = str(row["method"])
            color = str(METHODS[method]["color"])
            x = float(row["relative_accuracy"])
            y_prefill = float(row["prefill_gops"])
            y_decode = float(row["decode_gops"])
            ax.plot([x, x], [y_decode, y_prefill], color=color, linewidth=0.8, alpha=0.55)
            size = 74 if method == "Ultra-DSP" else 46
            edge = "black" if method == "Ultra-DSP" else "white"
            ax.scatter(x, y_prefill, marker="o", s=size, color=color, edgecolor=edge, linewidth=0.8, zorder=3)
            ax.scatter(x, y_decode, marker="^", s=size, color=color, edgecolor=edge, linewidth=0.8, zorder=3)
            # Previous figure only annotated WP521 and Ultra-DSP; keep all
            # methods explicitly labeled so the plot is self-contained even
            # without relying on the legend.
            # if method in {"Ultra-DSP", "WP521"}:
            #     ax.annotate(method, (x, y_prefill), textcoords="offset points", xytext=(4, 4), fontsize=8)
            dx, dy, ha = LABEL_OFFSETS[method]
            ax.annotate(
                method,
                (x, y_prefill),
                textcoords="offset points",
                xytext=(dx, dy),
                fontsize=7.5,
                ha=ha,
                bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.72},
            )
        ax.set_title(f"{int(ratio * 100)}% LUT/FF Budget", fontsize=11)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
        ax.set_xlim(44, 106)
        ax.set_ylim(0, 16000)

    for ax in axes[:, 0]:
        ax.set_ylabel("Throughput (GOPS)")
    for ax in axes[-1, :]:
        ax.set_xlabel("Relative Accuracy to W4A4 OSTQuant / Ultra-DSP (%)")

    method_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=str(info["color"]), label=name, markersize=7)
        for name, info in METHODS.items()
    ]
    point_handles = [
        plt.Line2D([0], [0], marker="o", color="black", label="Prefill", linestyle="None", markersize=7),
        plt.Line2D([0], [0], marker="^", color="black", label="Decode", linestyle="None", markersize=7),
    ]
    fig.legend(handles=method_handles + point_handles, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Accuracy-Throughput Pareto under Fixed 4096-DSP Budget (45.4% of U55C DSPs)", fontsize=13)
    fig.tight_layout(rect=(0, 0.11, 1, 0.94))
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".png"), dpi=240)
    fig.savefig(out_prefix.with_suffix(".pdf"))
    plt.close(fig)


def _choose_accuracy_axis(x: float, ax_low: plt.Axes, ax_high: plt.Axes) -> plt.Axes:
    return ax_low if x < 75.0 else ax_high


def _draw_axis_break(ax_low: plt.Axes, ax_high: plt.Axes) -> None:
    ax_low.spines["right"].set_visible(False)
    ax_high.spines["left"].set_visible(False)
    ax_high.yaxis.set_visible(False)
    tick_length = 0.012
    break_style = {"color": "0.25", "clip_on": False, "linewidth": 1.1}
    ax_low.plot((1 - tick_length, 1 + tick_length), (-tick_length, tick_length), transform=ax_low.transAxes, **break_style)
    ax_low.plot((1 - tick_length, 1 + tick_length), (1 - tick_length, 1 + tick_length), transform=ax_low.transAxes, **break_style)
    ax_high.plot((-tick_length, tick_length), (-tick_length, tick_length), transform=ax_high.transAxes, **break_style)
    ax_high.plot((-tick_length, tick_length), (1 - tick_length, 1 + tick_length), transform=ax_high.transAxes, **break_style)


def _style_zoom_axes(ax_low: plt.Axes, ax_high: plt.Axes) -> None:
    for ax in (ax_low, ax_high):
        ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.42)
        # Plot throughput in TOPS while keeping the CSV fields in GOPS for
        # traceability. Earlier versions plotted raw GOPS values here.
        # ax.set_ylim(0, 16500)
        # ax.set_yticks(range(0, 16001, 4000))
        ax.set_ylim(0, 16.5)
        ax.set_yticks([0, 4, 8, 12, 16])
        ax.tick_params(axis="both", labelsize=20)
    ax_low.set_xlim(*ZOOM_LOW_XLIM)
    ax_high.set_xlim(*ZOOM_HIGH_XLIM)
    ax_low.set_xticks(ZOOM_LOW_TICKS)
    ax_high.set_xticks(ZOOM_HIGH_TICKS)
    ax_low.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax_high.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    _draw_axis_break(ax_low, ax_high)


def plot_points(points: list[dict[str, object]], out_prefix: Path) -> None:
    by_budget: dict[float, list[dict[str, object]]] = {}
    for point in points:
        by_budget.setdefault(float(point["budget_ratio"]), []).append(point)

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 23,
            "axes.titlesize": 23,
            "legend.fontsize": 24,
            "xtick.labelsize": 18,
            "ytick.labelsize": 20,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(23.5, 11.7))
    outer = fig.add_gridspec(2, 3, left=0.070, right=0.997, bottom=0.192, top=0.835, wspace=0.08, hspace=0.45)
    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    for idx, ratio in enumerate(BUDGET_RATIOS):
        row_idx, col_idx = divmod(idx, 3)
        inner = outer[row_idx, col_idx].subgridspec(1, 2, width_ratios=[1.9, 0.95], wspace=0.035)
        ax_low = fig.add_subplot(inner[0, 0])
        ax_high = fig.add_subplot(inner[0, 1], sharey=ax_low)
        _style_zoom_axes(ax_low, ax_high)

        if col_idx == 0:
            # Convert the plotted unit from GOPS to TOPS.
            # ax_low.set_ylabel("Throughput (GOPS)", fontsize=23)
            ax_low.set_ylabel("Throughput (TOPS)", fontsize=24)
        else:
            ax_low.tick_params(axis="y", labelleft=False)

        # Earlier drafts used ax_low.set_title(..., loc="left"), but a split
        # x-axis makes that look visually off-center. It was then moved above
        # each panel; the active version places the compact caption below.
        # ax_low.set_title(f"{panel_labels[idx]} {int(ratio * 100)}% LUT/FF", loc="left", pad=8, fontweight="semibold")
        low_box = ax_low.get_position()
        high_box = ax_high.get_position()
        title_x = (low_box.x0 + high_box.x1) / 2
        # title_y = max(low_box.y1, high_box.y1) + 0.009
        # fig.text(title_x, title_y, f"{panel_labels[idx]} {int(ratio * 100)}% LUT/FF", ha="center", va="bottom", fontsize=16, fontweight="semibold")
        if row_idx == 0:
            panel_caption_y = low_box.y0 - 0.058
            fig.text(title_x, panel_caption_y, f"{panel_labels[idx]} {int(ratio * 100)}% LUT/FF Budget", ha="center", va="top", fontsize=22, fontweight="semibold")
        if row_idx == 1:
            # The global x-label was removed to clean up the figure; add a
            # compact per-panel label for the bottom row so the x-axis remains
            # explicit without adding a long caption-like note.
            # Previous bottom-row captions sat above the x-axis label. Move
            # them below the x-axis label so the reading order is clearer.
            fig.text(title_x, low_box.y0 - 0.054, "Rel. GEOMEAN Acc. (% of OSTQuant)", ha="center", va="top", fontsize=19)
            fig.text(title_x, low_box.y0 - 0.104, f"{panel_labels[idx]} {int(ratio * 100)}% LUT/FF Budget", ha="center", va="top", fontsize=22, fontweight="semibold")
        rows = by_budget[ratio]
        for row in rows:
            method = str(row["method"])
            color = str(METHODS[method]["color"])
            x = float(row["relative_accuracy"])
            # CSV values are stored in GOPS; divide by 1000 for TOPS display.
            # y_prefill = float(row["prefill_gops"])
            # y_decode = float(row["decode_gops"])
            y_prefill = float(row["prefill_gops"]) / 1000.0
            y_decode = float(row["decode_gops"]) / 1000.0
            ax = _choose_accuracy_axis(x, ax_low, ax_high)
            ax.plot([x, x], [y_decode, y_prefill], color=color, linewidth=1.05, alpha=0.48, zorder=2)
            # Earlier drafts used a larger marker for Ultra-DSP. Use the same
            # enlarged marker size for every method to avoid visual bias.
            # size = 104 if method == "Ultra-DSP" else 72
            # edge = "black" if method == "Ultra-DSP" else "white"
            size = 118
            edge = "#2F3437"
            ax.scatter(x, y_prefill, marker="o", s=size, color=color, edgecolor=edge, linewidth=0.75, zorder=4)
            ax.scatter(x, y_decode, marker="^", s=size, color=color, edgecolor=edge, linewidth=0.75, zorder=4)
            dx, dy, ha = ZOOM_LABEL_OFFSETS[method]
            ax.annotate(
                method,
                (x, y_prefill),
                textcoords="offset points",
                xytext=(dx, dy),
                fontsize=22.0,
                ha=ha,
                va="center",
                annotation_clip=False,
                # Earlier versions used small white rounded boxes behind
                # method labels; remove them for a cleaner paper figure.
                # bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "0.85", "linewidth": 0.4, "alpha": 0.86},
            )

    method_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=str(METHODS[name]["color"]), markeredgecolor="#2F3437", label=name, markersize=20)
        for name in LEGEND_METHOD_ORDER
    ]
    point_handles = [
        # Prefilling/Decoding are marker-shape semantics, so keep these legend
        # handles neutral instead of reusing any method color.
        # plt.Line2D([0], [0], marker="o", color="black", label="Prefill", linestyle="None", markersize=8),
        # plt.Line2D([0], [0], marker="^", color="black", label="Decode", linestyle="None", markersize=8),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#2F3437", label="Prefilling", linestyle="None", markersize=20, markeredgewidth=2.0),
        plt.Line2D([0], [0], marker="^", color="none", markerfacecolor="white", markeredgecolor="#2F3437", label="Decoding", linestyle="None", markersize=20, markeredgewidth=2.0),
    ]
    fig.legend(
        handles=method_handles + point_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.952),
        ncol=8,
        # The earlier expanded legend stretched columns too far apart. Keep it
        # compact and use larger handles/text so it reads clearly at figure scale.
        # mode="expand",
        frameon=False,
        columnspacing=0.55,
        handletextpad=0.24,
        borderaxespad=0.0,
    )
    # The bottom relative-accuracy note was removed per the latest figure
    # cleanup request; x-axis tick values still communicate the metric.
    # fig.supxlabel("Relative Accuracy to W4A4 OSTQuant / Ultra-DSP (%)  |  x-axis break omits 50.15%-99.55%", fontsize=15, y=0.055)
    # Earlier drafts used a separate footnote here, but it collided with the
    # enlarged x-axis label; the break note is now folded into the xlabel.
    # fig.text(0.5, 0.077, "The 50.15%-99.55% blank accuracy interval is omitted to enlarge WP521/DSP-Packing/DB-MixQ differences.", ha="center", fontsize=10.5, color="0.25")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".png"), dpi=260)
    fig.savefig(out_prefix.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table6-summary", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    wp521_acc = load_wp521_accuracy(args.table6_summary)
    points = build_points(wp521_acc)
    write_points(points, args.out_dir / "pareto_budget_points.csv")
    plot_points(points, args.out_dir / "pareto_budget_2x3")
    print(f"WP521 relative accuracy: {wp521_acc:.2f}%")
    print(f"Wrote {args.out_dir / 'pareto_budget_points.csv'}")
    print(f"Wrote {args.out_dir / 'pareto_budget_2x3.png'} and .pdf")


if __name__ == "__main__":
    main()
PACKAGE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = PACKAGE_ROOT / "results" / "figure12"
