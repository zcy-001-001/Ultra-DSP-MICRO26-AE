#!/usr/bin/env python3
"""绘制 W4A4 各消融方式的 P/D layout 图。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PACKAGE_ROOT / "results" / "component_ablation"


# =========================
# 可调参数区
# =========================

ORDER = ["V0", "V1", "V2", "V3", "V4"]
LABELS = {
    "V0": "Normal Signed Packing",
    "V1": "+ Sign-Magnitude",
    "V2": "+ Overlap",
    "V3": "+ Full Correction",
    "V4": "+ ILP Layout\n+ Resource Opt.",
}
PHASES = ["P", "D"]

FIG_SIZE = (14.5, 8.2)
FIG_DPI = 300
X_LIMIT = 38
BAR_HEIGHT = 0.46
TEXT_SIZE = 8
TITLE_SIZE = 11
AXIS_LABEL_SIZE = 10

COLORS = {
    "V0": "#99d98c",
    "V1": "#76c893",
    "V2": "#52b69a",
    "V3": "#34a0a4",
    "V4": "#168aad",
}
EDGE_COLOR = "#2F2F2F"
GRID_COLOR = "#000000"


def read_layouts(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {(row["variant"], row["phase"]): row for row in rows}


def parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.split()] if text.strip() else []


def draw_phase(ax, row: dict[str, str], color: str) -> None:
    width = int(row["product_width"])
    offsets = parse_ints(row["product_offsets"])

    for lane, offset in enumerate(offsets, start=1):
        rect = Rectangle(
            (offset, 0.5 - BAR_HEIGHT / 2),
            width,
            BAR_HEIGHT,
            facecolor=color,
            edgecolor=EDGE_COLOR,
            linewidth=1.0,
            alpha=0.88,
            hatch="////" if row["layout_type"] in {"hand_tuned_overlap", "hand_tuned_corrected", "ilp_layout"} else None,
        )
        ax.add_patch(rect)
        ax.text(offset + width / 2, 0.5, str(lane), ha="center", va="center", fontsize=TEXT_SIZE)
        ax.text(offset, 0.09, str(offset), ha="center", va="center", fontsize=TEXT_SIZE - 1)

    ax.set_xlim(0, X_LIMIT)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks(range(0, X_LIMIT + 1, 4))
    ax.grid(axis="x", linestyle=(0, (4, 3)), linewidth=0.6, alpha=0.45, color=GRID_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(EDGE_COLOR)
    ax.set_xlabel("DSP P-port product bit offset", fontsize=AXIS_LABEL_SIZE)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layouts", default="data/layouts_w4a4.csv")
    parser.add_argument("--out-prefix", type=Path, default=RESULTS_DIR / "layout_diagrams_w4a4")
    args = parser.parse_args()

    layouts = read_layouts(Path(args.layouts))
    fig, axes = plt.subplots(len(ORDER), len(PHASES), figsize=FIG_SIZE, dpi=FIG_DPI)

    for row_idx, variant in enumerate(ORDER):
        for col_idx, phase in enumerate(PHASES):
            ax = axes[row_idx][col_idx]
            row = layouts[(variant, phase)]
            draw_phase(ax, row, COLORS[variant])
            title = f"{LABELS[variant]} - {phase}, packing={row['packing']}"
            ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold")
            if col_idx == 0:
                ax.text(-0.08, 0.5, variant, transform=ax.transAxes, ha="right", va="center", fontsize=TITLE_SIZE, fontweight="bold")

    fig.subplots_adjust(left=0.08, right=0.985, top=0.96, bottom=0.07, hspace=0.72, wspace=0.18)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".png"), dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    print(f"Wrote {out_prefix.with_suffix('.png')} and {out_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
