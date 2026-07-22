#!/usr/bin/env python3
"""Plot W4A4 overlap-depth selected-layout and Vivado resource summaries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT.parents[1]
OUT_DIR = PACKAGE_ROOT / "results" / "overlap_depth_sweep"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_int(value: str, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    return int(float(value))


def to_float(value: str, default: float | None = None) -> float | None:
    if value is None or value == "" or value.upper() == "NA":
        return default
    return float(value)


def build_layout_stats(all_layouts: list[dict[str, str]]) -> dict[int, dict[str, int]]:
    stats: dict[int, dict[str, int]] = {}
    for row in all_layouts:
        depth = int(row["depth"])
        current = stats.setdefault(
            depth,
            {
                "legal_layout_count": 0,
                "max_legal_T": 0,
                "max_prefill_pareto_T": 0,
                "max_decode_pareto_T": 0,
            },
        )
        current["legal_layout_count"] += 1
        current["max_legal_T"] = max(current["max_legal_T"], int(row["T"]))
        if row.get("is_prefill_pareto") == "1":
            current["max_prefill_pareto_T"] = max(current["max_prefill_pareto_T"], int(row["T"]))
        if row.get("is_decode_pareto") == "1":
            current["max_decode_pareto_T"] = max(current["max_decode_pareto_T"], int(row["decode_T"]))
    return stats


def build_summary(
    selected: list[dict[str, str]],
    resource_rows: list[dict[str, str]],
    all_layouts: list[dict[str, str]],
) -> list[dict[str, object]]:
    resources = {row["top_module"]: row for row in resource_rows}
    layout_stats = build_layout_stats(all_layouts)
    rows: list[dict[str, object]] = []
    for row in selected:
        resource = resources.get(row["top_module"], {})
        stats = layout_stats.get(int(row["depth"]), {})
        rows.append(
            {
                "depth": int(row["depth"]),
                "top_module": row["top_module"],
                "prefill_schema": row["prefill_schema"],
                "decode_schema": row["decode_schema"],
                "prefill_T": int(row["prefill_T"]),
                "decode_T": int(row["decode_T"]),
                "pointwise_limit": int(row.get("pointwise_limit", "2") or 2),
                "legal_layout_count": stats.get("legal_layout_count"),
                "max_legal_T": stats.get("max_legal_T"),
                "max_prefill_pareto_T": stats.get("max_prefill_pareto_T"),
                "max_decode_pareto_T": stats.get("max_decode_pareto_T"),
                "prefill_actual_max_overlap": int(row["prefill_actual_max_overlap"]),
                "decode_actual_max_overlap": int(row["decode_actual_max_overlap"]),
                "prefill_pointwise_overlap": int(row.get("prefill_pointwise_overlap", "2") or 2),
                "decode_pointwise_overlap": int(row.get("decode_pointwise_overlap", "2") or 2),
                "prefill_total_overlap": int(row["prefill_total_overlap"]),
                "decode_total_overlap": int(row["decode_total_overlap"]),
                "clb_luts": to_int(resource.get("clb_luts", "")),
                "lut_as_logic": to_int(resource.get("lut_as_logic", "")),
                "ff": to_int(resource.get("ff", resource.get("clb_registers", ""))),
                "clb_registers": to_int(resource.get("clb_registers", "")),
                "dsp": to_int(resource.get("dsp", "")),
                "wns_ns": to_float(resource.get("wns_ns", "")),
                "timing_status": resource.get("timing_status", ""),
                "note": row["note"],
            }
        )
    return rows


def write_summary(rows: list[dict[str, object]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "depth",
        "top_module",
        "prefill_schema",
        "decode_schema",
        "prefill_T",
        "decode_T",
        "pointwise_limit",
        "legal_layout_count",
        "max_legal_T",
        "max_prefill_pareto_T",
        "max_decode_pareto_T",
        "prefill_actual_max_overlap",
        "decode_actual_max_overlap",
        "prefill_pointwise_overlap",
        "decode_pointwise_overlap",
        "prefill_total_overlap",
        "decode_total_overlap",
        "clb_luts",
        "lut_as_logic",
        "ff",
        "clb_registers",
        "dsp",
        "wns_ns",
        "timing_status",
        "note",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, object]], out_prefix: Path) -> None:
    depths = [int(row["depth"]) for row in rows]
    prefill_t = [int(row["prefill_T"]) for row in rows]
    decode_t = [int(row["decode_T"]) for row in rows]
    max_legal_t = [int(row["max_legal_T"] or 0) for row in rows]
    max_overlap = [
        max(int(row["prefill_actual_max_overlap"]), int(row["decode_actual_max_overlap"]))
        for row in rows
    ]
    pointwise = [
        max(int(row["prefill_pointwise_overlap"]), int(row["decode_pointwise_overlap"]))
        for row in rows
    ]
    luts = [row["clb_luts"] if row["clb_luts"] is not None else 0 for row in rows]
    regs = [row["clb_registers"] if row["clb_registers"] is not None else 0 for row in rows]
    wns = [row["wns_ns"] if row["wns_ns"] is not None else 0.0 for row in rows]
    has_resource = any(row["clb_luts"] is not None for row in rows)
    has_wns = any(row["wns_ns"] is not None for row in rows)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), dpi=220)

    x = range(len(depths))
    width = 0.34
    axes[0].bar([i - width / 2 for i in x], prefill_t, width=width, label="Prefill T")
    axes[0].bar([i + width / 2 for i in x], decode_t, width=width, label="Decode T")
    axes[0].plot(list(x), max_legal_t, color="#1d8f5a", marker="s", linewidth=1.4, label="Max uniform legal T")
    axes[0].plot(list(x), max_overlap, color="black", marker="o", linewidth=1.4, label="Actual max overlap")
    axes[0].plot(list(x), pointwise, color="#a14d9a", marker="^", linewidth=1.4, label="Pointwise overlap")
    axes[0].set_xticks(list(x), [str(d) for d in depths])
    axes[0].set_xlabel("Allowed overlap cap")
    axes[0].set_ylabel("Packed mults / DSP")
    axes[0].set_title("Layout Benefit")
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)
    axes[0].legend(fontsize=8)

    axes[1].bar([i - width / 2 for i in x], luts, width=width, label="CLB LUTs")
    axes[1].bar([i + width / 2 for i in x], regs, width=width, label="CLB Registers")
    axes[1].set_xticks(list(x), [str(d) for d in depths])
    axes[1].set_xlabel("Allowed overlap cap")
    axes[1].set_title("Vivado Synthesis Resource")
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)
    axes[1].legend(fontsize=8)
    if not has_resource:
        axes[1].text(0.5, 0.5, "Vivado reports not found", transform=axes[1].transAxes, ha="center")

    axes[2].bar(list(x), wns, color="#6f8fcf")
    axes[2].axhline(0, color="black", linewidth=1)
    axes[2].set_xticks(list(x), [str(d) for d in depths])
    axes[2].set_xlabel("Allowed overlap cap")
    axes[2].set_title("Timing WNS")
    axes[2].grid(axis="y", linestyle="--", alpha=0.35)
    if not has_wns:
        axes[2].text(0.5, 0.5, "No timing constraints (WNS=NA)", transform=axes[2].transAxes, ha="center")

    fig.suptitle("W4A4 Overlap-Depth Sweep", y=1.04, fontsize=14)
    fig.tight_layout()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", default=str(OUT_DIR / "layouts_w4a4_selected.csv"))
    parser.add_argument("--all-layouts", default=str(OUT_DIR / "layouts_w4a4_all.csv"))
    parser.add_argument("--resource", default=str(OUT_DIR / "vivado_resource_summary.csv"))
    parser.add_argument("--summary-out", default=str(OUT_DIR / "overlap_depth_summary.csv"))
    parser.add_argument("--figure-out", default=str(OUT_DIR / "overlap_depth_sweep"))
    args = parser.parse_args()

    selected = read_csv(Path(args.selected))
    if not selected:
        raise FileNotFoundError(f"No selected-layout CSV rows found at {args.selected}")
    all_layouts = read_csv(Path(args.all_layouts))
    resources = read_csv(Path(args.resource))
    rows = build_summary(selected, resources, all_layouts)
    write_summary(rows, Path(args.summary_out))
    plot(rows, Path(args.figure_out))
    print(f"Wrote {args.summary_out}")
    print(f"Wrote {Path(args.figure_out).with_suffix('.png')}")
    print(f"Wrote {Path(args.figure_out).with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
