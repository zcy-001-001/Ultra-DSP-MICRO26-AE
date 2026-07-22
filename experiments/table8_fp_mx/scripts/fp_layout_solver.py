#!/usr/bin/env python3
"""Single-DSP layout search for decoded FP4/FP8 product backends.

This file is a command-line port of the core layout routines from
``Ultra-DSP-main/ILP-Solver/pareto.ipynb``.  The names intentionally mirror the
notebook functions (`get_pos`, `overlap_check`, `solve_feasible`,
`solve_all_4`, and `find_pareto_front`) so the rebuttal artifact can be audited
against the original solver.
"""

from __future__ import annotations

import argparse
import csv
from functools import cache
from pathlib import Path
from typing import Iterable

from fp_format_defs import FPFormat, iter_formats


DSP_X_LIM = 17
DSP_Y_LIM = 26
DSP_R_LIM = 47
DEFAULT_MAX_RESULT_OVERLAP = 3


def get_pos(n: int, W: int, pn: int) -> list[int]:
    if n == 1:
        return [0]
    pos = [0] * n
    pad = pn // (n - 1) if pn else 0
    for i in range(n - 1):
        pos[i + 1] = pos[i] + pad
        if i < pn % (n - 1):
            pos[i + 1] += 1
    return pos


def _get_layout_positions(r: dict) -> tuple[list[int], list[int]]:
    if "x_pos" in r and "y_pos" in r:
        return list(r["x_pos"]), list(r["y_pos"])
    return get_pos(r["nx"], r["X"], r["xn"]), get_pos(r["ny"], r["Y"], r["yn"])


def _get_result_starts_from_layout(r: dict) -> list[int]:
    x_pos, y_pos = _get_layout_positions(r)
    return sorted(x + y for x in x_pos for y in y_pos)


def _calc_adjacent_result_overlaps(result_starts: list[int], result_width: int) -> list[int]:
    return [
        max(0, result_width - (right - left))
        for left, right in zip(result_starts[:-1], result_starts[1:])
    ]


def _get_max_pointwise_overlap(r: dict, R_lim: int) -> int:
    result_width = r["X"] + r["Y"]
    starts = _get_result_starts_from_layout(r)
    if not starts:
        return -1
    if starts[0] < 0 or starts[-1] + result_width > R_lim:
        return -1
    occupancy = [0] * R_lim
    for start in starts:
        for idx in range(start, start + result_width):
            occupancy[idx] += 1
    return max(occupancy, default=0)


def _inputs_non_overlapping(pos: list[int], width: int, limit: int) -> bool:
    if not pos:
        return False
    if pos[0] < 0 or pos[-1] + width > limit:
        return False
    return all(right - left >= width for left, right in zip(pos[:-1], pos[1:]))


def overlap_check(r: dict, R_lim: int) -> bool:
    x_pos, y_pos = _get_layout_positions(r)
    if not _inputs_non_overlapping(x_pos, r["X"], r["X_lim"]):
        return False
    if not _inputs_non_overlapping(y_pos, r["Y"], r["Y_lim"]):
        return False

    result_width = r["X"] + r["Y"]
    starts = _get_result_starts_from_layout(r)
    if not starts or starts[0] < 0 or starts[-1] + result_width > R_lim:
        return False

    max_result_overlap = int(r.get("max_result_overlap", DEFAULT_MAX_RESULT_OVERLAP))
    min_result_spacing = max(0, result_width - max_result_overlap)
    spacing_ok = all(
        right - left >= min_result_spacing for left, right in zip(starts[:-1], starts[1:])
    )
    pointwise_ok = _get_max_pointwise_overlap(r, R_lim) <= 2
    return spacing_ok and pointwise_ok


def _iter_feasible_xy(
    n1: int,
    n2: int,
    X: int,
    Y: int,
    X_lim: int,
    Y_lim: int,
    R_lim: int,
    max_result_overlap: int,
) -> Iterable[tuple[int, int]]:
    max_xn = X_lim - X
    max_yn = Y_lim - Y
    if max_xn < 0 or max_yn < 0:
        return

    for xn in range(max_xn, -1, -1):
        x_pos = get_pos(n1, X, xn)
        if not _inputs_non_overlapping(x_pos, X, X_lim):
            continue
        for yn in range(max_yn, -1, -1):
            y_pos = get_pos(n2, Y, yn)
            if not _inputs_non_overlapping(y_pos, Y, Y_lim):
                continue
            r = {
                "nx": n1,
                "ny": n2,
                "X": X,
                "Y": Y,
                "X_lim": X_lim,
                "Y_lim": Y_lim,
                "xn": xn,
                "yn": yn,
                "max_result_overlap": max_result_overlap,
            }
            if overlap_check(r, R_lim):
                yield xn, yn


def solve_feasible(
    n1: int,
    n2: int,
    X: int,
    Y: int,
    X_lim: int,
    Y_lim: int,
    R_lim: int,
    max_result_overlap: int = DEFAULT_MAX_RESULT_OVERLAP,
) -> dict | None:
    best_result = None
    for xn, yn in _iter_feasible_xy(n1, n2, X, Y, X_lim, Y_lim, R_lim, max_result_overlap):
        candidate = {
            "nx": n1,
            "ny": n2,
            "T": n1 * n2,
            "cost": float((n1 * n2 - 1) * (X + Y) - (xn + yn)),
            "X": X,
            "Y": Y,
            "X_lim": X_lim,
            "Y_lim": Y_lim,
            "R_lim": R_lim,
            "xn": int(xn),
            "yn": int(yn),
            "max_result_overlap": int(max_result_overlap),
        }
        if best_result is None or candidate["cost"] < best_result["cost"]:
            best_result = candidate
    return best_result


def solve_all(
    X: int,
    Y: int,
    X_lim: int,
    Y_lim: int,
    R_lim: int,
    label: str = "",
    max_result_overlap: int = DEFAULT_MAX_RESULT_OVERLAP,
) -> list[dict]:
    results = []
    for n1 in range(1, X_lim // X + 1):
        for n2 in range(1, Y_lim // Y + 1):
            if n1 == 1 and n2 == 1:
                continue
            best = solve_feasible(n1, n2, X, Y, X_lim, Y_lim, R_lim, max_result_overlap)
            if best is not None:
                best["label"] = label
                results.append(best)
    return results


@cache
def solve_all_4(
    X: int,
    Y: int,
    X_lim: int = DSP_X_LIM,
    Y_lim: int = DSP_Y_LIM,
    R_lim: int = DSP_R_LIM,
    max_result_overlap: int = DEFAULT_MAX_RESULT_OVERLAP,
) -> tuple[dict, ...]:
    res = []
    res += solve_all(X, Y, X_lim, Y_lim, R_lim, label="A", max_result_overlap=max_result_overlap)
    res += solve_all(Y, X, X_lim, Y_lim, R_lim, label="B", max_result_overlap=max_result_overlap)
    res += solve_all(X, Y, Y_lim, X_lim, R_lim, label="C", max_result_overlap=max_result_overlap)
    res += solve_all(Y, X, Y_lim, X_lim, R_lim, label="D", max_result_overlap=max_result_overlap)

    used_config = set()
    unique_res = []
    for r in res:
        x_cfg = (r["nx"], r["X"], r["xn"], r["X_lim"])
        y_cfg = (r["ny"], r["Y"], r["yn"], r["Y_lim"])
        if (x_cfg, y_cfg) in used_config:
            continue
        unique_res.append(r)
        used_config.add((x_cfg, y_cfg))
        used_config.add((y_cfg, x_cfg))
    return tuple(unique_res)


def find_pareto_front(solutions: list[tuple[float, float]]) -> list[bool]:
    is_pareto = [True] * len(solutions)
    for i, sol_i in enumerate(solutions):
        for j, sol_j in enumerate(solutions):
            if i == j:
                continue
            all_j_le_i = all(a <= b for a, b in zip(sol_j, sol_i))
            any_j_lt_i = any(a < b for a, b in zip(sol_j, sol_i))
            if all_j_le_i and any_j_lt_i:
                is_pareto[i] = False
                break
    return is_pareto


def annotate_layout(r: dict) -> dict:
    x_pos, y_pos = _get_layout_positions(r)
    starts = _get_result_starts_from_layout(r)
    overlaps = _calc_adjacent_result_overlaps(starts, r["X"] + r["Y"])
    out = dict(r)
    out["x_pos"] = x_pos
    out["y_pos"] = y_pos
    out["result_starts"] = starts
    out["max_adjacent_result_overlap"] = max(overlaps) if overlaps else 0
    out["max_pointwise_result_overlap"] = _get_max_pointwise_overlap(r, r["R_lim"])
    return out


def choose_best_layout(width: int, max_result_overlap: int) -> dict:
    candidates = [annotate_layout(r) for r in solve_all_4(width, width, max_result_overlap=max_result_overlap)]
    if not candidates:
        raise RuntimeError(f"No feasible layout for width={width}, max_result_overlap={max_result_overlap}")
    return min(candidates, key=lambda r: (-r["T"], r["cost"], r["nx"], r["ny"]))


def rows_for_format(fmt: FPFormat) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = [
        {
            "format": fmt.name,
            "format_label": fmt.label,
            "product_kind": fmt.product_kind,
            "product_width": fmt.product_width,
            "design": "NoPackingScalar",
            "max_result_overlap": 0,
            "nx": 1,
            "ny": 1,
            "products_per_dsp": 1,
            "cost": 0,
            "x_pos": "0",
            "y_pos": "0",
            "result_starts": "0",
            "max_adjacent_result_overlap": 0,
            "max_pointwise_result_overlap": 1,
            "valid": 1,
            "notes": "scalar one-product-per-DSP baseline",
        }
    ]

    for design, overlap in (("NormalNonOverlap", 0), ("UltraDSP-Packing", 3)):
        layout = choose_best_layout(fmt.product_width, overlap)
        rows.append(
            {
                "format": fmt.name,
                "format_label": fmt.label,
                "product_kind": fmt.product_kind,
                "product_width": fmt.product_width,
                "design": design,
                "max_result_overlap": overlap,
                "nx": layout["nx"],
                "ny": layout["ny"],
                "products_per_dsp": layout["T"],
                "cost": layout["cost"],
                "x_pos": " ".join(map(str, layout["x_pos"])),
                "y_pos": " ".join(map(str, layout["y_pos"])),
                "result_starts": " ".join(map(str, layout["result_starts"])),
                "max_adjacent_result_overlap": layout["max_adjacent_result_overlap"],
                "max_pointwise_result_overlap": layout["max_pointwise_result_overlap"],
                "valid": int(overlap_check(layout, layout["R_lim"])),
                "notes": "sanity baseline" if overlap == 0 else "correction-backed overpacking layout",
            }
        )
    return rows


def write_layout_csv(out: Path, formats: tuple[FPFormat, ...]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [row for fmt in formats for row in rows_for_format(fmt)]
    fieldnames = [
        "format",
        "format_label",
        "product_kind",
        "product_width",
        "design",
        "max_result_overlap",
        "nx",
        "ny",
        "products_per_dsp",
        "cost",
        "x_pos",
        "y_pos",
        "result_starts",
        "max_adjacent_result_overlap",
        "max_pointwise_result_overlap",
        "valid",
        "notes",
    ]
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} layout rows to {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "fp_layouts.csv")
    parser.add_argument("--formats", nargs="*", default=None)
    args = parser.parse_args()
    write_layout_csv(args.out, iter_formats(args.formats))


if __name__ == "__main__":
    main()
PACKAGE_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PACKAGE_ROOT / "results" / "table8"
