#!/usr/bin/env python3
"""Audit how different W4A4 pointwise-overlap caps affect packing."""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path

from generate_overlap_depth_sweep import (
    A_WIDTH,
    B_WIDTH,
    DEPTH_CAPS,
    MAG_BITS,
    OUT_DIR,
    POINTWISE_LIMIT_BY_DEPTH,
    PRODUCT_WIDTH,
    R_LIM,
    X_LIM,
    Y_LIM,
    Candidate,
    iter_uniform_extents_full,
    pointwise_overlap,
    product_start_multiset,
    spread_positions,
)


POINTWISE_CAPS: tuple[int | None, ...] = (2, 3, 4, None)


def fmt_ints(values: tuple[int, ...] | list[int]) -> str:
    return " ".join(str(value) for value in values)


def adjacent_max_overlap(starts: tuple[int, ...]) -> int:
    if len(starts) <= 1:
        return 0
    return max(max(0, PRODUCT_WIDTH - (right - left)) for left, right in zip(starts[:-1], starts[1:]))


def is_legal_positions_with_pointwise(
    x_pos: tuple[int, ...],
    y_pos: tuple[int, ...],
    depth: int,
    pointwise_cap: int | None,
) -> bool:
    if not x_pos or not y_pos:
        return False
    if x_pos[0] < 0 or y_pos[0] < 0:
        return False
    if any((right - left) < MAG_BITS for left, right in zip(x_pos[:-1], x_pos[1:])):
        return False
    if any((right - left) < MAG_BITS for left, right in zip(y_pos[:-1], y_pos[1:])):
        return False
    # Match generate_overlap_depth_sweep.py and the original pareto.ipynb
    # packing-bound convention; B_WIDTH/A_WIDTH are physical RTL widths.
    if x_pos[-1] + MAG_BITS > X_LIM:
        return False
    if y_pos[-1] + MAG_BITS > Y_LIM:
        return False

    starts = tuple(product_start_multiset(x_pos, y_pos))
    if not starts or starts[-1] + PRODUCT_WIDTH > R_LIM:
        return False

    min_spacing = PRODUCT_WIDTH - depth
    if any((right - left) < min_spacing for left, right in zip(starts[:-1], starts[1:])):
        return False
    if pointwise_cap is not None and pointwise_overlap(starts) > pointwise_cap:
        return False
    return True


def candidate_from_positions(
    depth: int,
    x_pos: tuple[int, ...],
    y_pos: tuple[int, ...],
    source: str,
) -> Candidate:
    nx = len(x_pos)
    ny = len(y_pos)
    cost = (nx * ny - 1) * PRODUCT_WIDTH - (x_pos[-1] + y_pos[-1])
    return Candidate(depth, nx, ny, x_pos[-1], y_pos[-1], x_pos, y_pos, cost, source=source)


def update_best(best: Candidate | None, candidate: Candidate, mode: str) -> Candidate:
    if best is None:
        return candidate
    if mode == "prefill":
        old_key = (-best.t, best.cost, best.max_adjacent_overlap, best.nx, best.ny, best.x_pos, best.y_pos)
        new_key = (-candidate.t, candidate.cost, candidate.max_adjacent_overlap, candidate.nx, candidate.ny, candidate.x_pos, candidate.y_pos)
    elif mode == "decode":
        old_key = (-best.decode_t, best.cost, best.max_adjacent_overlap, best.nx, best.ny, best.x_pos, best.y_pos)
        new_key = (-candidate.decode_t, candidate.cost, candidate.max_adjacent_overlap, candidate.nx, candidate.ny, candidate.x_pos, candidate.y_pos)
    else:
        raise ValueError(mode)
    return candidate if new_key < old_key else best


def summarize_best(candidate: Candidate | None, prefix: str) -> dict[str, object]:
    if candidate is None:
        return {
            f"{prefix}_schema": "",
            f"{prefix}_T": "",
            f"{prefix}_decode_T": "",
            f"{prefix}_x_pos": "",
            f"{prefix}_y_pos": "",
            f"{prefix}_product_offsets": "",
            f"{prefix}_cost": "",
            f"{prefix}_max_adjacent_overlap": "",
            f"{prefix}_pointwise_overlap": "",
        }
    starts = candidate.product_starts
    return {
        f"{prefix}_schema": f"{candidate.nx}x{candidate.ny}",
        f"{prefix}_T": candidate.t,
        f"{prefix}_decode_T": candidate.decode_t,
        f"{prefix}_x_pos": fmt_ints(candidate.x_pos),
        f"{prefix}_y_pos": fmt_ints(candidate.y_pos),
        f"{prefix}_product_offsets": fmt_ints(starts),
        f"{prefix}_cost": candidate.cost,
        f"{prefix}_max_adjacent_overlap": candidate.max_adjacent_overlap,
        f"{prefix}_pointwise_overlap": pointwise_overlap(starts),
    }


def pointwise_label(pointwise_cap: int | None) -> str:
    return "disabled" if pointwise_cap is None else f"<={pointwise_cap}"


def audit_uniform(depth: int, pointwise_cap: int | None) -> dict[str, object]:
    legal_count = 0
    best_prefill: Candidate | None = None
    best_decode: Candidate | None = None

    for nx in range(1, B_WIDTH // MAG_BITS + 1):
        for ny in range(1, A_WIDTH // MAG_BITS + 1):
            if nx == 1 and ny == 1:
                continue
            for x_extent, y_extent in iter_uniform_extents_full(nx, ny, depth) or []:
                x_pos = spread_positions(nx, x_extent)
                y_pos = spread_positions(ny, y_extent)
                if not is_legal_positions_with_pointwise(x_pos, y_pos, depth, pointwise_cap):
                    continue
                candidate = candidate_from_positions(depth, x_pos, y_pos, "uniform")
                legal_count += 1
                best_prefill = update_best(best_prefill, candidate, "prefill")
                best_decode = update_best(best_decode, candidate, "decode")

    note = (
        "configured generator cap"
        if pointwise_cap == POINTWISE_LIMIT_BY_DEPTH.get(depth)
        else "constraint audit"
    )
    return {
        "depth": depth,
        "search_model": "uniform",
        "pointwise_limit": pointwise_label(pointwise_cap),
        "legal_layouts": legal_count,
        **summarize_best(best_prefill, "best_prefill"),
        **summarize_best(best_decode, "best_decode"),
        "note": note,
    }


def nonoverlap_combinations(max_start: int, width: int) -> dict[int, list[tuple[int, ...]]]:
    out: dict[int, list[tuple[int, ...]]] = {}
    for count in range(1, max_start // width + 2):
        values: list[tuple[int, ...]] = []
        for combo in combinations(range(max_start + 1), count):
            if all((right - left) >= width for left, right in zip(combo[:-1], combo[1:])):
                values.append(combo)
        if values:
            out[count] = values
    return out


def pointwise_bucket_upto_four(starts: tuple[int, ...]) -> int:
    layers = [0, 0, 0, 0, 0]
    for start in starts:
        mask = ((1 << PRODUCT_WIDTH) - 1) << start
        for idx in range(4, 0, -1):
            layers[idx] |= layers[idx - 1] & mask
        layers[0] |= mask
    for idx in range(4, -1, -1):
        if layers[idx]:
            return idx + 1
    return 0


def update_exhaustive_state(
    state: dict[tuple[int, int | None], dict[str, object]],
    x_pos: tuple[int, ...],
    y_pos: tuple[int, ...],
    required_depth: int,
    pointwise_cap: int | None,
) -> None:
    for depth in DEPTH_CAPS:
        if required_depth > depth:
            continue
        candidate = candidate_from_positions(depth, x_pos, y_pos, "exhaustive")
        key = (depth, pointwise_cap)
        state[key]["legal_layouts"] = int(state[key]["legal_layouts"]) + 1
        state[key]["best_prefill"] = update_best(
            state[key]["best_prefill"], candidate, "prefill"
        )
        state[key]["best_decode"] = update_best(
            state[key]["best_decode"], candidate, "decode"
        )


def exhaustive_rows_from_state(
    state: dict[tuple[int, int | None], dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for depth in DEPTH_CAPS:
        for cap in POINTWISE_CAPS:
            key = (depth, cap)
            rows.append(
                {
                    "depth": depth,
                    "search_model": "exhaustive_nonuniform",
                    "pointwise_limit": pointwise_label(cap),
                    "legal_layouts": state[key]["legal_layouts"],
                    **summarize_best(state[key]["best_prefill"], "best_prefill"),
                    **summarize_best(state[key]["best_decode"], "best_decode"),
                    "note": "non-uniform constraint audit",
                }
            )
    return rows


def audit_exhaustive_all_caps() -> list[dict[str, object]]:
    max_x_start = X_LIM - MAG_BITS
    max_y_start = Y_LIM - MAG_BITS
    x_combos = nonoverlap_combinations(max_x_start, MAG_BITS)
    y_combos = nonoverlap_combinations(max_y_start, MAG_BITS)

    state: dict[tuple[int, int | None], dict[str, object]] = {
        (depth, cap): {"legal_layouts": 0, "best_prefill": None, "best_decode": None}
        for depth in DEPTH_CAPS
        for cap in POINTWISE_CAPS
    }

    xs_flat = [combo for combos in x_combos.values() for combo in combos]
    ys_flat = [combo for combos in y_combos.values() for combo in combos]

    for x_pos in xs_flat:
        max_allowed_y = R_LIM - PRODUCT_WIDTH - x_pos[-1]
        for y_pos in ys_flat:
            if y_pos[-1] > max_allowed_y:
                continue
            if len(x_pos) == 1 and len(y_pos) == 1:
                continue
            starts = tuple(sorted(x + y for x in x_pos for y in y_pos))
            required_depth = adjacent_max_overlap(starts)
            pointwise_bucket = pointwise_bucket_upto_four(starts)
            for cap in POINTWISE_CAPS:
                if cap is not None and pointwise_bucket > cap:
                    continue
                update_exhaustive_state(state, x_pos, y_pos, required_depth, cap)

    return exhaustive_rows_from_state(state)


def audit_exhaustive(enforce_pointwise: bool) -> list[dict[str, object]]:
    # Compatibility wrapper retained for earlier callers. The main path uses
    # audit_exhaustive_all_caps() to avoid scanning the same search space twice.
    return [
        row for row in audit_exhaustive_all_caps()
        if (row["pointwise_limit"] != "disabled") == enforce_pointwise
    ]


def write_rows(rows: list[dict[str, object]], out: Path) -> None:
    fields = [
        "depth",
        "search_model",
        "pointwise_limit",
        "legal_layouts",
        "best_prefill_schema",
        "best_prefill_T",
        "best_prefill_decode_T",
        "best_prefill_x_pos",
        "best_prefill_y_pos",
        "best_prefill_product_offsets",
        "best_prefill_cost",
        "best_prefill_max_adjacent_overlap",
        "best_prefill_pointwise_overlap",
        "best_decode_schema",
        "best_decode_T",
        "best_decode_decode_T",
        "best_decode_x_pos",
        "best_decode_y_pos",
        "best_decode_product_offsets",
        "best_decode_cost",
        "best_decode_max_adjacent_overlap",
        "best_decode_pointwise_overlap",
        "note",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT_DIR / "constraint_audit_w4a4.csv"))
    # The non-uniform exhaustive audit is useful as a separate stress check, but
    # it is too slow for the default reproduction path on this Windows machine.
    parser.add_argument("--include-exhaustive", action="store_true")
    parser.add_argument("--skip-exhaustive", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for depth in DEPTH_CAPS:
        for cap in POINTWISE_CAPS:
            rows.append(audit_uniform(depth, cap))

    if args.include_exhaustive and not args.skip_exhaustive:
        rows.extend(audit_exhaustive_all_caps())

    rows.sort(key=lambda row: (int(row["depth"]), str(row["search_model"]), str(row["pointwise_limit"])))
    write_rows(rows, Path(args.out))
    print(f"Wrote {args.out} with {len(rows)} rows")


if __name__ == "__main__":
    main()
