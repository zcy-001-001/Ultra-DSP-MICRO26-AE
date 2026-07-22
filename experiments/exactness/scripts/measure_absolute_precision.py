#!/usr/bin/env python3
"""Product-level absolute-precision sweep for Ultra-DSP rebuttal.

The measurement follows the same random-input protocol as
component_ablation/scripts/measure_accuracy.py: signed symmetric operands,
20 groups, 10,000 samples per group, and seed 1305 by default.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS = PACKAGE_ROOT / "results" / "exactness"


DSP_A_WIDTH = 27
DSP_B_WIDTH = 18
DSP_P_WIDTH = 48
DELTA_MAX = 3


@dataclass(frozen=True)
class Layout:
    method: str
    precision: str
    w_bits: int
    a_bits: int
    nx: int
    ny: int
    x_offsets: tuple[int, ...]
    y_offsets: tuple[int, ...]
    x_kind: str
    overhead: int

    @property
    def lanes(self) -> int:
        return self.nx * self.ny


def signed_values(bits: int) -> np.ndarray:
    """Use the symmetric signed range from the existing W4A4 script.

    For INT4 this is [-7, 7], intentionally excluding -8.
    """
    limit = (1 << (bits - 1)) - 1
    return np.arange(-limit, limit + 1, dtype=np.int32)


def mask(width: int) -> int:
    return (1 << width) - 1


def decode_twos(values: np.ndarray, width: int) -> np.ndarray:
    sign = 1 << (width - 1)
    values = values.astype(np.int64) & mask(width)
    return ((values ^ sign) - sign).astype(np.int64)


def construct_offsets(last: int, count: int) -> tuple[int, ...]:
    if count == 1:
        return (0,)
    base = last // (count - 1)
    rem = last % (count - 1)
    return tuple((i * base) + max(0, i - count + 1 + rem) for i in range(count))


def solve_orientation(
    precision: str,
    w_bits: int,
    a_bits: int,
    x_mag_bits: int,
    y_mag_bits: int,
    x_kind: str,
) -> Layout | None:
    wr = x_mag_bits + y_mag_bits
    best: Layout | None = None
    max_nx = DSP_A_WIDTH // x_mag_bits
    max_ny = DSP_B_WIDTH // y_mag_bits

    for nx in range(1, max_nx + 1):
        for ny in range(1, max_ny + 1):
            x_min = (nx - 1) * x_mag_bits
            x_max = DSP_A_WIDTH - x_mag_bits
            y_min = (ny - 1) * y_mag_bits
            y_max = DSP_B_WIDTH - y_mag_bits
            if x_min > x_max or y_min > y_max:
                continue

            for x_last in range(x_min, x_max + 1):
                for y_last in range(y_min, y_max + 1):
                    if x_last + y_last + wr > DSP_P_WIDTH:
                        continue
                    if nx > 1 and (nx - 1) * wr - x_last > DELTA_MAX * (nx - 1):
                        continue
                    if ny > 1 and (ny - 1) * (x_last + wr) - y_last > DELTA_MAX * (ny - 1):
                        continue

                    overhead = (nx * ny - 1) * wr - (x_last + y_last)
                    candidate = Layout(
                        method="Ultra-DSP",
                        precision=precision,
                        w_bits=w_bits,
                        a_bits=a_bits,
                        nx=nx,
                        ny=ny,
                        x_offsets=construct_offsets(x_last, nx),
                        y_offsets=construct_offsets(y_last, ny),
                        x_kind=x_kind,
                        overhead=overhead,
                    )
                    if best is None:
                        best = candidate
                        continue
                    if candidate.lanes > best.lanes:
                        best = candidate
                    elif candidate.lanes == best.lanes and candidate.overhead < best.overhead:
                        best = candidate
    return best


def solve_ultradsp_layout(w_bits: int, a_bits: int) -> Layout:
    precision = f"W{w_bits}A{a_bits}"
    w_mag_bits = w_bits - 1
    a_mag_bits = a_bits - 1
    candidates = [
        solve_orientation(precision, w_bits, a_bits, w_mag_bits, a_mag_bits, "weight"),
        solve_orientation(precision, w_bits, a_bits, a_mag_bits, w_mag_bits, "activation"),
    ]
    valid = [candidate for candidate in candidates if candidate is not None]
    if not valid:
        raise RuntimeError(f"No legal Ultra-DSP layout found for {precision}")
    return max(valid, key=lambda c: (c.lanes, -c.overhead))


def sorted_lane_offsets(layout: Layout) -> list[tuple[int, int, int]]:
    lanes: list[tuple[int, int, int]] = []
    for ix, x in enumerate(layout.x_offsets):
        for iy, y in enumerate(layout.y_offsets):
            lanes.append((x + y, ix, iy))
    lanes.sort(key=lambda item: item[0])
    return lanes


def sample_layout_operands(
    rng: np.random.Generator, layout: Layout, samples: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    if layout.x_kind == "weight":
        w = rng.choice(signed_values(layout.w_bits), size=(samples, layout.nx), replace=True)
        a = rng.choice(signed_values(layout.a_bits), size=(samples, layout.ny), replace=True)
        lane_info = sorted_lane_offsets(layout)
        refs = []
        mags = []
        signs = []
        offsets = []
        for offset, ix, iy in lane_info:
            prod = w[:, ix].astype(np.int64) * a[:, iy].astype(np.int64)
            refs.append(prod[:, None])
            mags.append(np.abs(prod)[:, None])
            signs.append((prod < 0)[:, None])
            offsets.append(offset)
    else:
        a = rng.choice(signed_values(layout.a_bits), size=(samples, layout.nx), replace=True)
        w = rng.choice(signed_values(layout.w_bits), size=(samples, layout.ny), replace=True)
        lane_info = sorted_lane_offsets(layout)
        refs = []
        mags = []
        signs = []
        offsets = []
        for offset, ix, iy in lane_info:
            prod = w[:, iy].astype(np.int64) * a[:, ix].astype(np.int64)
            refs.append(prod[:, None])
            mags.append(np.abs(prod)[:, None])
            signs.append((prod < 0)[:, None])
            offsets.append(offset)

    return (
        np.concatenate(refs, axis=1),
        np.concatenate(mags, axis=1),
        np.concatenate(signs, axis=1),
        np.array(offsets, dtype=np.int64),
        offsets,
    )


def ultradsp_recover_magnitudes(magnitudes: np.ndarray, offsets: np.ndarray, mag_width: int) -> np.ndarray:
    """Model Ultra-DSP boundary correction plus LSB concatenation.

    The packed word first contains all overlapped magnitude products. For each
    adjacent overlapped pair, the higher product's intrusive low bits are
    subtracted at the shared boundary. The higher product then restores those
    true low bits through the final LSB-concat path.
    """
    samples, lanes = magnitudes.shape
    packed = np.zeros(samples, dtype=np.int64)
    for i in range(lanes):
        packed += magnitudes[:, i].astype(np.int64) << int(offsets[i])

    correction = np.zeros(samples, dtype=np.int64)
    prev_overlap = np.zeros(lanes, dtype=np.int64)
    for i in range(lanes - 1):
        overlap = max(0, int(offsets[i]) + mag_width - int(offsets[i + 1]))
        if overlap > 0:
            low = magnitudes[:, i + 1].astype(np.int64) & mask(overlap)
            correction += low << int(offsets[i + 1])
            prev_overlap[i + 1] = overlap

    corrected = packed - correction
    recovered = []
    for i in range(lanes):
        lsb_width = int(prev_overlap[i])
        high_width = mag_width - lsb_width
        high = (corrected >> (int(offsets[i]) + lsb_width)) & mask(high_width)
        if lsb_width:
            low = magnitudes[:, i].astype(np.int64) & mask(lsb_width)
            value = (high << lsb_width) | low
        else:
            value = high
        recovered.append(value[:, None])
    return np.concatenate(recovered, axis=1)


def apply_sign(magnitudes: np.ndarray, signs: np.ndarray) -> np.ndarray:
    return np.where(signs, -magnitudes, magnitudes).astype(np.int64)


def dsp_packing_msb_only(refs: np.ndarray) -> np.ndarray:
    """W4A4, six lanes, overlap=-2, MSB-only repair.

    This intentionally omits LSB restoration and carry-error correction.
    """
    width = 8
    step = 6
    offsets = [i * step for i in range(6)]
    packed = np.zeros(refs.shape[0], dtype=np.int64)
    encoded = refs.astype(np.int64) & mask(width)
    for i, offset in enumerate(offsets):
        packed += encoded[:, i] << offset

    outs = []
    for i, offset in enumerate(offsets):
        raw = (packed >> offset) & mask(width)
        if i < len(offsets) - 1:
            repaired_msb = (((raw >> (width - 2)) - (encoded[:, i + 1] & 0b11)) & 0b11)
            raw = (repaired_msb << (width - 2)) | (raw & mask(width - 2))
        outs.append(decode_twos(raw, width)[:, None])
    return np.concatenate(outs, axis=1)


def deepburning_no_carry(refs: np.ndarray) -> np.ndarray:
    """Paper-literal W4A4 DeepBurning-style one-bit overlap correction.

    The model follows Ultra-DSP-main/DSP_Packaging_Verilog/deepburning.v and
    intentionally does not add carry-specific compensation.
    """
    width = 8
    step = 7
    offsets = [i * step for i in range(6)]
    encoded = refs.astype(np.int64) & mask(width)
    packed = np.zeros(refs.shape[0], dtype=np.int64)
    for i, offset in enumerate(offsets):
        packed += encoded[:, i] << offset

    outs = []
    for i, offset in enumerate(offsets):
        raw = (packed >> offset) & mask(width)
        corr = raw.copy()
        if i < len(offsets) - 1:
            corr = (corr + ((encoded[:, i + 1] & 1) << (width - 1))) & mask(width)
        if i > 0:
            overlap_bit = (packed >> offset) & 1
            msb_fix = overlap_bit ^ (encoded[:, i] & 1)
            corr = (corr + msb_fix) & mask(width)
        outs.append(decode_twos(corr, width)[:, None])
    return np.concatenate(outs, axis=1)


def summarize(diff: np.ndarray) -> dict[str, float | int]:
    total = int(diff.size)
    abs_diff = np.abs(diff.astype(np.int64))
    errors = int(np.count_nonzero(diff))
    return {
        "outputs": total,
        "error_count": errors,
        "ep": float(errors / total) if total else 0.0,
        "mae": float(np.mean(abs_diff)) if total else 0.0,
        "mse": float(np.mean(diff.astype(np.int64) ** 2)) if total else 0.0,
        "max_abs_err": int(np.max(abs_diff)) if total else 0,
    }


def stable_seed(base_seed: int, group: int, w_bits: int, a_bits: int, method_id: int) -> int:
    return base_seed + group * 100_000 + w_bits * 1_000 + a_bits * 100 + method_id


def measure_ultradsp_group(layout: Layout, samples: int, seed: int) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    refs, mags, signs, offsets, _ = sample_layout_operands(rng, layout, samples)
    recovered_mag = ultradsp_recover_magnitudes(mags, offsets, layout.w_bits + layout.a_bits - 2)
    got = apply_sign(recovered_mag, signs)
    return summarize(got - refs)


def sample_w4a4_six_lanes(samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = rng.choice(signed_values(4), size=(samples, 2), replace=True)
    a = rng.choice(signed_values(4), size=(samples, 3), replace=True)
    refs = []
    for wi in range(2):
        for ai in range(3):
            refs.append((w[:, wi].astype(np.int64) * a[:, ai].astype(np.int64))[:, None])
    return np.concatenate(refs, axis=1)


def measure_baseline_group(method: str, samples: int, seed: int) -> dict[str, float | int]:
    refs = sample_w4a4_six_lanes(samples, seed)
    if method == "DSP-Packing":
        got = dsp_packing_msb_only(refs)
    elif method == "DeepBurning-MixQ":
        got = deepburning_no_carry(refs)
    else:
        raise ValueError(method)
    return summarize(got - refs)


def aggregate_group_metrics(metrics: list[dict[str, float | int]]) -> dict[str, float | int]:
    outputs = sum(int(m["outputs"]) for m in metrics)
    errors = sum(int(m["error_count"]) for m in metrics)
    weighted_mae = sum(float(m["mae"]) * int(m["outputs"]) for m in metrics) / outputs
    weighted_mse = sum(float(m["mse"]) * int(m["outputs"]) for m in metrics) / outputs
    return {
        "outputs": outputs,
        "error_count": errors,
        "ep": errors / outputs,
        "mae": weighted_mae,
        "mse": weighted_mse,
        "max_abs_err": max(int(m["max_abs_err"]) for m in metrics),
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt_metric(value: object) -> str:
    if value is None:
        return "N/A"
    number = float(value)
    if number == 0:
        return "0"
    return f"{number:.6g}"


def write_5x2_markdown(path: Path, summary_rows: list[dict[str, object]]) -> None:
    by_precision: dict[str, dict[str, dict[str, object]]] = {}
    for row in summary_rows:
        by_precision.setdefault(str(row["precision"]), {})[str(row["method"])] = row

    lines = [
        "# Absolute Precision 5x2 Table",
        "",
        "| Precision | Ultra-DSP EP | Ultra-DSP MAE | Ultra-DSP MSE | DSP-Packing EP | DSP-Packing MAE | DSP-Packing MSE | DeepBurning EP | DeepBurning MAE | DeepBurning MSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for w_bits in range(2, 9):
        for a_bits in range(2, 9):
            precision = f"W{w_bits}A{a_bits}"
            methods = by_precision.get(precision, {})
            ultra = methods.get("Ultra-DSP")
            dsp = methods.get("DSP-Packing")
            db = methods.get("DeepBurning-MixQ")
            lines.append(
                "| "
                + " | ".join(
                    [
                        precision,
                        fmt_metric(ultra["ep"] if ultra else None),
                        fmt_metric(ultra["mae"] if ultra else None),
                        fmt_metric(ultra["mse"] if ultra else None),
                        fmt_metric(dsp["ep"] if dsp else None),
                        fmt_metric(dsp["mae"] if dsp else None),
                        fmt_metric(dsp["mse"] if dsp else None),
                        fmt_metric(db["ep"] if db else None),
                        fmt_metric(db["mae"] if db else None),
                        fmt_metric(db["mse"] if db else None),
                    ]
                )
                + " |"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=int, default=20)
    parser.add_argument("--samples-per-group", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1305)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    summary_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []

    layouts = [solve_ultradsp_layout(w_bits, a_bits) for w_bits in range(2, 9) for a_bits in range(2, 9)]
    for layout in layouts:
        group_metrics = []
        _, _, _, _, offsets = sample_layout_operands(
            np.random.default_rng(stable_seed(args.seed, 0, layout.w_bits, layout.a_bits, 0)),
            layout,
            1,
        )
        for group in range(args.groups):
            metrics = measure_ultradsp_group(
                layout,
                args.samples_per_group,
                stable_seed(args.seed, group, layout.w_bits, layout.a_bits, 0),
            )
            group_metrics.append(metrics)
            group_rows.append(
                {
                    "group": group,
                    "method": layout.method,
                    "precision": layout.precision,
                    "w_bits": layout.w_bits,
                    "a_bits": layout.a_bits,
                    "lanes": layout.lanes,
                    "nx": layout.nx,
                    "ny": layout.ny,
                    "x_kind": layout.x_kind,
                    "offsets": " ".join(map(str, offsets)),
                    **metrics,
                }
            )
        aggregate = aggregate_group_metrics(group_metrics)
        summary_rows.append(
            {
                "method": layout.method,
                "precision": layout.precision,
                "w_bits": layout.w_bits,
                "a_bits": layout.a_bits,
                "lanes": layout.lanes,
                "nx": layout.nx,
                "ny": layout.ny,
                "x_kind": layout.x_kind,
                "offsets": " ".join(map(str, offsets)),
                "groups": args.groups,
                "samples_per_group": args.samples_per_group,
                **aggregate,
            }
        )

    for method_id, method in enumerate(["DSP-Packing", "DeepBurning-MixQ"], start=1):
        group_metrics = []
        for group in range(args.groups):
            metrics = measure_baseline_group(
                method,
                args.samples_per_group,
                stable_seed(args.seed, group, 4, 4, method_id),
            )
            group_metrics.append(metrics)
            group_rows.append(
                {
                    "group": group,
                    "method": method,
                    "precision": "W4A4",
                    "w_bits": 4,
                    "a_bits": 4,
                    "lanes": 6,
                    "nx": 2,
                    "ny": 3,
                    "x_kind": "paper_baseline",
                    "offsets": "0 6 12 18 24 30" if method == "DSP-Packing" else "0 7 14 21 28 35",
                    **metrics,
                }
            )
        aggregate = aggregate_group_metrics(group_metrics)
        summary_rows.append(
            {
                "method": method,
                "precision": "W4A4",
                "w_bits": 4,
                "a_bits": 4,
                "lanes": 6,
                "nx": 2,
                "ny": 3,
                "x_kind": "paper_baseline",
                "offsets": "0 6 12 18 24 30" if method == "DSP-Packing" else "0 7 14 21 28 35",
                "groups": args.groups,
                "samples_per_group": args.samples_per_group,
                **aggregate,
            }
        )

    fields = [
        "method",
        "precision",
        "w_bits",
        "a_bits",
        "lanes",
        "nx",
        "ny",
        "x_kind",
        "offsets",
        "groups",
        "samples_per_group",
        "outputs",
        "error_count",
        "ep",
        "mae",
        "mse",
        "max_abs_err",
    ]
    group_fields = ["group", *[field for field in fields if field not in {"groups", "samples_per_group"}]]
    write_csv(out_dir / "absolute_precision_summary.csv", summary_rows, fields)
    write_csv(out_dir / "absolute_precision_groups.csv", group_rows, group_fields)
    write_5x2_markdown(out_dir / "absolute_precision_5x2_table.md", summary_rows)

    print(f"Wrote {out_dir / 'absolute_precision_summary.csv'}")
    print(f"Wrote {out_dir / 'absolute_precision_groups.csv'}")
    print(f"Wrote {out_dir / 'absolute_precision_5x2_table.md'}")
    for row in summary_rows:
        if row["method"] != "Ultra-DSP" or row["precision"] == "W4A4":
            print(
                f"{row['method']} {row['precision']}: "
                f"EP={float(row['ep']):.6g} MAE={float(row['mae']):.6g} "
                f"MSE={float(row['mse']):.6g} MaxAbsErr={row['max_abs_err']}"
            )


if __name__ == "__main__":
    main()
