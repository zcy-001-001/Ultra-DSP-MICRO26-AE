#!/usr/bin/env python3
"""Measure single-DSP W4A4 product-level EP for cumulative ablation steps."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PACKAGE_ROOT / "results" / "component_ablation"


SIGNED_VALUES = np.arange(-7, 8, dtype=np.int16)


@dataclass(frozen=True)
class Variant:
    variant: str
    short_name: str
    plot_label: str
    prefill_t: int
    decode_t: int
    ep_lanes: int
    slot_width: int
    offsets: tuple[int, ...]
    mode: str


VARIANTS = [
    Variant("V0", "NormalSigned", "Normal Signed Packing", 4, 2, 2, 8, (0, 8), "exact"),
    Variant("V1", "SignMagnitude", "+ Sign-Magnitude", 6, 5, 6, 6, (0, 6, 12, 18, 24, 30), "exact"),
    Variant("V2", "OverlapNoCorr", "+ Overlap", 8, 6, 8, 6, (0, 6, 9, 12, 15, 18, 21, 27), "raw_signmag_overlap"),
    Variant("V3", "FullCorrection", "+ Full Correction", 8, 6, 8, 6, (0, 6, 9, 12, 15, 18, 21, 27), "exact"),
    Variant("V4", "FullUltraDSP", "+ ILP Layout + Resource Opt.", 9, 7, 9, 6, (0, 4, 8, 11, 15, 19, 23, 27, 31), "exact"),
]


def twos_mask(width: int) -> int:
    return (1 << width) - 1


def encode_twos(values: np.ndarray, width: int) -> np.ndarray:
    return values.astype(np.int64) & twos_mask(width)


def decode_twos(values: np.ndarray, width: int) -> np.ndarray:
    sign = 1 << (width - 1)
    mask = twos_mask(width)
    values = values.astype(np.int64) & mask
    return ((values ^ sign) - sign).astype(np.int32)


def random_operands(rng: np.random.Generator, samples: int, lanes: int) -> tuple[np.ndarray, np.ndarray]:
    w = rng.choice(SIGNED_VALUES, size=(samples, lanes), replace=True).astype(np.int32)
    a = rng.choice(SIGNED_VALUES, size=(samples, lanes), replace=True).astype(np.int32)
    return w, a


def exact_products(w: np.ndarray, a: np.ndarray) -> np.ndarray:
    return w * a


def raw_signed_overlap(products: np.ndarray, offsets: tuple[int, ...], slot_width: int) -> np.ndarray:
    packed = np.zeros(products.shape[0], dtype=np.int64)
    for i, offset in enumerate(offsets):
        packed += encode_twos(products[:, i], slot_width) << offset

    decoded = []
    for offset in offsets:
        field = (packed >> offset) & twos_mask(slot_width)
        decoded.append(decode_twos(field, slot_width)[:, None])
    return np.concatenate(decoded, axis=1)


def raw_signmag_overlap(w: np.ndarray, a: np.ndarray, offsets: tuple[int, ...], slot_width: int) -> np.ndarray:
    """Model the raw-overlap sign-magnitude datapath.

    Magnitude products are packed as unsigned fields in the DSP P word; sign
    bits are carried separately and reapplied after slicing the polluted fields.
    """
    magnitudes = np.abs(w.astype(np.int64)) * np.abs(a.astype(np.int64))
    negative = (w < 0) ^ (a < 0)
    packed = np.zeros(w.shape[0], dtype=np.int64)
    for i, offset in enumerate(offsets):
        packed += magnitudes[:, i] << offset

    decoded = []
    for i, offset in enumerate(offsets):
        field = ((packed >> offset) & twos_mask(slot_width)).astype(np.int32)
        signed_field = np.where(negative[:, i], -field, field)
        decoded.append(signed_field[:, None])
    return np.concatenate(decoded, axis=1)


def summarize_diff(diff: np.ndarray) -> dict[str, float | int]:
    total = int(diff.size)
    errors = int(np.count_nonzero(diff))
    abs_diff = np.abs(diff.astype(np.int64))
    return {
        "outputs": total,
        "error_count": errors,
        "ep": float(errors / total),
        "mae": float(np.mean(abs_diff)),
        "mse": float(np.mean(diff.astype(np.int64) ** 2)),
        "max_abs_err": int(np.max(abs_diff)) if total else 0,
    }


def measure_group(rng: np.random.Generator, variant: Variant, samples: int) -> dict[str, float | int]:
    w, a = random_operands(rng, samples, variant.ep_lanes)
    ref = exact_products(w, a)
    if variant.mode == "raw_signed_overlap":
        got = raw_signed_overlap(ref, variant.offsets, variant.slot_width)
    elif variant.mode == "raw_signmag_overlap":
        got = raw_signmag_overlap(w, a, variant.offsets, variant.slot_width)
    else:
        got = ref.copy()
    return summarize_diff(got - ref)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=int, default=20)
    parser.add_argument("--samples-per-group", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1305)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "component_ablation_w4a4.csv")
    parser.add_argument("--group-out", type=Path, default=RESULTS_DIR / "component_ablation_w4a4_groups.csv")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    group_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for variant in VARIANTS:
        variant_group_metrics = []
        for group_id in range(args.groups):
            metrics = measure_group(rng, variant, args.samples_per_group)
            variant_group_metrics.append(metrics)
            group_rows.append(
                {
                    "group": group_id,
                    "variant": variant.variant,
                    "short_name": variant.short_name,
                    "plot_label": variant.plot_label,
                    "prefill_t": variant.prefill_t,
                    "decode_t": variant.decode_t,
                    "ep_lanes": variant.ep_lanes,
                    "slot_width": variant.slot_width,
                    "offsets": " ".join(map(str, variant.offsets)),
                    **metrics,
                }
            )

        outputs = sum(int(m["outputs"]) for m in variant_group_metrics)
        errors = sum(int(m["error_count"]) for m in variant_group_metrics)
        ep_values = np.array([float(m["ep"]) for m in variant_group_metrics])
        weighted_mae = sum(float(m["mae"]) * int(m["outputs"]) for m in variant_group_metrics) / outputs
        weighted_mse = sum(float(m["mse"]) * int(m["outputs"]) for m in variant_group_metrics) / outputs
        summary_rows.append(
            {
                "variant": variant.variant,
                "short_name": variant.short_name,
                "plot_label": variant.plot_label,
                "prefill_t": variant.prefill_t,
                "decode_t": variant.decode_t,
                "ep_lanes": variant.ep_lanes,
                "slot_width": variant.slot_width,
                "offsets": " ".join(map(str, variant.offsets)),
                "groups": args.groups,
                "samples_per_group": args.samples_per_group,
                "outputs": outputs,
                "error_count": errors,
                "ep": errors / outputs,
                "ep_std": float(ep_values.std(ddof=1)) if len(ep_values) > 1 else 0.0,
                "ep_min": float(ep_values.min()),
                "ep_max": float(ep_values.max()),
                "mae": weighted_mae,
                "mse": weighted_mse,
                "max_abs_err": max(int(m["max_abs_err"]) for m in variant_group_metrics),
            }
        )

    write_csv(
        Path(args.group_out),
        group_rows,
        [
            "group",
            "variant",
            "short_name",
            "plot_label",
            "prefill_t",
            "decode_t",
            "ep_lanes",
            "slot_width",
            "offsets",
            "outputs",
            "error_count",
            "ep",
            "mae",
            "mse",
            "max_abs_err",
        ],
    )
    write_csv(
        Path(args.out),
        summary_rows,
        [
            "variant",
            "short_name",
            "plot_label",
            "prefill_t",
            "decode_t",
            "ep_lanes",
            "slot_width",
            "offsets",
            "groups",
            "samples_per_group",
            "outputs",
            "error_count",
            "ep",
            "ep_std",
            "ep_min",
            "ep_max",
            "mae",
            "mse",
            "max_abs_err",
        ],
    )

    print(f"Wrote {args.out}")
    print(f"Wrote {args.group_out}")
    for row in summary_rows:
        print(
            f"{row['variant']} {row['short_name']}: "
            f"P/D={row['prefill_t']}/{row['decode_t']} EP={100 * float(row['ep']):.4f}% "
            f"MAE={float(row['mae']):.4f} MSE={float(row['mse']):.4f}"
        )


if __name__ == "__main__":
    main()
