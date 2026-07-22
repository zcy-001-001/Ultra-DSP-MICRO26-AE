#!/usr/bin/env python3
"""Generate the W4A4 single-DSP component-ablation packing summary."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PACKAGE_ROOT / "results" / "component_ablation"


@dataclass(frozen=True)
class Layout:
    prefill_t: int
    decode_t: int
    ep_lanes: int
    slot_width: int
    offsets: tuple[int, ...]

    def max_adjacent_overlap(self) -> int:
        offsets = self.offsets
        if len(offsets) <= 1:
            return 0
        return max(max(0, offsets[i - 1] + self.slot_width - offsets[i]) for i in range(1, len(offsets)))

    def total_adjacent_overlap(self) -> int:
        offsets = self.offsets
        return sum(max(0, offsets[i - 1] + self.slot_width - offsets[i]) for i in range(1, len(offsets)))


VARIANTS = [
    {
        "variant": "V0",
        "short_name": "NormalSigned",
        "plot_label": "Normal Signed Packing",
        "contribution": "baseline normal signed two's-complement packing; P pre-adds two activations so two physical lanes represent four effective MACs",
        "layout": Layout(4, 2, 2, 8, (0, 8)),
        "layout_source": "baseline",
        "exact": True,
    },
    {
        "variant": "V1",
        "short_name": "SignMagnitude",
        "plot_label": "+ Sign-Magnitude",
        "contribution": "online sign-magnitude decoupling without output-domain overlap",
        "layout": Layout(6, 5, 6, 6, (0, 6, 12, 18, 24, 30)),
        "layout_source": "manual_non_ilp",
        "exact": True,
    },
    {
        "variant": "V2",
        "short_name": "OverlapNoCorr",
        "plot_label": "+ Overlap",
        "contribution": "hand-tuned sign-magnitude overlap without correction; not claimed to be ILP-optimal",
        "layout": Layout(8, 6, 8, 6, (0, 6, 9, 12, 15, 18, 21, 27)),
        "layout_source": "hand_tuned",
        "exact": False,
    },
    {
        "variant": "V3",
        "short_name": "FullCorrection",
        "plot_label": "+ Full Correction",
        "contribution": "full correction on the same hand-tuned P=8/D=6 overlap layout",
        "layout": Layout(8, 6, 8, 6, (0, 6, 9, 12, 15, 18, 21, 27)),
        "layout_source": "hand_tuned",
        "exact": True,
    },
    {
        "variant": "V4",
        "short_name": "FullUltraDSP",
        "plot_label": "+ ILP Layout + Resource Opt.",
        "contribution": "ILP-selected and resource-optimized Ultra-DSP W4A4 3x3 prefill and 1x7 decode layout",
        "layout": Layout(9, 7, 9, 6, (0, 4, 8, 11, 15, 19, 23, 27, 31)),
        "layout_source": "ilp_solver",
        "exact": True,
    },
]


def write_summary(out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "short_name",
                "plot_label",
                "prefill_t",
                "decode_t",
                "ep_lanes",
                "slot_width_bits",
                "offsets",
                "max_adjacent_overlap_bits",
                "total_adjacent_overlap_bits",
                "layout_source",
                "exact_arithmetic_expected",
                "contribution",
            ],
        )
        writer.writeheader()
        for item in VARIANTS:
            layout: Layout = item["layout"]
            writer.writerow(
                {
                    "variant": item["variant"],
                    "short_name": item["short_name"],
                    "plot_label": item["plot_label"],
                    "prefill_t": layout.prefill_t,
                    "decode_t": layout.decode_t,
                    "ep_lanes": layout.ep_lanes,
                    "slot_width_bits": layout.slot_width,
                    "offsets": " ".join(map(str, layout.offsets)),
                    "max_adjacent_overlap_bits": layout.max_adjacent_overlap(),
                    "total_adjacent_overlap_bits": layout.total_adjacent_overlap(),
                    "layout_source": item["layout_source"],
                    "exact_arithmetic_expected": int(item["exact"]),
                    "contribution": item["contribution"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "packing_summary.csv")
    args = parser.parse_args()
    write_summary(Path(args.out))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
