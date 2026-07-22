#!/usr/bin/env python3
"""Write the active Vivado resource summary for the W4A4 ablation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PACKAGE_ROOT / "results" / "component_ablation"


ACTIVE_TOPS = [
    ("V0", "w4a4_sf_v0_normal_signed_p2d2_single_dsp"),
    ("V1", "w4a4_sf_v1_signmag_nonoverlap_p6d5_single_dsp"),
    ("V2", "w4a4_sf_v2_overlap_no_correction_p8d6_single_dsp"),
    ("V3", "w4a4_sf_v3_full_correction_p8d6_single_dsp"),
    ("V4", "W4A4_PD_End"),
]


def read_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return {row["top_module"]: row for row in csv.DictReader(f)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=RESULTS_DIR / "vivado_signmag_first_summary.csv")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "vivado_signmag_first_active_summary.csv")
    args = parser.parse_args()

    rows_by_top = read_csv(Path(args.summary))
    out_rows = []
    for variant, top in ACTIVE_TOPS:
        if top not in rows_by_top:
            raise SystemExit(f"Missing Vivado utilization row for active top: {top}")
        row = rows_by_top[top]
        out_rows.append(
            {
                "variant": variant,
                "top_module": top,
                "clb_luts": row["clb_luts"],
                "clb_registers": row["clb_registers"],
                "dsp": row["dsp"],
                "report": row["report"],
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["variant", "top_module", "clb_luts", "clb_registers", "dsp", "report"],
        )
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {out} with {len(out_rows)} active rows")


if __name__ == "__main__":
    main()
