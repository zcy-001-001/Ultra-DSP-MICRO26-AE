#!/usr/bin/env python3
"""Build the paper-order Table 7 geomean from Table 6/7 model-level rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = HERE.parents[3]
TABLE7_RESULTS = PACKAGE_ROOT / "results" / "table7"
TABLE6_RESULTS = PACKAGE_ROOT / "results" / "table6"
DEFAULT_MIXED = TABLE7_RESULTS / "mixed_precision_ultradsp_summary.json"
DEFAULT_TABLE6 = TABLE6_RESULTS / "table6_archived_summary.json"
# Historical defaults wrote directly to the package-level Table 7 names:
# DEFAULT_JSON = TABLE7_RESULTS / "table7_geomean_summary.json"
# DEFAULT_CSV = TABLE7_RESULTS / "table7_geomean_summary.csv"
# DEFAULT_MD = TABLE7_RESULTS / "table7_geomean_table.md"
#
# AE policy keeps a newly generated run separate from the paper anchor.  The
# optional recomputation therefore writes to explicitly non-canonical names.
DEFAULT_JSON = TABLE7_RESULTS / "table7_optional_recomputed_summary.json"
DEFAULT_CSV = TABLE7_RESULTS / "table7_optional_recomputed_summary.csv"
DEFAULT_MD = TABLE7_RESULTS / "table7_optional_recomputed_table.md"

ORDER = ["W5A5", "W4A5", "W4A4", "W5A4", "W3A5", "W3A4", "W5A3", "W4A3"]
PAPER = {
    "W5A5": 68.86,
    "W4A5": 68.11,
    "W4A4": 66.78,
    "W5A4": 65.92,
    "W3A5": 63.81,
    "W3A4": 60.78,
    "W5A3": 45.94,
    "W4A3": 44.40,
}


def load(path: Path) -> list[dict[str, object]]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_avg(rows: list[dict[str, object]], model: str, method: str) -> float:
    matches = [row for row in rows if row["model"] == model and row["method"] == method]
    if len(matches) != 1:
        raise ValueError(f"expected one row for {model}/{method}, found {len(matches)}")
    return float(matches[0]["avg"])


def build(mixed: list[dict[str, object]], table6: list[dict[str, object]]) -> list[dict[str, object]]:
    result = []
    for precision in ORDER:
        if precision == "W4A4":
            l2 = find_avg(table6, "llama2_7b", "Ultra-DSP")
            l3 = find_avg(table6, "llama3_8b", "Ultra-DSP")
            source_kind = "TABLE6_ARCHIVED_SUMMARY"
        else:
            method = f"Ultra-DSP {precision}"
            l2 = find_avg(mixed, "llama2_7b", method)
            l3 = find_avg(mixed, "llama3_8b", method)
            source_kind = "RECOMPUTED_FROM_LOGS"
        geomean = round(math.sqrt(l2 * l3), 2)
        paper = PAPER[precision]
        result.append(
            {
                "format": precision,
                "llama2_avg": l2,
                "llama3_avg": l3,
                "geomean": geomean,
                "paper_anchor": paper,
                "abs_diff": round(abs(geomean - paper), 2),
                "source_kind": source_kind,
                "status": "PASS" if abs(geomean - paper) <= 0.01 else "MISMATCH",
            }
        )
    return result


def write(rows: list[dict[str, object]], out_json: Path, out_csv: Path, out_md: Path) -> None:
    for path in (out_json, out_csv, out_md):
        path.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "| Format | Llama-2 Avg | Llama-3 Avg | Geomean | Paper | Source | Status |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['format']} | {row['llama2_avg']:.2f} | {row['llama3_avg']:.2f} | "
            f"{row['geomean']:.2f} | {row['paper_anchor']:.2f} | {row['source_kind']} | {row['status']} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixed-summary", type=Path, default=DEFAULT_MIXED)
    parser.add_argument("--table6-summary", type=Path, default=DEFAULT_TABLE6)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    rows = build(load(args.mixed_summary), load(args.table6_summary))
    write(rows, args.out_json, args.out_csv, args.out_md)
    if any(row["status"] != "PASS" for row in rows):
        raise SystemExit("TABLE7_GEOMEAN_MISMATCH")
    print(f"TABLE7_GEOMEAN_PASS formats={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
