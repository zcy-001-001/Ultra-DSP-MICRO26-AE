#!/usr/bin/env python3
"""Compare a fresh Table 6 summary with the archived numerical anchor.

The comparison never rewrites either input.  It reports raw deltas, applies
explicit AE tolerances, and separately checks the paper's main method trend.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

METRICS = ("arc_easy", "hellaswag", "piqa", "openbookqa")
MODELS = ("llama2_7b", "llama3_8b")
METHODS = ("Baseline(BF16)", "Ultra-DSP", "DSP-Packing", "DB-MixQ")
FORMAL_KEYS = {(model, method) for model in MODELS for method in METHODS}


def load_rows(path: Path) -> list[dict[str, object]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON row list: {path}")
    return rows


def row_key(row: dict[str, object]) -> tuple[str, str]:
    return str(row["model"]), str(row["method"])


def author_accept_key(value: str) -> tuple[str, str]:
    if "/" not in value:
        raise argparse.ArgumentTypeError(
            "author acceptance must use MODEL/METHOD, for example llama2_7b/Ultra-DSP"
        )
    model, method = value.split("/", 1)
    if not model or not method:
        raise argparse.ArgumentTypeError("author acceptance MODEL and METHOD must be non-empty")
    return model, method


def trend_checks(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key = {row_key(row): row for row in rows}
    checks: list[dict[str, object]] = []
    for model in MODELS:
        required = list(METHODS)
        missing = [method for method in required if (model, method) not in by_key]
        if missing:
            checks.append(
                {
                    "model": model,
                    "check": "BF16 > Ultra-DSP > lossy packing baselines",
                    "status": "FAIL",
                    "detail": f"missing rows: {', '.join(missing)}",
                }
            )
            continue
        averages = {
            method: float(by_key[(model, method)]["avg"]) for method in required
        }
        passed = (
            averages["Baseline(BF16)"] > averages["Ultra-DSP"]
            and averages["Ultra-DSP"] > averages["DSP-Packing"]
            and averages["Ultra-DSP"] > averages["DB-MixQ"]
        )
        checks.append(
            {
                "model": model,
                "check": "BF16 > Ultra-DSP > lossy packing baselines",
                "status": "PASS" if passed else "FAIL",
                "detail": averages,
            }
        )
    return checks


def markdown_report(report: dict[str, object]) -> str:
    lines = [
        "# Table 6 fresh-run tolerance comparison",
        "",
        f"Run ID: `{report['run_id']}`  ",
        f"Per-task tolerance: {report['metric_tolerance']:.2f} percentage points  ",
        f"Average tolerance: {report['average_tolerance']:.2f} percentage points",
        f"Strict status: **{report['strict_status']}**  ",
        f"Final AE status: **{report['status']}**",
        "",
        "| Model | Method | Max task delta | Average delta | Strict status | AE status |",
        "|---|---|---:|---:|---|---|",
    ]
    for item in report["comparisons"]:
        lines.append(
            f"| {item['model']} | {item['method']} | {item['max_task_abs_delta']:.2f} | "
            f"{item['average_delta']:+.2f} | {item['strict_status']} | {item['status']} |"
        )
    for item in report["reference_only_rows"]:
        lines.append(
            f"| {item['model']} | {item['method']} | n/a | n/a | "
            "REFERENCE_ONLY | REFERENCE_ONLY |"
        )
    lines.extend(["", "## Trend checks", ""])
    for check in report["trend_checks"]:
        lines.append(f"- {check['status']}: `{check['model']}` - {check['check']}")
    # Historical mojibake formatter retained below as unreachable maintenance
    # reference; it is not used to render the public report.
    for check in ():
        lines.append(f"- {check['status']}: `{check['model']}` — {check['check']}")
    lines.extend(["", f"Overall status: **{report['status']}**", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", required=True, type=Path)
    parser.add_argument("--expected", required=True, type=Path)
    parser.add_argument("--metric-tol", type=float, default=3.0)
    parser.add_argument("--avg-tol", type=float, default=2.0)
    parser.add_argument(
        "--run-id",
        help="Portable identifier shared with the fresh summary and formal evidence",
    )
    parser.add_argument(
        "--author-accept",
        action="append",
        default=[],
        type=author_accept_key,
        metavar="MODEL/METHOD",
        help=(
            "Explicitly accept one strictly out-of-tolerance row while preserving "
            "its raw deltas and strict_status; may be repeated"
        ),
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--out-md", type=Path)
    args = parser.parse_args()

    fresh = load_rows(args.fresh)
    row_run_ids = [str(row["run_id"]) for row in fresh if row.get("run_id")]
    if row_run_ids and len(row_run_ids) != len(fresh):
        parser.error("fresh rows must either all contain run_id or all omit it")
    unique_run_ids = set(row_run_ids)
    if len(unique_run_ids) > 1:
        parser.error("fresh rows contain multiple run_id values")
    inferred_run_id = next(iter(unique_run_ids), "ARCHIVE_SELFTEST")
    run_id = args.run_id or inferred_run_id
    if unique_run_ids and run_id != inferred_run_id:
        parser.error("--run-id disagrees with the fresh summary run_id")
    fresh_keys = [row_key(row) for row in fresh]
    if len(fresh_keys) != len(set(fresh_keys)):
        parser.error("fresh summary contains duplicate model/method rows")
    if set(fresh_keys) != FORMAL_KEYS:
        missing = FORMAL_KEYS - set(fresh_keys)
        extra = set(fresh_keys) - FORMAL_KEYS
        detail = []
        if missing:
            detail.append(
                "missing=" + ",".join(f"{model}/{method}" for model, method in sorted(missing))
            )
        if extra:
            detail.append(
                "extra=" + ",".join(f"{model}/{method}" for model, method in sorted(extra))
            )
        parser.error("fresh Table 6 must contain exactly the eight paper rows: " + "; ".join(detail))

    # The archived rebuttal anchor may also contain WP521 rows used by Figure
    # 12.  They are intentionally filtered here: formal Table 6 compares only
    # the eight paper rows and reports no reference-only extension rows.
    archived = {row_key(row): row for row in load_rows(args.expected)}
    expected = {key: archived[key] for key in FORMAL_KEYS if key in archived}
    if set(expected) != FORMAL_KEYS:
        parser.error("archived anchor is missing one or more formal Table 6 rows")
    author_accepts = set(args.author_accept)
    unknown_accepts = author_accepts - set(expected)
    if unknown_accepts:
        parser.error(
            "--author-accept names rows without an archived comparison: "
            + ", ".join(f"{model}/{method}" for model, method in sorted(unknown_accepts))
        )
    comparisons: list[dict[str, object]] = []
    reference_only: list[dict[str, str]] = []
    for row in fresh:
        key = row_key(row)
        anchor = expected[key]
        deltas = {metric: float(row[metric]) - float(anchor[metric]) for metric in METRICS}
        average_delta = float(row["avg"]) - float(anchor["avg"])
        max_task_delta = max(abs(value) for value in deltas.values())
        passed = max_task_delta <= args.metric_tol and abs(average_delta) <= args.avg_tol
        author_accepted = not passed and key in author_accepts
        comparisons.append(
            {
                "model": key[0],
                "method": key[1],
                "task_deltas": deltas,
                "max_task_abs_delta": round(max_task_delta, 2),
                "average_delta": round(average_delta, 2),
                "strict_status": "PASS" if passed else "OUT_OF_TOLERANCE",
                "status": (
                    "PASS" if passed
                    else "AUTHOR_ACCEPTED_VARIANCE" if author_accepted
                    else "OUT_OF_TOLERANCE"
                ),
            }
        )

    trends = trend_checks(fresh)
    strict_passed = (
        len(comparisons) == len(expected)
        and all(item["strict_status"] == "PASS" for item in comparisons)
        and all(item["status"] == "PASS" for item in trends)
    )
    final_passed = (
        len(comparisons) == len(expected)
        and all(item["status"] in {"PASS", "AUTHOR_ACCEPTED_VARIANCE"} for item in comparisons)
        and all(item["status"] == "PASS" for item in trends)
    )
    final_status = (
        "PASS" if strict_passed
        else "PASS_WITH_AUTHOR_ACCEPTED_VARIANCE" if final_passed
        else "OUT_OF_TOLERANCE"
    )
    report = {
        "run_id": run_id,
        "strict_status": "PASS" if strict_passed else "OUT_OF_TOLERANCE",
        "status": final_status,
        "metric_tolerance": args.metric_tol,
        "average_tolerance": args.avg_tol,
        "comparisons": comparisons,
        "reference_only_rows": reference_only,
        "trend_checks": trends,
    }
    rendered = markdown_report(report)
    print(rendered, end="")
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.out_md:
        args.out_md.write_text(rendered, encoding="utf-8")
    signal = (
        f"TABLE6_TOLERANCE_{report['status']} comparable_rows={len(comparisons)} "
        f"reference_only_rows={len(reference_only)} trend_checks={len(trends)}"
    )
    print(signal)
    if not final_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
