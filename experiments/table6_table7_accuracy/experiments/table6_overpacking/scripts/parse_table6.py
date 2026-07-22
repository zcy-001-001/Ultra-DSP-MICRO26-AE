#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

TASKS = ["arc_easy", "hellaswag", "piqa", "openbookqa"]
ROWS = [
    ("llama2_7b", "Baseline(BF16)", "bf16/log.txt"),
    ("llama2_7b", "Ultra-DSP", "ultra_dsp/log.txt"),
    ("llama2_7b", "DSP-Packing", "dsp_packing/log.txt"),
    ("llama2_7b", "DB-MixQ", "db_mixq/log.txt"),
    # WP521 is a rebuttal/Figure 12 extension, not a paper Table 6 row.
    # Historical fresh-parser entry retained for maintenance reference:
    # ("llama2_7b", "WP521", "wp521/log.txt"),
    ("llama3_8b", "Baseline(BF16)", "bf16/log.txt"),
    ("llama3_8b", "Ultra-DSP", "ultra_dsp/log.txt"),
    ("llama3_8b", "DSP-Packing", "dsp_packing/log.txt"),
    ("llama3_8b", "DB-MixQ", "db_mixq/log.txt"),
    # ("llama3_8b", "WP521", "wp521/log.txt"),
]

MODEL_LABELS = {
    "llama2_7b": "Llama-2-7B",
    "llama3_8b": "Llama-3-8B",
}


def parse_tables(log_path: Path) -> list[dict[str, float]]:
    tables: list[dict[str, float]] = []
    current: dict[str, float] | None = None
    last_task: str | None = None
    for line in log_path.read_text(errors="ignore").splitlines():
        # Historical parser depended on lm-eval's exact column padding:
        # if line.startswith("|    Tasks") or line.startswith("|Tasks"):
        # Current lm-eval may render `|  Tasks   |`; identify the semantic
        # first column instead so harmless padding changes remain parseable.
        header_parts = [part.strip() for part in line.split("|")]
        if len(header_parts) > 1 and header_parts[1] == "Tasks":
            current = {}
            last_task = None
            continue
        if current is None:
            continue
        if not line.startswith("|"):
            if current:
                tables.append(current)
            current = None
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 8:
            continue
        task = parts[1]
        metric = parts[5]
        value = parts[7]
        if task and set(task) != {"-"} and task != "Tasks":
            last_task = task
        else:
            task = last_task
        if task in TASKS and metric in {"acc_norm", "acc"}:
            try:
                val = float(value) * 100.0
            except ValueError:
                continue
            # Prefer acc_norm when both are present.
            key = f"{task}:{metric}"
            current[key] = val
    if current:
        tables.append(current)
    return tables


def task_values(log_path: Path) -> dict[str, float]:
    tables = parse_tables(log_path)
    if not tables:
        raise RuntimeError(f"No lm-eval table found in {log_path}")
    table = tables[-1]
    vals = {}
    for task in TASKS:
        vals[task] = table.get(f"{task}:acc_norm", table.get(f"{task}:acc"))
        if vals[task] is None:
            raise RuntimeError(f"Missing {task} in {log_path}")
        vals[task] = round(vals[task], 2)
    vals["avg"] = round(sum(vals[t] for t in TASKS) / len(TASKS), 2)
    return vals


def format_markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "| Model | Method | ARC-e | Hella | PIQA | OBQA | Avg |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        model = MODEL_LABELS.get(str(row["model"]), str(row["model"]))
        lines.append(
            f"| {model} | {row['method']} | {row['arc_easy']:.2f} | "
            f"{row['hellaswag']:.2f} | {row['piqa']:.2f} | {row['openbookqa']:.2f} | {row['avg']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--out-md", type=Path)
    ap.add_argument(
        "--run-id",
        help="Portable identifier shared by the fresh summary and evidence files",
    )
    ap.add_argument(
        "--source-kind",
        default="FRESH_REMOTE_RERUN",
        help="Provenance label written into every generated row",
    )
    args = ap.parse_args()
    run_id = args.run_id or args.run_dir.name

    rows = []
    for model, method, rel_log in ROWS:
        log_path = args.run_dir / model / rel_log
        vals = task_values(log_path)
        # Keep generated summaries portable and privacy-safe.  The run root may
        # be a private host path, while this model-relative path is sufficient
        # to locate the archived evidence after the result tree is copied.
        relative_log = (Path(model) / rel_log).as_posix()
        rows.append(
            {
                "model": model,
                "method": method,
                "log": relative_log,
                "source_kind": args.source_kind,
                "run_id": run_id,
                **vals,
            }
        )

    markdown = format_markdown(rows)
    print(markdown, end="")
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2) + "\n")
    if args.out_md:
        args.out_md.write_text(markdown)


if __name__ == "__main__":
    main()
