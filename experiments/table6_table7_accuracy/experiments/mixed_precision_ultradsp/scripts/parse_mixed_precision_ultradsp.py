#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

TASKS = ["arc_easy", "hellaswag", "piqa", "openbookqa"]
MODEL_ORDER = ["llama2_7b", "llama3_8b"]
PRECISION_ORDER = ["W5A5", "W4A3", "W3A4", "W4A5", "W5A4", "W3A5", "W5A3"]


def parse_tables(log_path: Path) -> list[dict[str, float]]:
    tables: list[dict[str, float]] = []
    current: dict[str, float] | None = None
    last_task: str | None = None
    for line in log_path.read_text(errors="ignore").splitlines():
        parts = [part.strip() for part in line.split("|")] if line.startswith("|") else []
        if len(parts) > 1 and parts[1] == "Tasks":
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
                current[f"{task}:{metric}"] = float(value) * 100.0
            except ValueError:
                continue
    if current:
        tables.append(current)
    return tables


def task_values(log_path: Path) -> dict[str, float]:
    tables = parse_tables(log_path)
    if not tables:
        raise RuntimeError(f"No lm-eval table found in {log_path}")
    table = tables[-1]
    vals: dict[str, float] = {}
    for task in TASKS:
        value = table.get(f"{task}:acc_norm", table.get(f"{task}:acc"))
        if value is None:
            raise RuntimeError(f"Missing {task} in {log_path}")
        vals[task] = round(value, 2)
    vals["avg"] = round(sum(vals[t] for t in TASKS) / len(TASKS), 2)
    return vals


def markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "| Model | Method | ARC-e | Hella | PIQA | OBQA | Avg |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['method']} | {row['arc_easy']:.2f} | "
            f"{row['hellaswag']:.2f} | {row['piqa']:.2f} | {row['openbookqa']:.2f} | {row['avg']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--out-json", type=Path)
    ap.add_argument("--out-md", type=Path)
    ap.add_argument("--allow-missing", action="store_true")
    args = ap.parse_args()

    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for model in MODEL_ORDER:
        for precision in PRECISION_ORDER:
            log_path = args.run_dir / model / precision / "eval_ultradsp" / "log.txt"
            if not log_path.exists():
                missing.append(str(log_path))
                continue
            vals = task_values(log_path)
            rows.append(
                {
                    "model": model,
                    "method": f"Ultra-DSP {precision}",
                    "precision": precision,
                    "log": str(log_path),
                    **vals,
                }
            )

    if missing and not args.allow_missing:
        raise RuntimeError("Missing logs:\n" + "\n".join(missing))

    md = markdown(rows)
    print(md, end="")
    if missing:
        print("\nMissing logs:")
        for item in missing:
            print(item)

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(rows, indent=2) + "\n")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md)


if __name__ == "__main__":
    main()
