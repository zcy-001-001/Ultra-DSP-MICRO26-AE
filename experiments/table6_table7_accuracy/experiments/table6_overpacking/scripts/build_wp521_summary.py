#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from parse_table6 import TASKS, format_markdown, parse_tables, task_values


DEFAULT_EXPECTED = (
    Path(__file__).resolve().parents[1]
    / "expected"
    / "table6_expected_from_existing_artifacts.json"
)

MODELS = ("llama2_7b", "llama3_8b")


def task_value_from_single_log(log_path: Path, task: str) -> float:
    tables = parse_tables(log_path)
    val = None
    if tables:
        table = tables[-1]
        val = table.get(f"{task}:acc_norm", table.get(f"{task}:acc"))
    if val is None:
        # Single-task lm-eval logs can omit the task-name cell in the final
        # AVERAGE row, so keep this fallback to avoid rerunning completed jobs.
        for line in log_path.read_text(errors="ignore").splitlines():
            if line.startswith("|AVERAGE"):
                nums = re.findall(r"\|([0-9]*\.?[0-9]+)\|", line)
                if nums:
                    val = float(nums[0]) * 100.0
                    break
    if val is None:
        raise RuntimeError(f"Missing {task} in {log_path}")
    return round(float(val), 2)


def wp521_values(run_dir: Path, model: str) -> dict[str, float] | None:
    combined_log = run_dir / model / "wp521" / "log.txt"
    if combined_log.exists() and "AVERAGE" in combined_log.read_text(errors="ignore"):
        return task_values(combined_log)

    vals: dict[str, float] = {}
    for task in TASKS:
        task_log = run_dir / model / "wp521_tasks" / task / "log.txt"
        if not task_log.exists():
            return None
        vals[task] = task_value_from_single_log(task_log, task)
    vals["avg"] = round(sum(vals[t] for t in TASKS) / len(TASKS), 2)
    return vals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--expected", type=Path, default=DEFAULT_EXPECTED)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--out-md", required=True, type=Path)
    ap.add_argument("--require-model", action="append", choices=MODELS, default=[])
    args = ap.parse_args()

    required_models = tuple(args.require_model) if args.require_model else MODELS
    rows = json.loads(args.expected.read_text())
    by_model: dict[str, list[dict[str, object]]] = {model: [] for model in MODELS}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)

    merged: list[dict[str, object]] = []
    for model in MODELS:
        merged.extend(by_model.get(model, []))
        vals = wp521_values(args.run_dir, model)
        if vals is not None:
            merged.append({"model": model, "method": "WP521", **vals})
        elif model in required_models:
            raise RuntimeError(f"Missing required WP521 logs for {model} under {args.run_dir / model}")

    markdown = format_markdown(merged)
    print(markdown, end="")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(merged, indent=2) + "\n")
    args.out_md.write_text(markdown)


if __name__ == "__main__":
    main()
