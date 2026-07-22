#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

TASKS = ["arc_easy", "hellaswag", "piqa", "winogrande"]
ROWS = [
    ("llama2_7b", "FP16 baseline", "fp16"),
    ("llama2_7b", "I-BERT", "ibert"),
    ("llama2_7b", "Gemmlowp", "gemmlowp"),
    ("llama3_8b", "FP16 baseline", "fp16"),
    ("llama3_8b", "I-BERT", "ibert"),
    ("llama3_8b", "Gemmlowp", "gemmlowp"),
]
MODEL_LABELS = {
    "llama2_7b": "LLaMA-2-7B",
    "llama3_8b": "LLaMA-3-8B",
}
METHOD_LABELS = {
    "fp16": "FP16 baseline",
    "ibert": "I-BERT",
    "gemmlowp": "Gemmlowp",
    "gemmlowp_strict": "Gemmlowp (strict)",
    "gemmlowp_w8a8": "Gemmlowp (W8A8)",
    "gemmlowp_w8a8_mid": "Gemmlowp (W8A8-mid)",
    "gemmlowp_w8a8_lossy": "Gemmlowp (W8A8-lossy)",
    "gemmlowp_fixedpoint": "Gemmlowp (fixedpoint)",
    "gemmlowp_softmax_int8": "Gemmlowp (softmax_int8)",
    "gemmlowp_no_mul": "Gemmlowp (no_mul)",
}


def parse_ppl(log_text: str) -> float | None:
    ppl = None
    for match in re.finditer(r"PPL:\s*([0-9]+(?:\.[0-9]+)?|nan|inf)", log_text, flags=re.IGNORECASE):
        token = match.group(1).lower()
        if token == "nan":
            ppl = float("nan")
        elif token == "inf":
            ppl = float("inf")
        else:
            ppl = float(token)
    return ppl


def parse_lm_eval_tables(log_text: str) -> list[dict[str, float]]:
    tables: list[dict[str, float]] = []
    current: dict[str, float] | None = None
    last_task: str | None = None

    for line in log_text.splitlines():
        if "|" in line and not line.lstrip().startswith("|"):
            line = line[line.index("|") :]
        stripped = line.lstrip()
        if stripped.startswith("|") and len([part.strip() for part in stripped.split("|")]) > 2:
            maybe_parts = [part.strip() for part in stripped.split("|")]
            if len(maybe_parts) > 2 and maybe_parts[1] == "Tasks":
                current = {}
                last_task = None
                continue
        if line.startswith("|    Tasks") or line.startswith("|Tasks"):
            current = {}
            last_task = None
            continue
        if current is None:
            continue
        if not stripped.startswith("|"):
            if current:
                tables.append(current)
            current = None
            continue

        parts = [part.strip() for part in stripped.split("|")]
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
            current[f"{task}:{metric}"] = val

    if current:
        tables.append(current)
    return tables


def _missing_task_values() -> dict[str, float | None]:
    vals: dict[str, float | None] = {task: None for task in TASKS}
    vals["acc_avg"] = None
    vals["acc_geomean"] = None
    return vals


def _task_geomean(vals: dict[str, float]) -> float:
    product = 1.0
    for task in TASKS:
        product *= max(vals[task], 0.0) / 100.0
    return round((product ** (1.0 / len(TASKS))) * 100.0, 2)


def task_values(log_text: str, log_path: Path, allow_missing_metrics: bool = False) -> dict[str, float | None]:
    tables = parse_lm_eval_tables(log_text)
    if not tables:
        if allow_missing_metrics:
            return _missing_task_values()
        raise RuntimeError(f"No lm-eval table found in {log_path}")
    table = tables[-1]
    vals: dict[str, float] = {}
    for task in TASKS:
        value = table.get(f"{task}:acc_norm", table.get(f"{task}:acc"))
        if value is None:
            if allow_missing_metrics:
                return _missing_task_values()
            raise RuntimeError(f"Missing {task} in {log_path}")
        vals[task] = round(value, 2)
    vals["acc_avg"] = round(sum(vals[task] for task in TASKS) / len(TASKS), 2)
    vals["acc_geomean"] = _task_geomean(vals)
    return vals


def read_command(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(errors="ignore").strip()


def collect_row(
    run_dir: Path,
    model: str,
    method_label: str,
    method_key: str,
    allow_missing_metrics: bool = False,
) -> dict[str, Any]:
    row_dir = run_dir / model / method_key
    log_path = row_dir / "log.txt"
    if not log_path.exists():
        raise FileNotFoundError(log_path)
    log_text = log_path.read_text(errors="ignore")
    ppl = parse_ppl(log_text)
    if ppl is None:
        raise RuntimeError(f"Missing PPL in {log_path}")
    vals = task_values(log_text, log_path, allow_missing_metrics)
    return {
        "model": model,
        "model_label": MODEL_LABELS.get(model, model),
        "method": method_label,
        "method_key": method_key,
        "ppl": round(ppl, 4),
        "log": str(log_path),
        "command_file": str(row_dir / "command.txt"),
        "command": read_command(row_dir / "command.txt"),
        **vals,
    }


def format_markdown(rows: list[dict[str, Any]]) -> str:
    def fmt(value: Any, digits: int = 2) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    lines = [
        "| Model | Method | PPL | ARC-e | HellaSwag | PIQA | Winogrande | Avg | Geomean | Log | Command |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model_label']} | {row['method']} | {row['ppl']:.4g} | "
            f"{fmt(row['arc_easy'])} | {fmt(row['hellaswag'])} | {fmt(row['piqa'])} | "
            f"{fmt(row['winogrande'])} | {fmt(row['acc_avg'])} | {fmt(row['acc_geomean'])} | "
            f"{row['log']} | {row['command_file']} |"
        )
    return "\n".join(lines) + "\n"


def discover_extra_rows(run_dir: Path, seen: set[tuple[str, str]]) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for model in MODEL_LABELS:
        model_dir = run_dir / model
        if not model_dir.exists():
            continue
        for child in sorted(model_dir.iterdir()):
            if not child.is_dir():
                continue
            method_key = child.name
            if (model, method_key) in seen:
                continue
            if method_key in METHOD_LABELS or method_key.startswith("gemmlowp_"):
                label = METHOD_LABELS.get(method_key, f"Gemmlowp ({method_key.removeprefix('gemmlowp_')})")
                rows.append((model, label, method_key))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--out-md", type=Path)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--allow-missing-metrics", action="store_true")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    seen: set[tuple[str, str]] = set()
    for model, method_label, method_key in ROWS:
        seen.add((model, method_key))
        try:
            rows.append(
                collect_row(
                    args.run_dir,
                    model,
                    method_label,
                    method_key,
                    allow_missing_metrics=args.allow_missing_metrics,
                )
            )
        except Exception as exc:
            if not args.allow_missing:
                raise
            missing.append(f"{model}/{method_key}: {exc}")
    for model, method_label, method_key in discover_extra_rows(args.run_dir, seen):
        try:
            rows.append(
                collect_row(
                    args.run_dir,
                    model,
                    method_label,
                    method_key,
                    allow_missing_metrics=args.allow_missing_metrics,
                )
            )
        except Exception as exc:
            if not args.allow_missing:
                raise
            missing.append(f"{model}/{method_key}: {exc}")

    markdown = format_markdown(rows)
    print(markdown, end="")
    if missing:
        print("\nMissing rows:")
        for item in missing:
            print(f"- {item}")

    payload = {"rows": rows, "missing": missing}
    if args.out:
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
    if args.out_md:
        args.out_md.write_text(markdown)


if __name__ == "__main__":
    main()
