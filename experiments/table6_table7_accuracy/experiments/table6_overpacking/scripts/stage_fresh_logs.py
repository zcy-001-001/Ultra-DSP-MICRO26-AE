#!/usr/bin/env python3
"""Validate, sanitize, and atomically stage the eight paper Table 6 logs."""

from __future__ import annotations

import argparse
import runpy
import shutil
import tempfile
from pathlib import Path

from parse_table6 import ROWS, task_values


def _package_root() -> Path:
    return Path(__file__).resolve().parents[5]


def stage_logs(run_dir: Path, out_dir: Path) -> tuple[int, int]:
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite staged evidence: {out_dir}")

    sources: list[tuple[Path, Path]] = []
    for model, _method, relative_log in ROWS:
        relative = Path(model) / relative_log
        source = run_dir / relative
        if not source.is_file():
            raise FileNotFoundError(f"missing fresh log: {relative.as_posix()}")
        text = source.read_text(encoding="utf-8", errors="replace")
        if "AVERAGE" not in text:
            raise RuntimeError(f"incomplete fresh log: {relative.as_posix()}")
        task_values(source)
        sources.append((source, relative))

    sanitizer_path = _package_root() / "scripts" / "sanitize_paths.py"
    sanitizer = runpy.run_path(str(sanitizer_path))
    sanitize_text = sanitizer["sanitize_text"]

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{out_dir.name}.", dir=out_dir.parent)
    )
    replacements = 0
    try:
        for source, relative in sources:
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            text = source.read_text(encoding="utf-8", errors="replace")
            sanitized, count = sanitize_text(text)
            replacements += count
            destination.write_text(sanitized, encoding="utf-8", newline="")
            task_values(destination)

        packaged = [path for path in temporary.rglob("*") if path.is_file()]
        if len(packaged) != len(ROWS):
            raise RuntimeError(
                f"expected {len(ROWS)} staged logs, found {len(packaged)}"
            )
        temporary.replace(out_dir)
    except Exception:
        # Only the helper-created temporary directory is removed. Existing
        # evidence is never deleted or overwritten.
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return len(sources), replacements


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    count, replacements = stage_logs(args.run_dir, args.out_dir)
    print(
        f"TABLE6_FRESH_LOGS_STAGED logs={count} "
        f"privacy_replacements={replacements} atomic=PASS"
    )


if __name__ == "__main__":
    main()
