#!/usr/bin/env python3
"""Archive existing remote text reports without launching implementation tools.

The script reads reports through OpenSSH, records the hash and metadata of the
remote raw bytes, sanitizes private path/host metadata in memory, and writes
only the public copy.  It never writes to the remote source tree.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent.parent


def _load_sanitizer():
    path = ROOT / "scripts/sanitize_paths.py"
    spec = importlib.util.spec_from_file_location("ae_sanitize_paths", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load sanitizer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SANITIZER = _load_sanitizer()


@dataclass(frozen=True)
class Request:
    kind: str
    source_relative_path: str
    public_relative_path: str | None = None


def _validate_relative(path: str) -> None:
    parsed = PurePosixPath(path)
    if parsed.is_absolute() or ".." in parsed.parts:
        raise ValueError(f"source path must be relative: {path!r}")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fetch_one(
    request: Request,
    *,
    ssh_host: str,
    source_root: str,
    output_root: Path,
) -> dict[str, object]:
    _validate_relative(request.source_relative_path)
    remote_path = str(PurePosixPath(source_root) / request.source_relative_path)
    command = (
        f"stat -c '%s|%Y' -- {shlex.quote(remote_path)} && "
        f"cat -- {shlex.quote(remote_path)}"
    )
    result = subprocess.run(
        ["ssh", ssh_host, command],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"remote read failed for {request.source_relative_path}: {detail}")
    header, separator, raw_payload = result.stdout.partition(b"\n")
    if not separator:
        raise RuntimeError(f"missing stat header for {request.source_relative_path}")
    raw_size_text, raw_mtime_text = header.decode("ascii").split("|", 1)
    raw_size = int(raw_size_text)
    if raw_size != len(raw_payload):
        raise RuntimeError(
            f"raw size mismatch for {request.source_relative_path}: "
            f"stat={raw_size}, received={len(raw_payload)}"
        )

    raw_text = raw_payload.decode("utf-8", errors="replace")
    public_text, replacements = SANITIZER.sanitize_text(raw_text)
    findings = SANITIZER.audit_text(public_text, request.source_relative_path)
    if findings:
        raise RuntimeError(
            f"privacy audit failed for {request.source_relative_path}: {findings[:5]}"
        )
    public_payload = public_text.encode("utf-8")
    public_relative = (
        PurePosixPath(request.public_relative_path)
        if request.public_relative_path
        else PurePosixPath("reports") / request.source_relative_path
    )
    _validate_relative(public_relative.as_posix())
    public_path = output_root / Path(*public_relative.parts)
    public_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.write_bytes(public_payload)
    return {
        "kind": request.kind,
        "source_relative_path": request.source_relative_path,
        "public_relative_path": public_relative.as_posix(),
        "raw_size_bytes": raw_size,
        "raw_mtime_epoch": int(raw_mtime_text),
        "raw_sha256": _sha256(raw_payload),
        "public_size_bytes": len(public_payload),
        "public_sha256": _sha256(public_payload),
        "sanitizer_replacements": replacements,
    }


def _read_column(path: Path, column: str, kind: str) -> list[Request]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or column not in rows[0]:
        raise ValueError(f"missing column {column!r} in {path}")
    return [Request(kind, str(row[column])) for row in rows]


def _parse_report(value: str) -> Request:
    kind, separator, path = value.partition("=")
    if not separator or not kind or not path:
        raise argparse.ArgumentTypeError(
            "--report must use KIND=SOURCE_RELATIVE_PATH[@PUBLIC_RELATIVE_PATH]"
        )
    source, mapping_separator, public = path.partition("@")
    if mapping_separator and not public:
        raise argparse.ArgumentTypeError("mapped public path must not be empty")
    return Request(kind, source, public if mapping_separator else None)


def _write_manifests(output_root: Path, rows: list[dict[str, object]]) -> None:
    rows = sorted(rows, key=lambda row: (str(row["kind"]), str(row["source_relative_path"])))
    csv_path = output_root / "raw_public_manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    sha_path = output_root / "PUBLIC_MANIFEST.sha256"
    sha_path.write_text(
        "".join(
            f"{row['public_sha256']}  {row['public_relative_path']}\n" for row in rows
        ),
        encoding="utf-8",
        newline="",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssh-host", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sweep-csv", type=Path)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--report", action="append", default=[], type=_parse_report)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    requests: list[Request] = list(args.report)
    if args.sweep_csv:
        requests.extend(_read_column(args.sweep_csv, "timing_report", "timing_sweep"))
    if args.summary_csv:
        requests.extend(
            _read_column(args.summary_csv, "utilization_report", "selected_utilization")
        )
    unique: dict[str, Request] = {}
    for request in requests:
        previous = unique.get(request.source_relative_path)
        if previous and previous != request:
            raise ValueError(
                f"duplicate path with conflicting metadata: {request.source_relative_path}"
            )
        unique[request.source_relative_path] = request
    if not unique:
        raise SystemExit("no reports requested")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                _fetch_one,
                request,
                ssh_host=args.ssh_host,
                source_root=args.source_root,
                output_root=output_root,
            ): request
            for request in unique.values()
        }
        for future in as_completed(futures):
            request = futures[future]
            try:
                rows.append(future.result())
            except Exception as error:
                raise RuntimeError(f"failed to archive {request.source_relative_path}") from error

    _write_manifests(output_root, rows)
    findings = SANITIZER.audit(output_root)
    if findings:
        raise RuntimeError(f"output privacy audit failed: {findings[:20]}")
    print(
        f"REMOTE_EVIDENCE_ARCHIVE_PASS reports={len(rows)} "
        f"raw_hashes={len(rows)} public_hashes={len(rows)} privacy=PASS"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
