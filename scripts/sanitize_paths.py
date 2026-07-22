#!/usr/bin/env python3
"""Replace workstation and server home paths with AE-safe placeholders."""

from __future__ import annotations

import argparse
import gzip
import io
import os
import re
import tarfile
import tempfile
import zipfile
from pathlib import Path


TEXT_SUFFIXES = {
    ".cfg", ".csv", ".html", ".ipynb", ".json", ".jsonl", ".log", ".md",
    ".out", ".err", ".ps1", ".py", ".rpt", ".rst", ".sh", ".sv", ".tcl",
    ".toml", ".tsv", ".txt", ".v", ".xdc", ".xml", ".yaml", ".yml",
}
ARCHIVE_SUFFIXES = (".tgz", ".tar.gz", ".zip")

REPLACEMENTS = (
    (re.compile(r"(?m)^(\| Host\s*:\s*)\S+"), r"\1<REDACTED_HOST>"),
    # AE packages must not expose internal fully qualified hostnames or
    # account-qualified Windows-domain names from Vivado/Vitis logs. Generic
    # lab-host aliases used by the reproduction wrappers remain configurable.
    # Vivado logs and TensorBoard run names may contain an internal FPGA node
    # either as a short alias or FQDN, often preceded by an underscore.  Treat
    # both forms as private host metadata in the public artifact.
    (re.compile(r"(?i)(?<![a-z0-9])fpga\d+(?:\.[a-z0-9.-]+)?(?![a-z0-9])"), "<REMOTE_HOST>"),
    # TensorBoard run names commonly delimit a node with an underscore (for
    # example, a timestamp followed by ``_gpuN-M``).  Underscore is a regex
    # word character, so a plain ``\b`` misses that private host identifier.
    (re.compile(r"(?i)(?<![a-z0-9])gpu\d+(?:-\d+)?\.[a-z0-9.-]+(?![a-z0-9])"), "<REMOTE_HOST>"),
    (re.compile(r"(?i)(?<![a-z0-9])gpu\d+(?:-\d+)?(?![a-z0-9])"), "<REMOTE_HOST>"),
    (re.compile(r"(?m)^(SLURM_(?:ARRAY_)?JOB_ID=)\d+$"), r"\1<JOB_ID>"),
    (re.compile(r"(?i)\bCONNECT\\[^\\/\s\"']+"), "<REMOTE_USER>"),
    (re.compile(r"(?i)[A-Z]:[\\/]Users[\\/][^\\/\s\"']+[\\/]Desktop[\\/]MICRO[\\/](?:AE-1305|Rebuttal)"), "<AE_ROOT>"),
    (re.compile(r"(?i)[A-Z]:[\\/]Users[\\/][^\\/\s\"']+[\\/]Desktop[\\/]Ultra_DSP\.pdf"), "<AE_ROOT>/paper/Ultra_DSP.pdf"),
    (re.compile(r"(?i)[A-Z]:[\\/]Users[\\/][^\\/\s\"']+"), "<LOCAL_HOME>"),
    (re.compile(r"/data-hdd/home/CONNECT/[^/\s\"']+/data/(?:data/)?MICRO26"), "<REMOTE_WORKSPACE>"),
    (re.compile(r"/home/CONNECT/[^/\s\"']+/data/(?:data/)?MICRO26"), "<REMOTE_WORKSPACE>"),
    (re.compile(r"/data/user/[^/\s\"']+"), "<REMOTE_WORKSPACE>"),
    (re.compile(r"/tmp/[^/\s\"']+"), "<REMOTE_TMP>"),
    (re.compile(r"/hpc2hdd/home/[^/\s\"']+"), "<REMOTE_HOME>"),
    (re.compile(r"/data-hdd/home/CONNECT/[^/\s\"']+"), "<REMOTE_HOME>"),
    (re.compile(r"/home/CONNECT/[^/\s\"']+"), "<REMOTE_HOME>"),
    # Private storage roots can expose internal model, cache, tool, and run
    # layouts even when no account name appears in the path.  Preserve the
    # useful suffix while replacing the machine-specific absolute prefix.
    (re.compile(r"(?<![A-Za-z0-9+/])/data-hdd/"), "<REMOTE_STORAGE>/"),
    (re.compile(r"(?<![A-Za-z0-9+/])/hpc2hdd/"), "<REMOTE_STORAGE>/"),
    (re.compile(r"(?<![A-Za-z0-9+/])/scratch/"), "<REMOTE_STORAGE>/"),
    (re.compile(r"(?<![A-Za-z0-9+/])/workspaces?/"), "<REMOTE_STORAGE>/"),
    (re.compile(r"(?<![A-Za-z0-9+/])/work/"), "<REMOTE_STORAGE>/"),
    (re.compile(r"(?<![A-Za-z0-9+/])/mnt/"), "<REMOTE_STORAGE>/"),
    (re.compile(r"(?<![A-Za-z0-9+/])/root(?=/|\b)"), "<REMOTE_HOME>"),
    (re.compile(r"(?<![A-Za-z0-9+/])/home/(?!CONNECT/)[^/\s\"']+"), "<REMOTE_HOME>"),
)

FORBIDDEN = (
    re.compile(r"(?i)[A-Z]:[\\/]Users[\\/][^\\/\s\"']+"),
    re.compile(r"/home/CONNECT/[^/\s\"']+"),
    re.compile(r"/data-hdd/home/CONNECT/[^/\s\"']+"),
    re.compile(r"/data/user/[^/\s\"']+"),
    re.compile(r"/tmp/[^/\s\"']+"),
    re.compile(r"/hpc2hdd/home/[^/\s\"']+"),
    re.compile(r"(?i)(?<![a-z0-9])fpga\d+(?:\.[a-z0-9.-]+)?(?![a-z0-9])"),
    re.compile(r"(?i)(?<![a-z0-9])gpu\d+(?:-\d+)?\.[a-z0-9.-]+(?![a-z0-9])"),
    re.compile(r"(?i)(?<![a-z0-9])gpu\d+(?:-\d+)?(?![a-z0-9])"),
    re.compile(r"(?m)^SLURM_(?:ARRAY_)?JOB_ID=\d+$"),
    re.compile(r"(?i)\bCONNECT\\[^\\/\s\"']+"),
    re.compile(r"(?<![A-Za-z0-9+/])/data-hdd/"),
    re.compile(r"(?<![A-Za-z0-9+/])/hpc2hdd/"),
    re.compile(r"(?<![A-Za-z0-9+/])/scratch/"),
    re.compile(r"(?<![A-Za-z0-9+/])/workspaces?/"),
    re.compile(r"(?<![A-Za-z0-9+/])/work/"),
    re.compile(r"(?<![A-Za-z0-9+/])/mnt/"),
    re.compile(r"(?<![A-Za-z0-9+/])/root(?=/|\b)"),
    re.compile(r"(?<![A-Za-z0-9+/])/home/(?!CONNECT/)[^/\s\"']+"),
)


def iter_text_files(root: Path):
    this_script = Path(__file__).resolve()
    for path in root.rglob("*"):
        if (path.is_file() and path.suffix.lower() in TEXT_SUFFIXES
                and path.resolve() != this_script):
            yield path


def is_archive(path: Path) -> bool:
    lower = path.name.lower()
    return any(lower.endswith(suffix) for suffix in ARCHIVE_SUFFIXES)


def iter_archives(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and is_archive(path):
            yield path


def is_text_member(name: str) -> bool:
    return Path(name).suffix.lower() in TEXT_SUFFIXES


def sanitize_text(text: str) -> tuple[str, int]:
    updated = text
    replacements = 0
    for pattern, replacement in REPLACEMENTS:
        updated, count = pattern.subn(replacement, updated)
        replacements += count
    return updated, replacements


def sanitize_tar_archive(path: Path) -> int:
    """Rewrite sanitized text members without extracting archive paths."""
    replacement_count = 0
    changed = False
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as raw_output:
            temporary_path = Path(raw_output.name)
            # A fixed gzip timestamp keeps repeated sanitization deterministic.
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw_output, mtime=0) as compressed:
                with tarfile.open(path, "r:*") as source, tarfile.open(
                    fileobj=compressed, mode="w"
                ) as destination:
                    for member in source.getmembers():
                        stream = source.extractfile(member) if member.isfile() else None
                        if stream is not None:
                            with stream:
                                payload = stream.read()
                        else:
                            payload = None
                        if payload is not None and is_text_member(member.name):
                            text = payload.decode("utf-8", errors="replace")
                            updated, count = sanitize_text(text)
                            replacement_count += count
                            if count:
                                payload = updated.encode("utf-8")
                                member.size = len(payload)
                                changed = True
                        destination.addfile(
                            member, io.BytesIO(payload) if payload is not None else None
                        )
        if changed:
            os.replace(temporary_path, path)
            temporary_path = None
        return replacement_count
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def sanitize_zip_archive(path: Path) -> int:
    """Rewrite sanitized text members while retaining ZipInfo metadata."""
    replacement_count = 0
    changed = False
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as output:
            temporary_path = Path(output.name)
        with zipfile.ZipFile(path, "r") as source, zipfile.ZipFile(
            temporary_path, "w"
        ) as destination:
            for member in source.infolist():
                payload = source.read(member)
                if not member.is_dir() and is_text_member(member.filename):
                    text = payload.decode("utf-8", errors="replace")
                    updated, count = sanitize_text(text)
                    replacement_count += count
                    if count:
                        payload = updated.encode("utf-8")
                        changed = True
                destination.writestr(member, payload)
        if changed:
            os.replace(temporary_path, path)
            temporary_path = None
        return replacement_count
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def sanitize_archive(path: Path) -> int:
    if path.name.lower().endswith(".zip"):
        return sanitize_zip_archive(path)
    return sanitize_tar_archive(path)


def sanitize(root: Path) -> tuple[int, int]:
    changed = 0
    replacements = 0
    for path in iter_text_files(root):
        text = path.read_text(encoding="utf-8", errors="replace")
        updated, count = sanitize_text(text)
        replacements += count
        if updated != text:
            path.write_text(updated, encoding="utf-8", newline="")
            changed += 1
    for path in iter_archives(root):
        count = sanitize_archive(path)
        replacements += count
        if count:
            changed += 1
    return changed, replacements


def audit_text(text: str, location: str) -> list[str]:
    findings: list[str] = []
    for line_no, line in enumerate(text.splitlines(), 1):
        if any(pattern.search(line) for pattern in FORBIDDEN):
            findings.append(f"{location}:{line_no}")
    return findings


def audit_archive(path: Path, root: Path) -> list[str]:
    findings: list[str] = []
    archive_name = path.relative_to(root).as_posix()
    try:
        if path.name.lower().endswith(".zip"):
            with zipfile.ZipFile(path, "r") as archive:
                for member in archive.infolist():
                    if member.is_dir() or not is_text_member(member.filename):
                        continue
                    text = archive.read(member).decode("utf-8", errors="replace")
                    findings.extend(audit_text(text, f"{archive_name}!{member.filename}"))
        else:
            with tarfile.open(path, "r:*") as archive:
                for member in archive.getmembers():
                    if not member.isfile() or not is_text_member(member.name):
                        continue
                    stream = archive.extractfile(member)
                    if stream is None:
                        continue
                    text = stream.read().decode("utf-8", errors="replace")
                    findings.extend(audit_text(text, f"{archive_name}!{member.name}"))
    except (OSError, tarfile.TarError, zipfile.BadZipFile) as error:
        findings.append(f"{archive_name}:ARCHIVE_READ_ERROR:{type(error).__name__}")
    return findings


def audit(root: Path) -> list[str]:
    findings: list[str] = []
    for path in iter_text_files(root):
        text = path.read_text(encoding="utf-8", errors="replace")
        findings.extend(audit_text(text, path.relative_to(root).as_posix()))
    for path in iter_archives(root):
        findings.extend(audit_archive(path, root))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    if not args.check:
        changed, replacements = sanitize(root)
        print(f"SANITIZED files={changed} replacements={replacements}")

    findings = audit(root)
    if findings:
        print("PRIVACY_AUDIT_FAIL")
        for finding in findings[:100]:
            print(finding)
        return 1
    print("PRIVACY_AUDIT_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
