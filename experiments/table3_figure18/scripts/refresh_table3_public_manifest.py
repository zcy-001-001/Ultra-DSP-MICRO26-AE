#!/usr/bin/env python3
"""Refresh only the sanitized-copy fields in the Table 3 evidence manifest.

The raw remote hashes are immutable provenance.  This helper is used after a
package-wide privacy pass changes a public placeholder; it never contacts a
remote host or changes report content.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent.parent
EVIDENCE = HERE / "evidence/table3"
MANIFEST = EVIDENCE / "manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    data["sanitization"] = "Vivado Host header replaced with <REDACTED_HOST>"
    entries = data["entries"]
    for entry in entries:
        relative = Path(entry["public_relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe public path: {relative}")
        report = EVIDENCE / relative
        if not report.is_file():
            raise FileNotFoundError(report)
        entry["public_sha256"] = sha256(report)
        entry["public_bytes"] = report.stat().st_size
    data["entry_count"] = len(entries)
    MANIFEST.write_text(
        json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"TABLE3_PUBLIC_MANIFEST_REFRESHED entries={len(entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
