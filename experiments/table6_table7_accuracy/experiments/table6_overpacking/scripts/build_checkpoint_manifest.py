#!/usr/bin/env python3
"""Build the portable manifest for generated Table 6 checkpoints.

The checkpoint payloads are licensed/generated model data and are not copied
into the AE package. This helper streams each file once and records only its
model-relative evidence path, byte count, and SHA-256 digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


CHECKPOINTS = (
    ("llama2_7b", "model.bin"),
    ("llama2_7b", "qmodel.pt"),
    ("llama3_8b", "model.bin"),
    ("llama3_8b", "qmodel.pt"),
)
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}")
CHUNK_BYTES = 8 * 1024 * 1024


def hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_BYTES):
            digest.update(chunk)
            byte_count += len(chunk)
    return digest.hexdigest(), byte_count


def build_manifest(run_dir: Path, run_id: str) -> dict[str, object]:
    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError(
            "run_id must be 3-128 portable characters: letters, digits, '.', '_' or '-'"
        )

    records: list[dict[str, object]] = []
    for model, kind in CHECKPOINTS:
        relative = Path(model) / "train_w4a4" / kind
        checkpoint = run_dir / relative
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing checkpoint: {relative.as_posix()}")
        digest, byte_count = hash_file(checkpoint)
        records.append(
            {
                "model": model,
                "kind": kind,
                "path": relative.as_posix(),
                "sha256": digest,
                "bytes": byte_count,
            }
        )
    return {"run_id": run_id, "checkpoints": records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    manifest = build_manifest(args.run_dir, args.run_id)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"TABLE6_CHECKPOINT_MANIFEST_WRITTEN checkpoints={len(manifest['checkpoints'])} "
        f"run_id={manifest['run_id']}"
    )


if __name__ == "__main__":
    main()
