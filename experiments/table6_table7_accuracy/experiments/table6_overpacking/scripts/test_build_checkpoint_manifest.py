#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

from build_checkpoint_manifest import CHECKPOINTS, build_manifest


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = Path(temporary) / "run"
        expected: dict[tuple[str, str], bytes] = {}
        for index, (model, kind) in enumerate(CHECKPOINTS, 1):
            payload = (f"{model}/{kind}\n".encode("utf-8") + bytes(range(index))) * index
            checkpoint = run_dir / model / "train_w4a4" / kind
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(payload)
            expected[(model, kind)] = payload

        manifest = build_manifest(run_dir, "table6-manifest-selftest")
        assert manifest["run_id"] == "table6-manifest-selftest"
        records = manifest["checkpoints"]
        assert isinstance(records, list) and len(records) == 4
        for record in records:
            key = str(record["model"]), str(record["kind"])
            payload = expected[key]
            assert record["path"] == f"{key[0]}/train_w4a4/{key[1]}"
            assert record["bytes"] == len(payload)
            assert record["sha256"] == hashlib.sha256(payload).hexdigest()

        try:
            build_manifest(run_dir, "private/run/id")
        except ValueError:
            pass
        else:
            raise AssertionError("non-portable run_id was accepted")

    print("TABLE6_CHECKPOINT_MANIFEST_SELFTEST_PASS checkpoints=4 paths=relative")


if __name__ == "__main__":
    main()
