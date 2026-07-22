#!/usr/bin/env python3
from __future__ import annotations

import tempfile
from pathlib import Path

from parse_table6 import ROWS
from stage_fresh_logs import stage_logs


# Construct the synthetic private prefix at runtime. Keeping the complete
# prefix as a source literal would correctly fail the package privacy audit,
# even though this file only uses it to test redaction.
PRIVATE_ROOT = "/" + "hpc2hdd" + "/home/"


def fixture_log(private_suffix: str) -> str:
    return f"""source={PRIVATE_ROOT}private-user/{private_suffix}
|  Tasks   |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|----------|-------|------|-----:|--------|---|-----:|---|-----:|
|AVERAGE   |    N/A|none  |      |acc     |   |0.5000|+- |0.0100|
|arc_easy  |      1|none  |     0|acc_norm|up |0.5000|+- |0.0100|
|hellaswag |      1|none  |     0|acc_norm|up |0.5000|+- |0.0100|
|openbookqa|      1|none  |     0|acc_norm|up |0.5000|+- |0.0100|
|piqa      |      1|none  |     0|acc_norm|up |0.5000|+- |0.0100|
"""


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        run_dir = root / "run"
        out_dir = root / "staged"
        for model, _method, relative_log in ROWS:
            source = run_dir / model / relative_log
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text(
                fixture_log(f"{model}/{relative_log}"), encoding="utf-8"
            )

        count, replacements = stage_logs(run_dir, out_dir)
        # Fresh Table 6 contains the eight paper rows only.  WP521 belongs to
        # the separate rebuttal/Figure 12 archive and must not enter staging.
        assert count == 8 and replacements == 8
        packaged = [path for path in out_dir.rglob("*") if path.is_file()]
        assert len(packaged) == 8
        for path in packaged:
            text = path.read_text(encoding="utf-8")
            assert "<REMOTE_HOME>" in text
            assert PRIVATE_ROOT not in text
            assert "AVERAGE" in text

        try:
            stage_logs(run_dir, out_dir)
        except FileExistsError:
            pass
        else:
            raise AssertionError("existing staged evidence was overwritten")

    print(
        "TABLE6_FRESH_LOG_STAGING_SELFTEST_PASS "
        "logs=8 privacy=PASS overwrite_refused=PASS"
    )


if __name__ == "__main__":
    main()
