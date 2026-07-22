from __future__ import annotations

import csv
import hashlib
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import simulate_figures as simulator


def main() -> None:
    summary = simulator.reproduce()
    expected = {
        "figure14_normalized_speedup": 2.31,
        "figure14_normalized_energy_efficiency": 2.22,
        "figure15_normalized_speedup": 1.88,
        "figure15_normalized_energy_efficiency": 1.88,
    }
    for key, paper_rounded in expected.items():
        actual = summary[key]
        assert round(actual, 2) == paper_rounded, (key, actual, paper_rounded)

    assert 0.0 < summary["raw_hls_matrix_fraction"] < 1.0
    assert 0.0 < summary["calibrated_prefill_matrix_fraction"] < 1.0
    assert not math.isclose(
        summary["raw_hls_matrix_fraction"],
        summary["calibrated_prefill_matrix_fraction"],
        rel_tol=1e-3,
    ), "Raw HLS and calibrated fractions must remain distinguishable"

    with (simulator.RESULT_DIR / "figure14_canonical.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    for required in ("model_value", "calibrated_value", "paper_anchor_value", "source_kind"):
        assert required in rows[0]

    source_root = ROOT / "hls_source_minimal"
    manifest = source_root / "SOURCE_MANIFEST.sha256"
    checked_sources = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        expected_hash, relative_path = line.split(maxsplit=1)
        source_path = source_root / relative_path
        actual_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        assert actual_hash == expected_hash, relative_path
        checked_sources += 1
    assert checked_sources == 9

    # Build tokens from pieces so this privacy test does not flag its own source.
    forbidden = [
        "/" + "home" + "/",
        "\\" + "Users" + "\\",
        "C" + ":\\",
        "cz" + "hang",
        "CON" + "NECT" + "\\",
    ]
    for path in ROOT.rglob("*"):
        if (
            not path.is_file()
            or "__pycache__" in path.parts
            or path.suffix.lower() in {".png", ".pdf", ".pyc"}
        ):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in forbidden:
            assert token.lower() not in text.lower(), (path.name, token)

    print("FIGURES14_15_TEST_PASS")


if __name__ == "__main__":
    main()
