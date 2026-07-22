#!/usr/bin/env python3
"""Dependency-free audit of the Table 6/7 narrow-symmetric wiring."""

from __future__ import annotations

from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[3]


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise AssertionError(f"missing {label}: {needle!r}")


def main() -> None:
    table6_runner = (REPO_DIR / "experiments/table6_overpacking/scripts/run_table6_full_regeneration.sh").read_text(encoding="utf-8")
    table7_runner = (REPO_DIR / "experiments/mixed_precision_ultradsp/scripts/run_one_precision_ultradsp.sh").read_text(encoding="utf-8")
    quantizer = (REPO_DIR / "quant/quantizer.py").read_text(encoding="utf-8")
    gptq = (REPO_DIR / "utils/gptq_utils.py").read_text(encoding="utf-8")

    for runner, label in ((table6_runner, "Table 6"), (table7_runner, "Table 7")):
        for flag in ("--a_asym\nFalse", "--k_asym\nFalse", "--v_asym\nFalse", "--narrow_symmetric=True"):
            require(runner, flag, f"{label} symmetric flag")

    require(quantizer, "self.qmin = -self.qmax if narrow_symmetric else -(2**(bits - 1))", "activation range")
    require(gptq, "minq = -maxq if narrow_symmetric else -maxq -1", "GPTQ range")
    require(gptq, 'narrow_symmetric=getattr(args, "narrow_symmetric", False)', "GPTQ propagation")
    print("SYMMETRIC_CONFIG_STATIC_PASS tables=6,7")


if __name__ == "__main__":
    main()
