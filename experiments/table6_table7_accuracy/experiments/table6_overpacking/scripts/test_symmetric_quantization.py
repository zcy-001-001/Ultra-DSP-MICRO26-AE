#!/usr/bin/env python3
"""Lightweight regression checks for the Table 6/7 symmetric convention."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_DIR))

from quant.quantizer import Quantizer  # noqa: E402
from utils.gptq_utils import WeightQuantizer, get_minq_maxq  # noqa: E402


def main() -> None:
    for bits in (3, 4, 5):
        expected_max = 2 ** (bits - 1) - 1
        expected_min = -expected_max

        activation = Quantizer(bits=bits, sym=True, narrow_symmetric=True)
        assert activation.qmin == expected_min
        assert activation.qmax == expected_max

        weight = WeightQuantizer()
        weight.configure(bits=bits, sym=True, narrow_symmetric=True)
        assert int(weight.minq) == expected_min
        assert int(weight.maxq) == expected_max

        minq, maxq = get_minq_maxq(bits, sym=True, narrow_symmetric=True)
        assert int(minq) == expected_min
        assert int(maxq) == expected_max

        probe = torch.tensor([[-100.0, -1.0, 0.0, 1.0, 100.0]])
        q, _, zero_point = activation.quantize_to_int(probe)
        assert int(q.min()) == expected_min
        assert int(q.max()) == expected_max
        assert torch.count_nonzero(zero_point) == 0

    runner = (Path(__file__).with_name("run_table6_full_regeneration.sh")).read_text(
        encoding="utf-8"
    )
    for required in (
        "--a_asym\nFalse",
        "--k_asym\nFalse",
        "--v_asym\nFalse",
        "--narrow_symmetric=True",
    ):
        assert required in runner, f"missing Table 6 runner flag: {required!r}"

    print("SYMMETRIC_QUANTIZATION_PASS bits=3,4,5")


if __name__ == "__main__":
    main()
