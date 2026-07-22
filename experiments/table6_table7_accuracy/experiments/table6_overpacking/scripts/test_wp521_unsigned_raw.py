#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_DIR))

from quant.approx_linear import quantized_linear  # noqa: E402


def assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    if not torch.allclose(actual, expected):
        raise AssertionError(f"mismatch\nactual={actual}\nexpected={expected}")


def main() -> None:
    lhs_raw = torch.tensor([[0, 1, 7, 15], [15, 0, 3, 4]], dtype=torch.int8)
    lhs_centered = (lhs_raw.to(torch.int16) - 8).to(torch.int8)
    rhs_centered = torch.tensor([[7, -8], [-3, 2], [1, -1], [-8, 7]], dtype=torch.int8)
    lhs_scale = torch.ones((lhs_raw.shape[0], 1), dtype=torch.float32)
    rhs_scale = torch.ones((rhs_centered.shape[1], 1), dtype=torch.float32)

    wp521 = quantized_linear(
        lhs_raw,
        lhs_scale,
        rhs_centered,
        rhs_scale,
        None,
        mode="approx",
        lhs_group_size=lhs_raw.shape[1],
        output_dtype=torch.float32,
        approx_variant="wp521_unsigned_raw",
    )
    expected_wp521 = (lhs_raw.to(torch.int32) @ rhs_centered.to(torch.int32)).to(torch.float32)
    assert_close(wp521, expected_wp521)

    exact = quantized_linear(
        lhs_centered,
        lhs_scale,
        rhs_centered,
        rhs_scale,
        None,
        mode="exact",
        lhs_group_size=lhs_raw.shape[1],
        output_dtype=torch.float32,
    )
    expected_exact = (lhs_centered.to(torch.int32) @ rhs_centered.to(torch.int32)).to(torch.float32)
    assert_close(exact, expected_exact)
    if torch.equal(wp521, exact):
        raise AssertionError("WP521 raw path unexpectedly matches centered exact path")

    # The formal fresh Table 6 rerun uses narrow-symmetric A4 integers with
    # zero-point 0.  In that configuration the WP521 reference-only branch
    # receives the already centered signed code, so preserving x_q must
    # intentionally collapse to the exact integer accumulator.  Keep this
    # separate from the archived raw-unsigned [0, 15] case above.
    symmetric_wp521 = quantized_linear(
        lhs_centered,
        lhs_scale,
        rhs_centered,
        rhs_scale,
        None,
        mode="approx",
        lhs_group_size=lhs_centered.shape[1],
        output_dtype=torch.float32,
        approx_variant="wp521_unsigned_raw",
    )
    assert_close(symmetric_wp521, exact)

    fixed_alpha = quantized_linear(
        lhs_centered,
        lhs_scale,
        rhs_centered,
        rhs_scale,
        None,
        mode="approx",
        lhs_group_size=lhs_raw.shape[1],
        output_dtype=torch.float32,
        approx_variant="fixed_alpha",
        approx_fixed_alpha=0.5,
    )
    assert_close(fixed_alpha, expected_exact + 0.5 * lhs_raw.shape[1])

    lsb3_zero = quantized_linear(
        lhs_centered,
        lhs_scale,
        rhs_centered,
        rhs_scale,
        None,
        mode="approx",
        lhs_group_size=lhs_raw.shape[1],
        output_dtype=torch.float32,
        approx_variant="lsb3_zero",
        approx_accum_dtype="fp32",
    )
    if lsb3_zero.shape != expected_exact.shape or not torch.isfinite(lsb3_zero).all():
        raise AssertionError("lsb3_zero regression check failed")

    print("WP521 unsigned raw and symmetric reference-only checks passed.")


if __name__ == "__main__":
    main()
