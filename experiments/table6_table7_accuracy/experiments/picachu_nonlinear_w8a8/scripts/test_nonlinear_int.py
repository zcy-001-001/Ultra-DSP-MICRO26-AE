#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_DIR))

from quant.nonlinear_int import get_nonlinear_backend  # noqa: E402


def assert_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise AssertionError(f"{name} produced NaN/Inf")


def test_backend(method: str, profile: str = "strict") -> None:
    torch.manual_seed(7)
    backend = get_nonlinear_backend(method, 8, profile)
    assert backend is not None

    x = torch.randn(2, 3, 16, dtype=torch.float32) * 2.0
    weight = torch.randn(16, dtype=torch.float32).abs() + 0.25
    rms = backend.rmsnorm(x, weight, 1e-5)
    assert rms.shape == x.shape
    assert_finite(f"{method}.rmsnorm", rms)

    logits = torch.randn(2, 4, 5, 9, dtype=torch.float32) * 3.0
    probs = backend.softmax(logits, -1)
    assert probs.shape == logits.shape
    assert_finite(f"{method}.softmax", probs)
    row_sum = probs.sum(dim=-1)
    if method == "gemmlowp" and profile in {"w8a8", "w8a8_lossy"}:
        if not torch.all((row_sum > 0.0) & (row_sum <= 1.05)):
            raise AssertionError(f"{method}.{profile}.softmax rows have invalid INT8 probability mass")
    elif not torch.allclose(row_sum, torch.ones_like(row_sum), atol=2e-2, rtol=0):
        raise AssertionError(f"{method}.softmax rows do not sum close to 1")

    silu = backend.silu(x)
    assert silu.shape == x.shape
    assert_finite(f"{method}.silu", silu)

    prod = backend.mul(x, torch.tanh(x))
    assert prod.shape == x.shape
    assert_finite(f"{method}.mul", prod)


def test_rope_stays_fp_path() -> None:
    source = (REPO_DIR / "quant" / "quant_ops.py").read_text()
    start = source.index("class QuantROPE")
    end = source.index("class QuantRMSNorm")
    rope_source = source[start:end]
    if "nonlinear_backend" in rope_source:
        raise AssertionError("QuantROPE should not be wired to nonlinear_backend when nonlinear_quant_rope=False")


def main() -> None:
    test_backend("ibert")
    for profile in (
        "strict",
        "w8a8",
        "w8a8_mid",
        "w8a8_lossy",
        "fixedpoint",
        "softmax_int8",
        "no_mul",
        "softmax_only",
        "rmsnorm_only",
        "silu_only",
        "mul_only",
        "no_rmsnorm",
        "softmax_silu",
        "softmax_silu_mul",
        "softmax_fp",
        "rmsnorm_noinput_only",
        "noinput_rmsnorm",
        "noinput_rmsnorm_out8",
    ):
        test_backend("gemmlowp", profile)
    test_rope_stays_fp_path()
    print("picachu nonlinear INT8 toy tests passed")


if __name__ == "__main__":
    main()
