#!/usr/bin/env python3
"""Floating-point format metadata for the single-PE packing experiment."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FPFormat:
    name: str
    label: str
    fp_bits: int
    exponent_bits: int
    mantissa_bits: int
    product_width: int
    product_kind: str
    notes: str = ""

    @property
    def max_value(self) -> int:
        return (1 << self.product_width) - 1


FORMATS: tuple[FPFormat, ...] = (
    FPFormat(
        name="FP4_E2M1",
        label="FP4 E2M1 mag1",
        fp_bits=4,
        exponent_bits=2,
        mantissa_bits=1,
        product_width=1,
        product_kind="decoded_mantissa_mag",
        notes="Main rebuttal scope: FP4 E2M1 mantissa/magnitude bit only; exponent is outside the packed DSP datapath.",
    ),
    FPFormat(
        name="FP4_E1M2",
        label="FP4 E1M2 mag2",
        fp_bits=4,
        exponent_bits=1,
        mantissa_bits=2,
        product_width=2,
        product_kind="decoded_mantissa_mag",
        notes="Main rebuttal scope: FP4 E1M2 mantissa/magnitude bits only; exponent is outside the packed DSP datapath.",
    ),
    FPFormat(
        name="FP8_E4M3",
        label="FP8 E4M3 mag3",
        fp_bits=8,
        exponent_bits=4,
        mantissa_bits=3,
        product_width=3,
        product_kind="decoded_mantissa_mag",
        notes="Main rebuttal scope: FP8 E4M3 mantissa/magnitude bits only; exponent, rounding, and packing are wrapper logic.",
    ),
    FPFormat(
        name="FP8_E5M2",
        label="FP8 E5M2 mag2",
        fp_bits=8,
        exponent_bits=5,
        mantissa_bits=2,
        product_width=2,
        product_kind="decoded_mantissa_mag",
        notes="Main rebuttal scope: FP8 E5M2 mantissa/magnitude bits only; exponent, rounding, and packing are wrapper logic.",
    ),
    FPFormat(
        name="FP8_E3M4",
        label="FP8 E3M4 mag4",
        fp_bits=8,
        exponent_bits=3,
        mantissa_bits=4,
        product_width=4,
        product_kind="decoded_mantissa_mag",
        notes="Additional rebuttal sweep: FP8 E3M4 mantissa/magnitude bits only; exponent, rounding, and packing are wrapper logic.",
    ),
    FPFormat(
        name="FP8_E2M5",
        label="FP8 E2M5 mag5",
        fp_bits=8,
        exponent_bits=2,
        mantissa_bits=5,
        product_width=5,
        product_kind="decoded_mantissa_mag",
        notes="Additional rebuttal sweep: FP8 E2M5 mantissa/magnitude bits only; exponent, rounding, and packing are wrapper logic.",
    ),
    FPFormat(
        name="FP10_E4M5",
        label="E4M5 mag5",
        fp_bits=10,
        exponent_bits=4,
        mantissa_bits=5,
        product_width=5,
        product_kind="decoded_mantissa_mag_requested",
        notes=(
            "Requested E4M5 sweep. Including sign, E4M5 is a 10-bit encoding rather than FP8; "
            "the product backend still uses the M=5 mantissa/magnitude width."
        ),
    ),
    FPFormat(
        name="MXFP4_E2M1_MAG1",
        label="MXFP4 E2M1 mag1",
        fp_bits=4,
        exponent_bits=2,
        mantissa_bits=1,
        product_width=1,
        product_kind="mxfp4_e2m1_mantissa_mag",
        notes=(
            "Main rebuttal scope for MXFP4: private E2M1 mantissa/magnitude bit. "
            "The shared E8M0 scale and element exponent interpretation stay outside the packed DSP datapath."
        ),
    ),
)


def get_format(name: str) -> FPFormat:
    for fmt in FORMATS:
        if fmt.name == name:
            return fmt
    known = ", ".join(fmt.name for fmt in FORMATS)
    raise KeyError(f"Unknown format {name!r}. Known formats: {known}")


def iter_formats(names: list[str] | None = None) -> tuple[FPFormat, ...]:
    if not names:
        return FORMATS
    return tuple(get_format(name) for name in names)
