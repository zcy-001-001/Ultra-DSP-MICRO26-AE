#!/usr/bin/env python3
"""Exhaustively check the corrected V0 normal signed packing arithmetic."""

from __future__ import annotations

from itertools import product


def sx(value: int, bits: int) -> int:
    value &= (1 << bits) - 1
    sign = 1 << (bits - 1)
    return value - (1 << bits) if value & sign else value


def decode_v0_lanes(w1: int, w2: int, b: int) -> tuple[int, int]:
    packed = w1 * b + ((w2 * b) << 8)
    lane0 = sx(packed, 8)
    lane1_polluted = sx(packed >> 8, 8)
    lane1 = lane1_polluted + (1 if lane0 < 0 else 0)
    return lane0, lane1


def main() -> None:
    values = range(-7, 8)
    errors = []
    for mode in (0, 1):
        for w1, w2, a1, a2 in product(values, values, values, values):
            b = a1 if mode else a1 + a2
            got = decode_v0_lanes(w1, w2, b)
            ref = (w1 * b, w2 * b)
            if got != ref:
                errors.append((mode, w1, w2, a1, a2, b, got, ref))
                break
        if errors:
            break

    if errors:
        print("V0 normal signed check failed:")
        print(errors[0])
        raise SystemExit(1)

    print("V0 normal signed lane correction is exact for [-7, 7] in both P and D modes.")


if __name__ == "__main__":
    main()
