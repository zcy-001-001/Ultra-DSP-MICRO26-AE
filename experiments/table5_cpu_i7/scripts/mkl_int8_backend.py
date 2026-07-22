#!/usr/bin/env python3
"""Portable MKL INT8 GEMM backend for the Table 5 i7 rerun.

The historical experiment used the same ``gemm_s8u8s32`` interface with a
machine-specific DLL path.  That path is intentionally not retained in the AE
package.  This version resolves MKL from ``MKL_RT_PATH``, the active Conda
environment, or the Python environment.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
from pathlib import Path

import numpy as np


def _mkl_candidates() -> list[str]:
    candidates: list[str] = []
    explicit = os.environ.get("MKL_RT_PATH")
    if explicit:
        candidates.append(explicit)

    roots = [os.environ.get("CONDA_PREFIX"), sys.prefix]
    names = ("mkl_rt.2.dll", "mkl_rt.dll", "libmkl_rt.so", "libmkl_rt.dylib")
    for root in roots:
        if not root:
            continue
        base = Path(root)
        for directory in (base / "Library" / "bin", base / "bin", base / "lib"):
            for name in names:
                candidates.append(str(directory / name))

    discovered = ctypes.util.find_library("mkl_rt")
    if discovered:
        candidates.append(discovered)
    candidates.extend(names)
    return candidates


def _load_mkl():
    failures: list[str] = []
    for candidate in dict.fromkeys(_mkl_candidates()):
        try:
            library = ctypes.cdll.LoadLibrary(candidate)
            gemm = library.gemm_s8u8s32
            gemm.restype = None
            gemm.argtypes = [
                ctypes.c_char_p,
                ctypes.c_char_p,
                ctypes.c_char_p,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int8),
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int8),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_void_p,
            ]
            return gemm
        except (OSError, AttributeError) as exc:
            failures.append(f"{candidate}: {exc}")
    raise RuntimeError(
        "Intel MKL runtime was not found. Activate the intended Conda "
        "environment or set MKL_RT_PATH to mkl_rt. Last attempts:\n"
        + "\n".join(failures[-4:])
    )


_MKL_GEMM = _load_mkl()


class MKLInt8GEMM:
    """Resident-weight batch-1 INT8 GEMM using MKL ``gemm_s8u8s32``."""

    def __init__(self, m: int, k: int, n: int):
        self.m, self.k, self.n = m, k, n
        self.a = np.asfortranarray(
            np.random.randint(-128, 127, (m, k), dtype=np.int8)
        )
        b_s8 = np.random.randint(-128, 127, (k, n), dtype=np.int8)
        self.b_u8 = np.asfortranarray(
            (b_s8.astype(np.int16) + 128).astype(np.uint8)
        )
        self.c = np.zeros((m, n), dtype=np.int32, order="F")
        self.co = np.zeros(1, dtype=np.int32)

        self._m = ctypes.c_int(m)
        self._n = ctypes.c_int(n)
        self._k = ctypes.c_int(k)
        self._alpha = ctypes.c_float(1.0)
        self._beta = ctypes.c_float(0.0)
        self._lda = ctypes.c_int(m)
        self._ldb = ctypes.c_int(k)
        self._ldc = ctypes.c_int(m)
        self._ao = ctypes.c_int8(0)
        self._bo = ctypes.c_int8(-128)

        self._args = (
            b"N",
            b"N",
            b"F",
            ctypes.byref(self._m),
            ctypes.byref(self._n),
            ctypes.byref(self._k),
            ctypes.byref(self._alpha),
            self.a.ctypes.data_as(ctypes.c_void_p),
            ctypes.byref(self._lda),
            ctypes.byref(self._ao),
            self.b_u8.ctypes.data_as(ctypes.c_void_p),
            ctypes.byref(self._ldb),
            ctypes.byref(self._bo),
            ctypes.byref(self._beta),
            self.c.ctypes.data_as(ctypes.c_void_p),
            ctypes.byref(self._ldc),
            self.co.ctypes.data_as(ctypes.c_void_p),
        )

    def run(self) -> None:
        _MKL_GEMM(*self._args)

