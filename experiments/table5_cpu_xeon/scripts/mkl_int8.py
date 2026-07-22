"""
MKL-backed INT8 GEMV engine with memory-aware streaming weight banks.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import glob
import os
from typing import Iterable

import numpy as np

os.environ.setdefault("MKL_DYNAMIC", "FALSE")
os.environ.setdefault("OMP_PROC_BIND", "spread")
os.environ.setdefault("OMP_PLACES", "cores")


def _find_mkl() -> str | None:
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    search_paths = []
    if conda_prefix:
        search_paths.append(os.path.join(conda_prefix, "lib", "libmkl_rt.so"))
    home = os.path.expanduser("~")
    search_paths.append(os.path.join(home, "miniconda3", "lib", "libmkl_rt.so"))
    search_paths.append(os.path.join(home, "anaconda3", "lib", "libmkl_rt.so"))
    search_paths.extend(
        [
            "/usr/lib/x86_64-linux-gnu/libmkl_rt.so",
            "/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_rt.so",
        ]
    )
    found = ctypes.util.find_library("mkl_rt")
    if found:
        search_paths.append(found)

    for path in search_paths:
        if os.path.isfile(path):
            return path
    return None


def _load_mkl():
    mkl_path = _find_mkl()
    if mkl_path is None:
        raise RuntimeError(
            "MKL not found. Install via conda or expose libmkl_rt.so in LD_LIBRARY_PATH."
        )

    mkl = ctypes.cdll.LoadLibrary(mkl_path)
    gemm = mkl.gemm_s8u8s32_
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
    return gemm, mkl_path, mkl


_MKL_GEMM, MKL_PATH, _MKL_LIB = _load_mkl()


def set_mkl_threads(num_threads: int):
    try:
        set_threads = _MKL_LIB.MKL_Set_Num_Threads
        set_threads.restype = None
        set_threads.argtypes = [ctypes.c_int]
        set_threads(num_threads)
    except Exception as exc:
        raise RuntimeError(f"Failed to set MKL threads: {exc}") from exc

    try:
        set_dynamic = _MKL_LIB.MKL_Set_Dynamic
        set_dynamic.restype = None
        set_dynamic.argtypes = [ctypes.c_int]
        set_dynamic(0)
    except Exception:
        pass


def parse_cpu_list(cpu_list: str) -> list[int]:
    cpus = []
    for part in cpu_list.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            cpus.extend(range(int(start), int(end) + 1))
        else:
            cpus.append(int(part))
    return cpus


def get_numa_cpu_sets() -> list[set[int]]:
    cpu_sets = []
    for path in sorted(glob.glob("/sys/devices/system/node/node*/cpulist")):
        try:
            with open(path) as f:
                cpu_list = f.read().strip()
        except Exception:
            continue
        parsed = parse_cpu_list(cpu_list)
        if parsed:
            cpu_sets.append(set(parsed))
    return cpu_sets


def numa_balancing_enabled() -> bool:
    try:
        with open("/proc/sys/kernel/numa_balancing") as f:
            return f.read().strip() == "1"
    except Exception:
        return False


def tensor_bytes_mb(num_bytes: int) -> float:
    return num_bytes / (1024.0 * 1024.0)


def choose_pool_size(weight_bytes: int, streaming_mb: int, min_pool_size: int, max_pool_size: int) -> int:
    target_bytes = streaming_mb * 1024 * 1024
    pool_size = -(-target_bytes // weight_bytes)
    return max(min_pool_size, min(max_pool_size, int(pool_size)))


class MKLInt8GEMVStream:
    """Stream across a bank of INT8 weights to force memory-bound GEMV."""

    def __init__(
        self,
        M: int,
        K: int,
        N: int,
        pool_size: int,
        seed: int = 0,
        numa_first_touch: bool = True,
    ):
        self.M = M
        self.K = K
        self.N = N
        self.pool_size = pool_size
        self.rng = np.random.default_rng(seed)
        self.order = np.arange(pool_size, dtype=np.int32)
        self.rng.shuffle(self.order)
        self._order_idx = 0

        self.A = np.asfortranarray(
            self.rng.integers(-128, 128, size=(M, K), dtype=np.int16).astype(np.int8)
        )
        self.C = np.zeros((M, N), dtype=np.int32, order="F")
        self.co = np.zeros(1, dtype=np.int32)

        self._m = ctypes.c_int(M)
        self._n = ctypes.c_int(N)
        self._k = ctypes.c_int(K)
        self._alpha = ctypes.c_float(1.0)
        self._beta = ctypes.c_float(0.0)
        self._lda = ctypes.c_int(M)
        self._ldb = ctypes.c_int(K)
        self._ldc = ctypes.c_int(M)
        self._ao = ctypes.c_int8(0)
        self._bo = ctypes.c_int8(-128)

        self._a_ptr = self.A.ctypes.data_as(ctypes.c_void_p)
        self._c_ptr = self.C.ctypes.data_as(ctypes.c_void_p)
        self._co_ptr = self.co.ctypes.data_as(ctypes.c_void_p)

        self.numa_cpu_sets = get_numa_cpu_sets() if numa_first_touch else []
        self.weights_s8, self.weights_u8, self._args_bank = self._build_weight_bank()

    def _build_weight_bank(self):
        weights_s8 = []
        weights_u8 = []
        args_bank = []
        original_affinity = None

        try:
            original_affinity = os.sched_getaffinity(0)
        except Exception:
            original_affinity = None

        for idx in range(self.pool_size):
            if original_affinity is not None and self.numa_cpu_sets:
                try:
                    os.sched_setaffinity(0, self.numa_cpu_sets[idx % len(self.numa_cpu_sets)])
                except Exception:
                    pass

            b_s8 = self.rng.integers(-128, 128, size=(self.K, self.N), dtype=np.int16).astype(np.int8)
            b_u8 = np.asfortranarray((b_s8.astype(np.int16) + 128).astype(np.uint8))

            weights_s8.append(b_s8)
            weights_u8.append(b_u8)
            args_bank.append(
                (
                    b"N",
                    b"N",
                    b"F",
                    ctypes.byref(self._m),
                    ctypes.byref(self._n),
                    ctypes.byref(self._k),
                    ctypes.byref(self._alpha),
                    self._a_ptr,
                    ctypes.byref(self._lda),
                    ctypes.byref(self._ao),
                    b_u8.ctypes.data_as(ctypes.c_void_p),
                    ctypes.byref(self._ldb),
                    ctypes.byref(self._bo),
                    ctypes.byref(self._beta),
                    self._c_ptr,
                    ctypes.byref(self._ldc),
                    self._co_ptr,
                )
            )

        if original_affinity is not None:
            try:
                os.sched_setaffinity(0, original_affinity)
            except Exception:
                pass

        return tuple(weights_s8), tuple(weights_u8), tuple(args_bank)

    def run(self):
        bank_idx = int(self.order[self._order_idx])
        self.run_bank_index(bank_idx)
        self._order_idx += 1
        if self._order_idx == self.pool_size:
            self._order_idx = 0
            self.rng.shuffle(self.order)

    def run_bank_index(self, bank_idx: int):
        _MKL_GEMM(*self._args_bank[bank_idx])

    def reference(self, bank_index: int = 0):
        return self.A.astype(np.int32) @ self.weights_s8[bank_index].astype(np.int32)

    @property
    def weight_bytes_mb(self) -> float:
        return round(tensor_bytes_mb(self.K * self.N), 2)

    @property
    def working_set_mb(self) -> float:
        return round(tensor_bytes_mb(self.pool_size * self.K * self.N), 2)


def print_system_info():
    print(f"  MKL runtime: {MKL_PATH}")
    print(f"  CPU count: {os.cpu_count()}")
    print(f"  NUMA balancing: {'enabled' if numa_balancing_enabled() else 'disabled'}")
    cpu_sets = get_numa_cpu_sets()
    if cpu_sets:
        summary = ", ".join(f"node{i}={len(cpu_set)} CPUs" for i, cpu_set in enumerate(cpu_sets))
        print(f"  NUMA nodes: {summary}")
