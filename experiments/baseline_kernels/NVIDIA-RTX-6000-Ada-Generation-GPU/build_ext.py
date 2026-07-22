#!/usr/bin/env python3
"""JIT-compile the CUTLASS GEMV extensions."""

import os
from functools import lru_cache
from pathlib import Path

from torch.utils.cpp_extension import CUDA_HOME as TORCH_CUDA_HOME
from torch.utils.cpp_extension import load

_HERE = Path(__file__).resolve().parent
_DEFAULT_CUTLASS_ROOT = (_HERE / ".." / "cutlass").resolve()
_CUTLASS_ROOT = Path(os.environ.get("CUTLASS_ROOT", _DEFAULT_CUTLASS_ROOT)).resolve()
_BUILD_ROOT = _HERE / ".torch_extensions"


def _cutlass_include_paths() -> list[str]:
    """Return CUTLASS include paths for either bundled or upstream layouts."""
    if (_CUTLASS_ROOT / "cutlass").is_dir() and (_CUTLASS_ROOT / "cute").is_dir():
        cutlass_inc = _CUTLASS_ROOT
    elif (_CUTLASS_ROOT / "include" / "cutlass").is_dir():
        cutlass_inc = _CUTLASS_ROOT / "include"
    else:
        raise RuntimeError(
            "CUTLASS headers not found. Use bundled ../cutlass or set "
            "CUTLASS_ROOT to an upstream CUTLASS checkout."
        )

    cutlass_tools = _CUTLASS_ROOT / "tools" / "util" / "include"
    if not cutlass_tools.is_dir():
        raise RuntimeError(
            "CUTLASS utility headers not found under tools/util/include. "
            "Set CUTLASS_ROOT to a complete CUTLASS checkout."
        )
    return [str(cutlass_inc), str(cutlass_tools)]


def _configure_cuda_env():
    candidates = [
        os.environ.get("CUDA_HOME"),
        os.environ.get("CUDA_PATH"),
        TORCH_CUDA_HOME,
    ]

    for candidate in candidates:
        if not candidate:
            continue
        nvcc = Path(candidate) / "bin" / "nvcc"
        if nvcc.exists():
            os.environ["CUDA_HOME"] = str(Path(candidate))
            path_entries = os.environ.get("PATH", "").split(os.pathsep)
            nvcc_bin = str(nvcc.parent)
            if nvcc_bin not in path_entries:
                os.environ["PATH"] = nvcc_bin + os.pathsep + os.environ.get("PATH", "")
            os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
            return

    raise RuntimeError("Could not find nvcc. Set CUDA_HOME before building the GEMV extensions.")


def _load_ext(name: str, source_name: str):
    _configure_cuda_env()
    build_dir = _BUILD_ROOT / name
    build_dir.mkdir(parents=True, exist_ok=True)

    return load(
        name=name,
        sources=[str(_HERE / source_name)],
        extra_include_paths=_cutlass_include_paths(),
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "-arch=sm_89",
            "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
            "--expt-relaxed-constexpr",
        ],
        build_directory=str(build_dir),
        verbose=True,
    )


@lru_cache(maxsize=1)
def get_int8_ext():
    return _load_ext("gemv_cutlass_int8_ext", "cutlass_int8_gemv.cu")


@lru_cache(maxsize=1)
def get_int4_ext():
    return _load_ext("gemv_cutlass_int4_ext", "cutlass_int4_gemv.cu")


if __name__ == "__main__":
    get_int8_ext()
    get_int4_ext()
    print("GEMV extensions compiled successfully.")
