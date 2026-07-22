from __future__ import annotations

from contextlib import contextmanager

import torch

# The largest integer exactly representable by fp32.  W4A4 dot products in
# these runs stay below this bound, so fp32 matmul can emulate int accumulation.
FP32_EXACT_INT_LIMIT = 2**24
INT_GEMM_TMP_BUDGET_BYTES = 64 * 1024 * 1024
APPROX_BMM_TMP_BUDGET_BYTES = 256 * 1024 * 1024
APPROX_SUPPORTED_DTYPES = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}
APPROX_VARIANTS = {"lsb3_zero", "fixed_alpha", "wp521_unsigned_raw"}

# Low-rank factorization of the product error caused by zeroing the low 3 bits
# of each signed 8-bit product.  This implements the DSP-Packing simulator.
DELTA_LHS_BASIS = (
    (0, 0, 0, 0, 0, 0),
    (-1, -2, -3, -4, -5, -6),
    (-2, -4, -6, 0, -2, -4),
    (-3, -6, -1, -4, -7, -2),
    (-4, 0, -4, 0, -4, 0),
    (-5, -2, -7, -4, -1, -6),
    (-6, -4, -2, 0, -6, -4),
    (-7, -6, -5, -4, -3, -2),
)
DELTA_RHS_BASIS = (
    (0, 1, 0, 0, 0, 0, 0, -1),
    (0, 0, 1, 0, 0, 0, 0, 0),
    (0, 0, 0, 1, 0, 0, 0, 1),
    (0, 0, 0, 0, 1, 0, 0, 0),
    (0, 0, 0, 0, 0, 1, 0, 1),
    (0, 0, 0, 0, 0, 0, 1, 0),
)

_DELTA_FACTOR_CACHE: dict[tuple[str], tuple[torch.Tensor, torch.Tensor]] = {}


@contextmanager
def exact_fp32_gemm_mode():
    prev_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    cuda_prev = cudnn_prev = None
    if torch.cuda.is_available():
        cuda_prev = torch.backends.cuda.matmul.allow_tf32
        cudnn_prev = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(prev_precision)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = cuda_prev
            torch.backends.cudnn.allow_tf32 = cudnn_prev


def check_fp32_exact_range(lhs_q: torch.Tensor, rhs_q_t: torch.Tensor) -> None:
    max_abs = lhs_q.shape[1] * int(lhs_q.abs().max().item()) * int(rhs_q_t.abs().max().item())
    if max_abs >= FP32_EXACT_INT_LIMIT:
        raise ValueError(f"K={lhs_q.shape[1]} can exceed exact fp32 integer accumulation ({max_abs}).")


def choose_rhs_block_cols(m: int, k: int, n: int, device: torch.device) -> int:
    if device.type != "cuda":
        return n
    budget_elems = INT_GEMM_TMP_BUDGET_BYTES // 4
    lhs_elems = m * k
    if lhs_elems >= budget_elems:
        return min(n, 128)
    cols = (budget_elems - lhs_elems) // max(1, k + (2 * m))
    cols = min(n, max(64, cols))
    return max(1, (cols // 64) * 64) if cols < n else n


def exact_int_accum_gemm(lhs_q: torch.Tensor, rhs_q_t: torch.Tensor) -> torch.Tensor:
    if lhs_q.ndim != 2 or rhs_q_t.ndim != 2 or lhs_q.shape[1] != rhs_q_t.shape[0]:
        raise ValueError(f"Incompatible int GEMM shapes: {tuple(lhs_q.shape)} and {tuple(rhs_q_t.shape)}")
    check_fp32_exact_range(lhs_q, rhs_q_t)
    m, k = lhs_q.shape
    n = rhs_q_t.shape[1]
    block_cols = choose_rhs_block_cols(m, k, n, lhs_q.device)
    with exact_fp32_gemm_mode():
        lhs_fp32 = lhs_q.to(torch.float32)
        if block_cols >= n:
            return torch.round(lhs_fp32 @ rhs_q_t.to(torch.float32)).to(torch.int32)
        out = torch.empty((m, n), dtype=torch.int32, device=lhs_q.device)
        for start in range(0, n, block_cols):
            end = min(start + block_cols, n)
            out[:, start:end] = torch.round(lhs_fp32 @ rhs_q_t[:, start:end].to(torch.float32)).to(torch.int32)
    return out


def build_delta_rank_factors(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    key = (str(device),)
    if key not in _DELTA_FACTOR_CACHE:
        _DELTA_FACTOR_CACHE[key] = (
            torch.tensor(DELTA_LHS_BASIS, dtype=torch.int8, device=device),
            torch.tensor(DELTA_RHS_BASIS, dtype=torch.int8, device=device),
        )
    return _DELTA_FACTOR_CACHE[key]


def resolve_approx_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if dtype_name not in APPROX_SUPPORTED_DTYPES:
        raise ValueError(f"Unsupported approx accumulation dtype: {dtype_name}")
    dtype = APPROX_SUPPORTED_DTYPES[dtype_name]
    if device.type != "cuda" and dtype != torch.float32:
        return torch.float32
    return dtype


def choose_term_chunk_size(m: int, k: int, n: int, *, device: torch.device, dtype: torch.dtype) -> int:
    if device.type != "cuda":
        return 1
    elem_bytes = torch.empty((), dtype=dtype, device=device).element_size()
    per_term = (m * k + k * n + m * n) * elem_bytes
    return max(1, min(7, APPROX_BMM_TMP_BUDGET_BYTES // max(1, per_term)))


def lsb3_zero_accum_gemm(lhs_q: torch.Tensor, rhs_q_t: torch.Tensor, *, accum_dtype_name: str) -> torch.Tensor:
    accum_dtype = resolve_approx_dtype(accum_dtype_name, lhs_q.device)
    chunk = choose_term_chunk_size(
        lhs_q.shape[0], lhs_q.shape[1], rhs_q_t.shape[1], device=lhs_q.device, dtype=accum_dtype
    )
    lhs_basis, rhs_basis = build_delta_rank_factors(lhs_q.device)
    lhs_residue = torch.remainder(lhs_q.to(torch.int16), 8).to(torch.long)
    rhs_residue = torch.remainder(rhs_q_t.to(torch.int16), 8).to(torch.long)
    lhs_pending = [lhs_q.to(accum_dtype)]
    rhs_pending = [rhs_q_t.to(accum_dtype)]
    total = None
    for basis_idx in range(lhs_basis.shape[1]):
        lhs_pending.append(lhs_basis[:, basis_idx][lhs_residue].to(accum_dtype))
        rhs_pending.append(rhs_basis[basis_idx][rhs_residue].to(accum_dtype))
        if len(lhs_pending) >= chunk or basis_idx == lhs_basis.shape[1] - 1:
            partial = torch.bmm(torch.stack(lhs_pending), torch.stack(rhs_pending)).sum(dim=0).to(torch.float32)
            total = partial if total is None else total + partial
            lhs_pending.clear()
            rhs_pending.clear()
    return total


def fixed_alpha_accum_gemm(lhs_q: torch.Tensor, rhs_q_t: torch.Tensor, *, fixed_alpha: float) -> torch.Tensor:
    exact = exact_int_accum_gemm(lhs_q, rhs_q_t).to(torch.float32)
    return exact.add_(float(fixed_alpha) * float(lhs_q.shape[1]))


def approx_int_accum_gemm(
    lhs_q: torch.Tensor,
    rhs_q_t: torch.Tensor,
    *,
    variant: str,
    accum_dtype_name: str,
    fixed_alpha: float,
) -> torch.Tensor:
    if variant == "lsb3_zero":
        return lsb3_zero_accum_gemm(lhs_q, rhs_q_t, accum_dtype_name=accum_dtype_name)
    if variant == "fixed_alpha":
        return fixed_alpha_accum_gemm(lhs_q, rhs_q_t, fixed_alpha=fixed_alpha)
    if variant == "wp521_unsigned_raw":
        return exact_int_accum_gemm(lhs_q, rhs_q_t).to(torch.float32)
    raise ValueError(f"Unsupported approx variant: {variant}")


def quantized_linear(
    lhs_q: torch.Tensor,
    lhs_scale: torch.Tensor,
    rhs_q_t: torch.Tensor,
    rhs_scale: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    mode: str,
    lhs_group_size: int,
    output_dtype: torch.dtype,
    approx_accum_dtype: str = "bf16",
    approx_variant: str = "lsb3_zero",
    approx_fixed_alpha: float = 1.0,
) -> torch.Tensor:
    if mode not in {"exact", "approx"}:
        raise ValueError(f"Unsupported linear int mode: {mode}")
    if lhs_q.ndim != 2 or rhs_q_t.ndim != 2 or lhs_q.shape[1] != rhs_q_t.shape[0]:
        raise ValueError(f"Incompatible quantized linear shapes: {tuple(lhs_q.shape)} and {tuple(rhs_q_t.shape)}")
    if rhs_scale.ndim != 2 or rhs_scale.shape[1] != 1:
        raise ValueError("Only per-row weight scaling is supported.")
    if lhs_group_size <= 0 or lhs_q.shape[1] % lhs_group_size != 0:
        raise ValueError("lhs_group_size must divide the input dimension.")

    m, k = lhs_q.shape
    num_groups = k // lhs_group_size
    if lhs_scale.ndim == 2 and lhs_scale.shape[1] == 1:
        lhs_group_scales = lhs_scale
    else:
        lhs_group_scales = lhs_scale.reshape(m, num_groups, lhs_group_size)[..., 0]
    if lhs_group_scales.ndim != 2 or lhs_group_scales.shape[1] != num_groups:
        raise ValueError("Input scales are incompatible with lhs_group_size.")

    rhs_scale_t = rhs_scale.to(torch.float32).transpose(0, 1)
    out = torch.zeros((m, rhs_q_t.shape[1]), dtype=torch.float32, device=lhs_q.device)
    for group_idx in range(num_groups):
        start = group_idx * lhs_group_size
        end = start + lhs_group_size
        if mode == "exact":
            accum = exact_int_accum_gemm(lhs_q[:, start:end], rhs_q_t[start:end]).to(torch.float32)
        else:
            accum = approx_int_accum_gemm(
                lhs_q[:, start:end],
                rhs_q_t[start:end],
                variant=approx_variant,
                accum_dtype_name=approx_accum_dtype,
                fixed_alpha=approx_fixed_alpha,
            )
        scale = lhs_group_scales[:, group_idx : group_idx + 1].to(torch.float32) * rhs_scale_t
        out += accum * scale
    if bias is not None:
        out += bias.to(torch.float32).view(1, -1)
    return out.to(output_dtype)
