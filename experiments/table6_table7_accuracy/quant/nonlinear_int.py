import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


def _reduce_for_last_dim(x: torch.Tensor) -> Tuple[int, ...]:
    if x.dim() <= 1:
        return (0,)
    return (-1,)


def _reduce_for_all_dims(x: torch.Tensor) -> Tuple[int, ...]:
    if x.dim() == 0:
        return (0,)
    return tuple(range(x.dim()))


def _signed_quantize(
    x: torch.Tensor,
    bits: int,
    reduce_dim: Optional[Tuple[int, ...]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x_f = torch.nan_to_num(x.float(), nan=0.0, posinf=1.0e4, neginf=-1.0e4)
    qmax = (2 ** (bits - 1)) - 1
    qmin = -(2 ** (bits - 1))
    if reduce_dim is None:
        reduce_dim = _reduce_for_last_dim(x_f)
    max_abs = torch.amax(x_f.abs(), dim=reduce_dim, keepdim=True).clamp(min=1e-8)
    scale = max_abs / qmax
    q = torch.round(x_f / scale).clamp(qmin, qmax)
    return q.to(torch.int32), scale


def _signed_fake_quant(
    x: torch.Tensor,
    bits: int,
    reduce_dim: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    dtype = x.dtype
    q, scale = _signed_quantize(x, bits, reduce_dim)
    out = q.float() * scale
    return torch.nan_to_num(out, nan=0.0, posinf=1.0e4, neginf=-1.0e4).to(dtype)


def _signed_fake_quant_all_dims(x: torch.Tensor, bits: int) -> torch.Tensor:
    return _signed_fake_quant(x, bits, reduce_dim=_reduce_for_all_dims(x))


def _unit_fake_quant(x: torch.Tensor, bits: int) -> torch.Tensor:
    dtype = x.dtype
    qmax = (2**bits) - 1
    x_f = torch.nan_to_num(x.float(), nan=0.0, posinf=1.0, neginf=0.0)
    q = torch.round(x_f.clamp(0.0, 1.0) * qmax).clamp(0, qmax)
    return torch.nan_to_num(q / qmax, nan=0.0, posinf=1.0, neginf=0.0).to(dtype)


def _renorm_probs(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_f = torch.nan_to_num(x.float(), nan=0.0, posinf=1.0, neginf=0.0)
    denom = x_f.sum(dim=dim, keepdim=True).clamp(min=1e-12)
    return torch.nan_to_num(x_f / denom, nan=0.0, posinf=1.0, neginf=0.0).to(x.dtype)


def _fixed_round(x: torch.Tensor, frac_bits: int = 12, clamp_abs: float = 32.0) -> torch.Tensor:
    scale = float(1 << frac_bits)
    x_f = x.float().clamp(-clamp_abs, clamp_abs)
    return torch.round(x_f * scale) / scale


@dataclass(frozen=True)
class NonlinearIntConfig:
    method: str = "disabled"
    bits: int = 8
    profile: str = "strict"


GEMMLOWP_PROFILES = {
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
}

GEMMLOWP_PROFILE_OPS = {
    # The first four profiles preserve the initial coarse experiment modes.
    "strict": {"rmsnorm", "softmax", "silu", "mul"},
    "w8a8": {"rmsnorm", "softmax", "silu", "mul"},
    # Middle-ground W8A8 profile for the final Gemmlowp row.  The previous
    # w8a8 profile deliberately keeps the rejected all-tensor/no-renorm path as
    # a diagnostic; this profile uses per-token dynamic scales for activation
    # tensors and renormalizes INT8 softmax probabilities.
    "w8a8_mid": {"rmsnorm", "softmax", "silu", "mul"},
    # Slightly stronger middle profile: still uses per-token activation scales,
    # but leaves the INT8 softmax probability mass un-repaired.  This keeps the
    # rejected all-tensor profile separate while allowing a less optimistic row.
    "w8a8_lossy": {"rmsnorm", "softmax", "silu", "mul"},
    "fixedpoint": {"rmsnorm", "softmax", "silu", "mul"},
    "softmax_int8": {"rmsnorm", "softmax", "silu", "mul"},
    "no_mul": {"rmsnorm", "softmax", "silu", "mul"},
    # Diagnostic profiles isolate the nonlinear wrappers.  They keep the old FP
    # implementation for every op not listed here instead of deleting that path.
    "softmax_only": {"softmax"},
    "rmsnorm_only": {"rmsnorm"},
    "silu_only": {"silu"},
    "mul_only": {"mul"},
    "no_rmsnorm": {"softmax", "silu", "mul"},
    "softmax_silu": {"softmax", "silu"},
    "softmax_silu_mul": {"softmax", "silu", "mul"},
    "softmax_fp": {"rmsnorm", "silu", "mul"},
    "rmsnorm_noinput_only": {"rmsnorm"},
    "noinput_rmsnorm": {"rmsnorm", "softmax", "silu", "mul"},
    "noinput_rmsnorm_out8": {"rmsnorm", "softmax", "silu", "mul"},
}

GEMMLOWP_RMSNORM_NO_INPUT_QUANT = {
    "rmsnorm_noinput_only",
    "noinput_rmsnorm",
    "noinput_rmsnorm_out8",
}


class NonlinearIntBackend:
    def __init__(self, config: NonlinearIntConfig):
        self.config = config
        self.method = config.method
        self.bits = config.bits
        self.profile = config.profile

    @property
    def enabled(self) -> bool:
        return self.method != "disabled"

    def rmsnorm(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        if self.method == "ibert":
            return self._ibert_rmsnorm(hidden_states, weight, eps)
        if self.method == "gemmlowp":
            if not self._gemmlowp_uses("rmsnorm"):
                return self._fp_rmsnorm(hidden_states, weight, eps)
            return self._gemmlowp_rmsnorm(hidden_states, weight, eps)
        raise ValueError(f"Unknown nonlinear int method: {self.method}")

    def softmax(
        self,
        attn_weights: torch.Tensor,
        dim: int,
    ) -> torch.Tensor:
        if self.method == "ibert":
            return self._ibert_softmax(attn_weights, dim)
        if self.method == "gemmlowp":
            if not self._gemmlowp_uses("softmax"):
                return self._fp_softmax(attn_weights, dim)
            return self._gemmlowp_softmax(attn_weights, dim)
        raise ValueError(f"Unknown nonlinear int method: {self.method}")

    def silu(self, x: torch.Tensor, smooth: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.method == "ibert":
            return self._ibert_silu(x, smooth)
        if self.method == "gemmlowp":
            if not self._gemmlowp_uses("silu"):
                return self._fp_silu(x, smooth)
            return self._gemmlowp_silu(x, smooth)
        raise ValueError(f"Unknown nonlinear int method: {self.method}")

    def mul(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.method == "ibert":
            return self._int8_mul(x1, x2, accum_frac_bits=8)
        if self.method == "gemmlowp":
            if not self._gemmlowp_uses("mul"):
                return self._fp_mul(x1, x2)
            return self._int8_mul(x1, x2, accum_frac_bits=12)
        raise ValueError(f"Unknown nonlinear int method: {self.method}")

    def _gemmlowp_uses(self, op_name: str) -> bool:
        return op_name in GEMMLOWP_PROFILE_OPS[self.profile]

    def _gemmlowp_keeps_fixedpoint_outputs(self, op_name: str) -> bool:
        if self.profile == "noinput_rmsnorm_out8":
            return op_name != "rmsnorm"
        return self.profile in {
            "fixedpoint",
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
        }

    def _fp_rmsnorm(self, hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        out = hidden_states * torch.rsqrt(variance + eps)
        return (out * weight.to(device=hidden_states.device, dtype=out.dtype)).to(dtype)

    def _fp_softmax(self, attn_weights: torch.Tensor, dim: int) -> torch.Tensor:
        dtype = attn_weights.dtype
        attn_f = attn_weights.float()
        attn_f = torch.where(torch.isfinite(attn_f), attn_f, torch.full_like(attn_f, -1.0e4))
        return torch.softmax(attn_f, dim=dim).to(dtype)

    def _fp_silu(self, x: torch.Tensor, smooth: Optional[torch.Tensor] = None) -> torch.Tensor:
        if smooth is None:
            return torch.nn.functional.silu(x)
        return x * torch.sigmoid(x / smooth.to(device=x.device, dtype=x.dtype).clamp(min=1e-8))

    def _fp_mul(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return x1 * x2

    def _ibert_exp_negative(self, x: torch.Tensor) -> torch.Tensor:
        x_neg = x.float().clamp(max=0.0)
        x_q, scale = _signed_quantize(x_neg, self.bits, reduce_dim=(-1,))
        x_int = x_q.float().clamp(max=0.0)

        x0_int = torch.floor(torch.tensor(-math.log(2.0), device=x.device) / scale).clamp(max=-1)
        quotient = torch.floor(x_int / x0_int).clamp(min=0, max=31)
        remainder = x_int - x0_int * quotient

        # I-BERT approximates exp on [-ln(2), 0] with a second-order polynomial
        # and range-reduces larger negative inputs by powers of two.
        coef0, coef1, coef2 = 0.35815147, 0.96963238, 1.0
        b = coef1 / coef0
        c = coef2 / coef0
        b_int = torch.floor(torch.tensor(b, device=x.device) / scale)
        c_int = torch.floor(torch.tensor(c, device=x.device) / (scale * scale))
        z = (remainder + b_int) * remainder + c_int
        exp_val = z * (coef0 * scale * scale) / torch.pow(2.0, quotient)
        return exp_val.clamp(0.0, 1.0)

    def _ibert_sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        exp_abs_neg = self._ibert_exp_negative(-x_f.abs())
        sig_pos = 1.0 / (1.0 + exp_abs_neg)
        sig_neg = exp_abs_neg / (1.0 + exp_abs_neg)
        sig = torch.where(x_f >= 0, sig_pos, sig_neg)
        return _unit_fake_quant(sig, self.bits).float()

    def _ibert_softmax(self, attn_weights: torch.Tensor, dim: int) -> torch.Tensor:
        dtype = attn_weights.dtype
        attn_f = attn_weights.float()
        attn_f = torch.where(torch.isfinite(attn_f), attn_f, torch.full_like(attn_f, -1.0e4))
        shifted = attn_f - attn_f.amax(dim=dim, keepdim=True)
        exp_vals = self._ibert_exp_negative(shifted)
        probs = exp_vals / exp_vals.sum(dim=dim, keepdim=True).clamp(min=1e-12)
        probs = _unit_fake_quant(probs, self.bits)
        return _renorm_probs(probs, dim).to(dtype)

    def _ibert_silu(self, x: torch.Tensor, smooth: Optional[torch.Tensor] = None) -> torch.Tensor:
        dtype = x.dtype
        x_f = x.float()
        if smooth is not None:
            x_sig = x_f / smooth.to(device=x.device, dtype=torch.float32).clamp(min=1e-8)
        else:
            x_sig = x_f
        sig = self._ibert_sigmoid(x_sig)
        out = x_f * sig
        return _signed_fake_quant(out, self.bits).to(dtype)

    def _ibert_rmsnorm(self, hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        dtype = hidden_states.dtype
        x_q, scale = _signed_quantize(hidden_states, self.bits, reduce_dim=(-1,))
        q_f = x_q.float()
        eps_q = eps / (scale * scale)
        mean_sq = q_f.pow(2).mean(dim=-1, keepdim=True) + eps_q
        denom = torch.floor(torch.sqrt(mean_sq) * 16.0).clamp(min=1.0) / 16.0
        normed = q_f / denom
        out = normed * weight.to(device=hidden_states.device, dtype=torch.float32)
        return _signed_fake_quant(out, self.bits).to(dtype)

    def _gemmlowp_exp_negative(self, x: torch.Tensor) -> torch.Tensor:
        x_fixed = _fixed_round(x.float().clamp(max=0.0), frac_bits=12, clamp_abs=32.0)
        exp_val = torch.exp(x_fixed).clamp(0.0, 1.0)
        # Simulate a Q0.31 output, then keep only the configured output bits.
        q31 = torch.round(exp_val * ((1 << 31) - 1)).clamp(0, (1 << 31) - 1)
        out = q31 / float((1 << 31) - 1)
        if self.profile in {"w8a8", "w8a8_mid", "w8a8_lossy"}:
            # The corrected fixedpoint profile keeps gemmlowp-style internal
            # precision.  W8A8 profiles also requantize activation-like
            # nonlinear intermediates so the result is not overly optimistic.
            return _unit_fake_quant(out, self.bits).float()
        return out

    def _gemmlowp_softmax(self, attn_weights: torch.Tensor, dim: int) -> torch.Tensor:
        dtype = attn_weights.dtype
        attn_f = attn_weights.float()
        attn_f = torch.where(torch.isfinite(attn_f), attn_f, torch.full_like(attn_f, -1.0e4))
        if self.profile == "w8a8":
            mask = attn_f < -1.0e3
            active_logits = attn_f.masked_fill(mask, 0.0)
            active_logits = _signed_fake_quant_all_dims(active_logits, self.bits).float()
            attn_f = active_logits.masked_fill(mask, -1.0e4)
        elif self.profile in {"w8a8_mid", "w8a8_lossy"}:
            mask = attn_f < -1.0e3
            active_logits = attn_f.masked_fill(mask, 0.0)
            active_logits = _signed_fake_quant(active_logits, self.bits, reduce_dim=(dim,)).float()
            attn_f = active_logits.masked_fill(mask, -1.0e4)
        shifted = attn_f - attn_f.amax(dim=dim, keepdim=True)
        exp_vals = self._gemmlowp_exp_negative(shifted)
        denom = _fixed_round(exp_vals.sum(dim=dim, keepdim=True), frac_bits=16, clamp_abs=65536.0).clamp(min=1e-12)
        probs = _fixed_round(exp_vals / denom, frac_bits=16, clamp_abs=1.0)
        if self.profile == "w8a8":
            # Do not repair the INT8 probability vector with an FP renormalize
            # pass in the literal W8A8 profile; that repair made Gemmlowp look
            # too optimistic for the follow-up requested by the user.
            probs = _unit_fake_quant(probs, self.bits)
            return torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0).to(dtype)
        if self.profile == "w8a8_mid":
            probs = _unit_fake_quant(probs, self.bits)
            return _renorm_probs(probs, dim).to(dtype)
        if self.profile == "w8a8_lossy":
            probs = _unit_fake_quant(probs, self.bits)
            return torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0).to(dtype)
        if not self._gemmlowp_keeps_fixedpoint_outputs("softmax") or self.profile == "softmax_int8":
            probs = _unit_fake_quant(probs, self.bits)
            return _renorm_probs(probs, dim).to(dtype)
        return _renorm_probs(probs, dim).to(dtype)

    def _gemmlowp_sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        x_f = _fixed_round(x.float(), frac_bits=10, clamp_abs=32.0)
        exp_abs_neg = self._gemmlowp_exp_negative(-x_f.abs())
        sig_pos = 1.0 / (1.0 + exp_abs_neg)
        sig_neg = exp_abs_neg / (1.0 + exp_abs_neg)
        sig = torch.where(x_f >= 0, sig_pos, sig_neg)
        sig = _fixed_round(sig, frac_bits=16, clamp_abs=1.0)
        if self._gemmlowp_keeps_fixedpoint_outputs("silu"):
            return sig.float()
        return _unit_fake_quant(sig, self.bits).float()

    def _gemmlowp_silu(self, x: torch.Tensor, smooth: Optional[torch.Tensor] = None) -> torch.Tensor:
        dtype = x.dtype
        x_f = x.float()
        if self.profile == "w8a8":
            x_f = _signed_fake_quant_all_dims(x_f, self.bits).float()
        elif self.profile in {"w8a8_mid", "w8a8_lossy"}:
            x_f = _signed_fake_quant(x_f, self.bits, reduce_dim=(-1,)).float()
        if smooth is not None:
            x_sig = x_f / smooth.to(device=x.device, dtype=torch.float32).clamp(min=1e-8)
        else:
            x_sig = x_f
        sig = self._gemmlowp_sigmoid(x_sig)
        out = _fixed_round(x_f * sig, frac_bits=12, clamp_abs=65536.0)
        if self._gemmlowp_keeps_fixedpoint_outputs("silu") or self.profile == "softmax_int8":
            return torch.nan_to_num(out, nan=0.0, posinf=1.0e4, neginf=-1.0e4).to(dtype)
        if self.profile == "w8a8":
            return _signed_fake_quant_all_dims(out, self.bits).to(dtype)
        if self.profile in {"w8a8_mid", "w8a8_lossy"}:
            return _signed_fake_quant(out, self.bits, reduce_dim=(-1,)).to(dtype)
        return _signed_fake_quant(out, self.bits).to(dtype)

    def _gemmlowp_rmsnorm(self, hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        dtype = hidden_states.dtype
        if self.profile in GEMMLOWP_RMSNORM_NO_INPUT_QUANT:
            # Keep the initial INT8-input RMSNorm path above as a diagnostic
            # profile.  This branch measures gemmlowp-style fixed-point rsqrt
            # and normalization without the extra RMSNorm input truncation.
            x_deq = _fixed_round(hidden_states.float(), frac_bits=16, clamp_abs=65536.0)
        else:
            reduce_dim = _reduce_for_all_dims(hidden_states) if self.profile == "w8a8" else (-1,)
            x_q, scale = _signed_quantize(hidden_states, self.bits, reduce_dim=reduce_dim)
            x_deq = x_q.float() * scale
        # The first diagnostic implementation rounded variance with 12
        # fractional bits; LLaMA hidden-state variances can be much smaller
        # than 2^-12, so that path collapsed to zero and badly overstated
        # gemmlowp error.  Use a high-precision fixed-point accumulator here.
        variance = _fixed_round(x_deq.pow(2).mean(dim=-1, keepdim=True), frac_bits=24, clamp_abs=65536.0)
        inv_rms = _fixed_round(torch.rsqrt(variance + eps), frac_bits=16, clamp_abs=65536.0)
        weight_f = weight.to(device=hidden_states.device, dtype=torch.float32)
        if self.profile in {"w8a8", "w8a8_mid", "w8a8_lossy"}:
            weight_f = _signed_fake_quant(weight_f, self.bits, reduce_dim=(0,)).float()
        out = x_deq * inv_rms * weight_f
        if self._gemmlowp_keeps_fixedpoint_outputs("rmsnorm") or self.profile == "softmax_int8":
            return torch.nan_to_num(out, nan=0.0, posinf=1.0e4, neginf=-1.0e4).to(dtype)
        if self.profile == "w8a8":
            return _signed_fake_quant_all_dims(out, self.bits).to(dtype)
        if self.profile in {"w8a8_mid", "w8a8_lossy"}:
            return _signed_fake_quant(out, self.bits, reduce_dim=(-1,)).to(dtype)
        return _signed_fake_quant(out, self.bits).to(dtype)

    def _int8_mul(self, x1: torch.Tensor, x2: torch.Tensor, accum_frac_bits: int) -> torch.Tensor:
        dtype = x1.dtype
        if self.method == "gemmlowp" and (
            self._gemmlowp_keeps_fixedpoint_outputs("mul") or self.profile in {"softmax_int8", "no_mul"}
        ):
            out = _fixed_round(x1.float() * x2.float(), frac_bits=accum_frac_bits, clamp_abs=65536.0)
            return torch.nan_to_num(out, nan=0.0, posinf=1.0e4, neginf=-1.0e4).to(dtype)
        reduce_dim = _reduce_for_all_dims(x1) if self.method == "gemmlowp" and self.profile == "w8a8" else (-1,)
        q1, s1 = _signed_quantize(x1, self.bits, reduce_dim=reduce_dim)
        q2, s2 = _signed_quantize(x2, self.bits, reduce_dim=reduce_dim)
        product = q1.float() * q2.float() * s1 * s2
        product = _fixed_round(product, frac_bits=accum_frac_bits, clamp_abs=65536.0)
        if self.method == "gemmlowp" and self.profile == "w8a8":
            return _signed_fake_quant_all_dims(product, self.bits).to(dtype)
        if self.method == "gemmlowp" and self.profile in {"w8a8_mid", "w8a8_lossy"}:
            return _signed_fake_quant(product, self.bits, reduce_dim=(-1,)).to(dtype)
        return _signed_fake_quant(product, self.bits).to(dtype)


def get_nonlinear_backend(method: str, bits: int, profile: str = "strict") -> Optional[NonlinearIntBackend]:
    method = (method or "disabled").lower()
    profile = (profile or "strict").lower()
    if method == "disabled":
        return None
    if method not in {"ibert", "gemmlowp"}:
        raise ValueError(f"Unknown nonlinear int method: {method}")
    if profile not in GEMMLOWP_PROFILES:
        raise ValueError(f"Unknown nonlinear int profile: {profile}")
    if bits < 2 or bits > 15:
        raise ValueError(f"nonlinear_bits must be in [2, 15], got {bits}")
    return NonlinearIntBackend(NonlinearIntConfig(method=method, bits=bits, profile=profile))
