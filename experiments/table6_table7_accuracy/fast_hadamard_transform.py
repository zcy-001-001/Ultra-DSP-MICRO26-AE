import math

import torch
import torch.nn.functional as F


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def hadamard_transform(x: torch.Tensor, scale=1.0) -> torch.Tensor:
    """Pure PyTorch fallback for Dao-AILab fast_hadamard_transform.

    The CUDA extension is preferred when it is installed.  On the rebuttal FPGA
    machines the extension may be unavailable, so this module keeps the OSTQuant
    flow runnable with the same mathematical transform.
    """
    if x.numel() == 0:
        return x

    original_shape = x.shape
    dim = x.shape[-1]
    padded_dim = _next_power_of_two(dim)

    y = x.reshape(-1, dim)
    if padded_dim != dim:
        y = F.pad(y, (0, padded_dim - dim))
    else:
        y = y.clone()

    stride = 1
    while stride < padded_dim:
        y = y.reshape(-1, padded_dim // (2 * stride), 2, stride)
        left = y[:, :, 0, :]
        right = y[:, :, 1, :]
        y = torch.stack((left + right, left - right), dim=2).reshape(-1, padded_dim)
        stride *= 2

    scale_tensor = torch.as_tensor(scale, dtype=y.dtype, device=y.device)
    y = y[:, :dim] * scale_tensor
    return y.reshape(original_shape)


def hadamard_transform_12N(x: torch.Tensor, scale=1.0) -> torch.Tensor:
    return hadamard_transform(x, scale=scale)


def hadamard_transform_20N(x: torch.Tensor, scale=1.0) -> torch.Tensor:
    return hadamard_transform(x, scale=scale)


def hadamard_transform_28N(x: torch.Tensor, scale=1.0) -> torch.Tensor:
    return hadamard_transform(x, scale=scale)


def hadamard_transform_40N(x: torch.Tensor, scale=1.0) -> torch.Tensor:
    return hadamard_transform(x, scale=scale)
