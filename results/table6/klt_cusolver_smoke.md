# KLT eigendecomposition compatibility smoke

## Purpose

The Table 6 regeneration computes a 4096 by 4096 covariance eigendecomposition
while constructing the OSTQuant KLT rotation. In the isolated A40 environment,
the historical CPU call reached oneMKL `SSYEVD` and stopped before training with
an internal `linalg.eigh` argument error.

The compatibility patch in `quant/ost_model_utils.py` retains the historical
CPU expression as a source comment and sends the same real symmetric covariance
matrix to CUDA for eigendecomposition. Eigenvectors are returned to the
original parameter device before the existing Hadamard composition. This does
not change the KLT objective or the narrow-symmetric quantization convention.

## Environment

- GPU: NVIDIA A40
- Python: 3.10.20
- PyTorch: 2.5.1 with CUDA 12.1 runtime
- Matrix dtype: FP32
- Matrix shape: 4096 by 4096

## Verification

The isolated smoke constructs a finite positive-semidefinite covariance matrix,
runs `torch.linalg.eigh` through CUDA, and checks the eigenvalue/eigenvector
shapes and finiteness.

```text
KLT_CUSOLVER_SMOKE_PASS size=4096 device=cuda:0
```

The smoke completed successfully in 20 seconds. This file records the
compatibility check only; the final Table 6 result summary and sanitized formal
run logs are stored separately after both model runs complete.
