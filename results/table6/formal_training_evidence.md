# Formal W4A4KV4 training evidence

This note records the completed checkpoint-generation stage of the fresh
Table 6 rerun. Evaluation results are archived separately only after every
method row finishes. No licensed model weights or generated checkpoint
binaries are redistributed.

## Effective quantization configuration

Both model logs report the following runtime values:

```text
a_asym=False
k_asym=False
v_asym=False
narrow_symmetric=True
k_bits=4
v_bits=4
train_enable_wquant=False
use_klt=True
w_gptq=True
```

The run therefore follows the narrow-symmetric W4A4KV4 sequence documented
in the experiment README: KLT initialization, OSTQuant rotation/smoothing
learning under activation and KV fake-quantization error, transform folding,
and GPTQ generation of the final W4 weights.

## Completed training stage

| Model | Steps | Runtime (s) | Final reported training loss | `model.bin` bytes | `qmodel.pt` bytes |
|---|---:|---:|---:|---:|---:|
| Llama-2-7B | 100 | 1391.7062 | 0.1008515338 | 138,051,423 | 13,555,347,222 |
| Llama-3-8B | 100 | 1434.5068 | 0.2371423160 | 87,142,374 | 16,139,169,642 |

The per-step records contain finite, nonzero gradient norms and a decaying
learning rate for both models. GPTQ completed all 32 transformer layers and
both checkpoint pairs were written successfully.

The two sanitized configuration/training logs are retained under
`table6_fresh_training_logs_sanitized/`. They preserve the effective runtime
configuration and the rotation plus smoothing training-start records for each
model without exposing the private run root. The compact table above records
the completed 100-step trainer summaries and checkpoint sizes; generated model
payloads are not redistributed.

## KLT/cuSOLVER evidence

The isolated compatibility smoke in `klt_cusolver_smoke.md` completed the
same 4096 by 4096 CUDA eigendecomposition used by the formal path. Both full
model runs then completed KLT initialization and all 100 optimization steps.
Their logs contain no oneMKL `SSYEVD`, `linalg.eigh`, traceback, runtime, or
CUDA out-of-memory error. This provides formal-model-path evidence in addition
to the isolated smoke.

## Execution scope

The previously archived BF16 baselines were measured on an A40 environment.
The fresh W4A4KV4 checkpoint generation and packing-method evaluation use one
NVIDIA A800-SXM4-80GB per model, Python 3.10.20, and the package versions
captured by the final sanitized environment records. This hardware distinction
is retained so the BF16 and fresh quantized evidence are not misrepresented as
one same-device run.
