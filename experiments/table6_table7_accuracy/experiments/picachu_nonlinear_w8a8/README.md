# PICACHU Nonlinear W8A8 Evaluation

This experiment measures the PICACHU Table 2 follow-up requested for LLaMA-2-7B
and LLaMA-3-8B: linear layers remain FP16/BF16, while selected nonlinear
operators are evaluated with I-BERT-style or gemmlowp-style INT8 fake-quant
emulation. RoPE is intentionally kept on the original FP path.

## Directory Summary

- `scripts/run_full.sh`: resume-aware runner for FP16 baseline, I-BERT, and
  gemmlowp rows. It writes one subfolder per model and method under `$RUN_DIR`.
  The default `gemmlowp` row uses `NONLINEAR_INT_PROFILE=w8a8_mid`. Profile
  sweeps can use method keys such as `gemmlowp_strict`,
  `gemmlowp_w8a8_mid`, `gemmlowp_w8a8_lossy`, `gemmlowp_w8a8`,
  `gemmlowp_fixedpoint`, `gemmlowp_softmax_int8`, and `gemmlowp_no_mul`.
- `scripts/parse_results.py`: parses `log.txt` files into JSON/Markdown summary
  tables with PPL, ARC-e, HellaSwag, PIQA, Winogrande, average accuracy, and
  geomean accuracy.
- `scripts/test_nonlinear_int.py`: CPU toy tests for RMSNorm, Softmax, SiLU, the
  SwiGLU multiply path, and the invariant that RoPE is not wired to the
  nonlinear backend.
- `../../runs/picachu_nonlinear_w8a8_*`: default output location for logs,
  commands, and parsed summaries. Each run contains
  `{llama2_7b,llama3_8b}/{fp16,ibert,gemmlowp}/`, plus any explicit
  `gemmlowp_*` profile rows requested in a sweep.

## Reproduction

Run the toy tests first:

```bash
cd <REMOTE_HOME>/rebuttal/OSTQuant_table6_clean
<REMOTE_HOME>/miniconda3/envs/nonlinear/bin/python \
  experiments/picachu_nonlinear_w8a8/scripts/test_nonlinear_int.py
```

Install the evaluation dependencies once if the `nonlinear` environment does
not have them:

```bash
<REMOTE_HOME>/miniconda3/envs/nonlinear/bin/python -m pip install \
  loguru geoopt einops lm_eval==0.4.4 peft==0.4.0
```

Smoke test on <REMOTE_HOST> with two WikiText2 chunks and 20 lm-eval examples per task:

```bash
cd <REMOTE_HOME>/rebuttal/OSTQuant_table6_clean
RUN_DIR=<REMOTE_HOME>/rebuttal/OSTQuant_table6_clean/runs/picachu_nonlinear_w8a8_smoke \
MODEL_KEYS=llama2_7b \
METHOD_KEYS="ibert gemmlowp" \
NONLINEAR_INT_PROFILE=w8a8_mid \
EVAL_SAMPLES=2 \
LM_EVAL_LIMIT=20 \
LM_EVAL_BATCH_SIZE=8 \
bash experiments/picachu_nonlinear_w8a8/scripts/run_full.sh
```

Optional all-W8A8 diagnostic smoke test. This is not the final PICACHU
follow-up setting, because the final setting keeps linear layers FP16/BF16; it
is only useful when checking the stricter interpretation where linear
weights/activations and nonlinear activations are all fake-quantized to INT8:

```bash
cd <REMOTE_HOME>/rebuttal/OSTQuant_table6_clean
RUN_DIR=<REMOTE_HOME>/rebuttal/OSTQuant_table6_clean/runs/picachu_all_w8a8_smoke \
MODEL_KEYS=llama2_7b \
METHOD_KEYS=gemmlowp_w8a8_mid \
NONLINEAR_INT_PROFILE=w8a8_mid \
W_BITS=8 A_BITS=8 DOWN_BITS=8 V_BITS=8 K_BITS=8 ACT_BITS=8 RESIDUAL_BITS=8 ATTN_BITS=8 \
W_GPTQ=False \
EVAL_SAMPLES=8 \
LM_EVAL_LIMIT=20 \
LM_EVAL_BATCH_SIZE=8 \
bash experiments/picachu_nonlinear_w8a8/scripts/run_full.sh
```

Gemmlowp profile sweep on <REMOTE_HOST> before the full run:

```bash
cd <REMOTE_HOME>/rebuttal/OSTQuant_table6_clean
RUN_DIR=<REMOTE_HOME>/rebuttal/OSTQuant_table6_clean/runs/picachu_nonlinear_w8a8_gemmlowp_sweep \
MODEL_KEYS=llama2_7b \
METHOD_KEYS="fp16 gemmlowp_strict gemmlowp_w8a8_mid gemmlowp_w8a8_lossy gemmlowp_w8a8 gemmlowp_fixedpoint gemmlowp_softmax_int8 gemmlowp_no_mul" \
EVAL_SAMPLES=8 \
RUN_LM_EVAL=0 \
bash experiments/picachu_nonlinear_w8a8/scripts/run_full.sh
```

Full run with one model per host:

```bash
ssh <REMOTE_HOST> "cd <REMOTE_HOME>/rebuttal/OSTQuant_table6_clean && \
  RUN_DIR=<REMOTE_HOME>/rebuttal/OSTQuant_table6_clean/runs/picachu_nonlinear_w8a8_full \
  MODEL_KEYS=llama2_7b \
  NONLINEAR_INT_PROFILE=w8a8_mid \
  bash experiments/picachu_nonlinear_w8a8/scripts/run_full.sh"

ssh <REMOTE_HOST> "cd <REMOTE_HOME>/rebuttal/OSTQuant_table6_clean && \
  RUN_DIR=<REMOTE_HOME>/rebuttal/OSTQuant_table6_clean/runs/picachu_nonlinear_w8a8_full \
  MODEL_KEYS=llama3_8b \
  NONLINEAR_INT_PROFILE=w8a8_mid \
  bash experiments/picachu_nonlinear_w8a8/scripts/run_full.sh"
```

After both hosts finish, regenerate the combined summary:

```bash
cd <REMOTE_HOME>/rebuttal/OSTQuant_table6_clean
<REMOTE_HOME>/miniconda3/envs/nonlinear/bin/python \
  experiments/picachu_nonlinear_w8a8/scripts/parse_results.py \
  --run-dir runs/picachu_nonlinear_w8a8_full \
  --out runs/picachu_nonlinear_w8a8_full/picachu_nonlinear_w8a8_summary.json \
  --out-md runs/picachu_nonlinear_w8a8_full/picachu_nonlinear_w8a8_summary.md
```

## Notes

- The runner uses `--use_sdpa=False` so attention softmax passes through
  `QuantSoftmax`.
- `HF_ENDPOINT` defaults to `https://hf-mirror.com` for dataset fetches on the
  fpga hosts; override it if your network reaches Hugging Face directly.
- Full PPL uses `wikitext/wikitext-2-raw-v1` test with `eval_samples=-1`.
- Downstream tasks use lm-eval default full public evaluation splits with
  `arc_easy`, `hellaswag`, `piqa`, and `winogrande`.
- `strict` preserves the initial all-INT8-output fake-quant gemmlowp path as a
  diagnostic. `fixedpoint` is the corrected but optimistic Gemmlowp profile:
  nonlinear inputs are fake-quantized where applicable, nonlinear primitives use
  fixed-point rounding, and RMSNorm variance/rsqrt use a high-precision
  fixed-point accumulator to avoid the unrealistic collapse seen with a 12-bit
  fractional variance. `w8a8_mid` is the selected Gemmlowp W8A8 nonlinear
  profile for the final row: it keeps per-token dynamic activation scales,
  requantizes nonlinear intermediates/outputs and RMSNorm weights to INT8
  fake-quant, and renormalizes INT8 softmax probabilities. `w8a8_lossy` keeps
  the same per-token scales but does not repair softmax probability mass, and
  is retained as a measured candidate. `w8a8` is retained only as the rejected
  all-tensor/no-renorm diagnostic because it overstates the accuracy loss.
- No device reboot or adb restart is required for this experiment.
