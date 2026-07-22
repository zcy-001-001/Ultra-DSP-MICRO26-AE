# Table 6 full regeneration

This folder reproduces Table 6 by regenerating the W4A4 OSTQuant checkpoints first, then evaluating the three overpacking simulators from the freshly generated checkpoint.  It intentionally does not support supplying a prebuilt qmodel artifact as the primary workflow.

Run the commands in this file from `experiments/table6_table7_accuracy/`.
All canonical evidence paths beginning with `results/table6/` are relative to
the AE repository root; optional reruns should use `../../results/rerun/table6/`.

## What is regenerated

For each model:

1. Run the BF16 baseline with `--pre_eval=False --rotate=False --lm_eval=True`.
2. Train/rotate/smooth/GPTQ a W4A4KV4 model and save `train_w4a4/qmodel.pt`.
3. Evaluate the generated qmodel with:
   - `--linear_int_mode=exact` for Ultra-DSP.
   - `--linear_int_mode=approx --linear_int_variant=lsb3_zero` for DSP-Packing.
   - `--linear_int_mode=approx --linear_int_variant=fixed_alpha --linear_int_fixed_alpha=0.5` for DB-MixQ.

The formal Table 6 result set contains exactly eight rows: BF16, Ultra-DSP,
DSP-Packing, and DB-MixQ for each of the two models. WP521 is not a Table 6 row,
so the formal runner, parser, staged-log set, and admission gate exclude it.
The historical unsigned-activation WP521 result remains separately archived
for the rebuttal and Figure 12; it is not a fresh Table 6 reference-only row.

The parser reports `acc_norm` for `arc_easy`, `hellaswag`, `piqa`, and `openbookqa`, then averages those four percentages.
The runner explicitly passes those four tasks to lm-eval so it does not fall
back to OSTQuant's broader default task list.

## Quantization convention

Table 6 uses the same narrow symmetric convention as Table 7:

- `--a_asym False`
- `--k_asym False`
- `--v_asym False`
- `--narrow_symmetric=True`

This is intentionally different from the original OSTQuant defaults. For a
4-bit symmetric quantizer, the original full signed range is `[-8, 7]`, while
this artifact uses the zero-centered narrow range `[-7, 7]`. The W4A4 qmodel
must therefore be regenerated with this runner; an older checkpoint produced
without these flags is not interchangeable. The BF16 baseline is unaffected.

## OSTQuant optimization stages and KV precision

The staged behavior is inherited from the official OSTQuant implementation,
pinned here at upstream commit
[`ab64362d`](https://github.com/BrotherHappy/OSTQuant/tree/ab64362da147291612d077accaab5d3ed7b508b6).
At that revision, both the official
[`w4a4kv4.sh`](https://github.com/BrotherHappy/OSTQuant/blob/ab64362da147291612d077accaab5d3ed7b508b6/scripts/w4a4kv4.sh)
and
[`w4a4kv16.sh`](https://github.com/BrotherHappy/OSTQuant/blob/ab64362da147291612d077accaab5d3ed7b508b6/scripts/w4a4kv16.sh)
set `--train_enable_wquant=False`, while `w_gptq=True` is the upstream default.
The artifact preserves this optimization ordering and adds the narrow-symmetric
quantizer convention, packing simulators, compatibility fixes, and evidence
packaging needed for Table 6.

The Table 6 runner regenerates a **W4A4KV4** checkpoint. In addition to
`--w_bits=4` and `--a_bits=4`, it sets `--k_bits=4` and `--v_bits=4`; the K/V
cache path is therefore fake-quantized to four bits during OSTQuant training
and evaluation, rather than retained in FP16/BF16.

`--train_enable_wquant=False` disables only the weight fake quantizer during
the OSTQuant parameter-learning phase. It does not disable fake quantization
as a whole. The full-precision teacher forward disables every quantizer, while
the student forward keeps activation and K/V fake quantization enabled. The
`kl_top` loss then trains the Stiefel rotation matrices and the paired smooth
scales through the straight-through quantizers. In particular, `smooth_qk`
redistributes range between Q and K, while `smooth_ov` redistributes range
between V and the O projection without changing the corresponding
full-precision algebra.

Weight quantization is deliberately handled as a second stage. After the
learned rotations and scales are folded into the floating-point model, GPTQ
uses calibration samples to generate the final W4 weights and saves the
resulting `qmodel.pt`. This is a post-training calibration/quantization pass,
not a second optimizer or gradient-training phase. All four packing rows load
that same checkpoint. Thus GPTQ does not replace OSTQuant: the packaged
sequence is KLT initialization, OSTQuant rotation/smoothing learning under
A4/KV4 error, transform folding, GPTQ W4 generation, and common-checkpoint
evaluation.

Before launching a full regeneration, run:

```bash
python experiments/table6_overpacking/scripts/test_symmetric_config_static.py
python experiments/table6_overpacking/scripts/test_symmetric_quantization.py
python experiments/table6_overpacking/scripts/test_build_checkpoint_manifest.py
python experiments/table6_overpacking/scripts/test_stage_fresh_logs.py
```

After parsing a fresh run, compare it with the immutable archived anchor. The
default AE tolerances are 3.0 percentage points per task and 2.0 points for the
four-task average; the script also requires the main `BF16 > Ultra-DSP > lossy
packing baselines` trend for both models. Neither input is modified.

```bash
python experiments/table6_overpacking/scripts/compare_table6.py \
  --fresh <RUN_DIR>/table6_summary.json \
  --expected experiments/table6_overpacking/expected/table6_expected_from_existing_artifacts.json \
  --out <RUN_DIR>/table6_tolerance_report.json \
  --out-md <RUN_DIR>/table6_tolerance_report.md
```

The JSON and Markdown always retain every raw task delta, average delta, and a
per-row `strict_status`. A strictly out-of-tolerance row fails by default. If
the author has reviewed a known environment-dependent difference and explicitly
accepts it for AE, name that exact row with a repeatable option:

```bash
python experiments/table6_overpacking/scripts/compare_table6.py \
  --fresh <RUN_DIR>/table6_summary.json \
  --expected experiments/table6_overpacking/expected/table6_expected_from_existing_artifacts.json \
  --author-accept llama3_8b/Ultra-DSP
```

Only named, strictly out-of-tolerance rows become
`AUTHOR_ACCEPTED_VARIANCE`; their deltas are not changed or hidden. In that
case the report keeps top-level `strict_status=OUT_OF_TOLERANCE` and uses final
`status=PASS_WITH_AUTHOR_ACCEPTED_VARIANCE`. Any unnamed out-of-tolerance row
still exits nonzero.

For the current formal rerun, the author has reviewed and accepted only
`llama3_8b/Ultra-DSP`. Its maximum task delta is 3.24 percentage points and its
average delta is -1.96 points; both remain visible in the report. Do not add
acceptance for any other row without a separate author review.

The first check requires only Python; the second uses PyTorch to verify the
3/4/5-bit integer limits and zero points.

## Run

```bash
cd OSTQuant_table6_clean
MODEL_LLAMA2=/path/to/Llama-2-7b-hf \
MODEL_LLAMA3=/path/to/Llama-3-8B \
RUN_DIR=runs/table6_full_regen \
NPROC=1 EVAL_NPROC=1 \
experiments/table6_overpacking/scripts/run_table6_full_regeneration.sh
```

For two isolated GPUs or hosts, set `MODEL_KEYS=llama2_7b SKIP_PARSE=1` and
`MODEL_KEYS=llama3_8b SKIP_PARSE=1` in separate invocations with distinct
`MASTER_PORT` values, then run `parse_table6.py` once over the shared result
tree. One rank per model is intentional: multiple evaluation ranks each load a
model copy and can exceed device memory.

The script writes logs under `$RUN_DIR/{llama2_7b,llama3_8b}/...` and writes the
eight-row parsed summaries to `$RUN_DIR/table6_summary.json` and
`$RUN_DIR/table6_summary.md`.

The script is resume-aware. It skips a BF16/eval row when its `log.txt` already
contains `AVERAGE`, and skips W4A4 regeneration when both
`train_w4a4/model.bin` and `train_w4a4/qmodel.pt` already exist.

## Expected values

`expected/table6_expected_from_existing_artifacts.json` records the values found in the original H100 logs.  Full regeneration should be close, but small drift is possible because the W4A4 qmodel is regenerated rather than loaded from an existing artifact.

The original screenshot has Llama-3-8B DB-MixQ `PIQA=51.57` / `Avg=32.68`; the H100 log artifact found for the corresponding run has `PIQA=51.74` / `Avg=32.72`.
The 0.04-point average difference is accepted for AE, while both raw and paper
values remain visible. `results/table6/preflight_environment_and_hashes.txt` records
the current package/GPU configuration plus runner, code, config, and
weight-index hashes without exposing host-specific paths.

`Baseline (BF16)` is the unquantized upper-bound row. It is distinct from the
paper's `original OSTQuant quantized baseline`, which denotes OSTQuant under
the same W4A4 setting and is preserved by the bit-exact Ultra-DSP row. See
`results/table6/table6_baseline_semantics.md` for the paper/code cross-check.

Additional packaged evidence under `results/table6/`:

- `a40_environment_lock.txt`: isolated formal-run package lock.
- `model_files_sha256.txt`: full top-level SHA-256 lists for both licensed
  external models; the model files themselves are not redistributed.
- `remote_model_load_smoke.md`: two-model A40 load/tokenizer/cache smoke and
  the resolved environment-compatibility note.
- `klt_cusolver_smoke.md`: compatibility rationale and PASS signal for the
  4096 by 4096 CUDA KLT eigendecomposition used by the formal run.
- `formal_training_evidence.md`: completed two-model W4A4KV4 training metrics,
  checkpoint sizes, and formal-model-path KLT/cuSOLVER evidence without
  redistributing generated model data.
- `table6_baseline_semantics.md`: paper/code cross-check distinguishing the
  unquantized BF16 reference from the W4A4 OSTQuant quantized baseline.
- `formal_source_provenance.json` and `.md`: formal execution hashes versus
  public artifact hashes, including the expected path-placeholder-only runner
  difference and the evidence boundary for the KLT helper.

### Fresh-result admission contract

The formal rerun is complete and admitted under portable run identifier
`table6-symmetric-a800-20260720-v2`. The package contains all 8 summary rows and
all 8 summary-referenced sanitized logs. Seven rows have strict status `PASS`;
`llama3_8b/Ultra-DSP` is the sole `AUTHOR_ACCEPTED_VARIANCE`, with maximum task
delta 3.24 percentage points and average delta -1.96 points. The final status is
`PASS_WITH_AUTHOR_ACCEPTED_VARIANCE`. WP521 is excluded.

The admitted portable evidence comprises:

- `table6_fresh_summary.json`: exactly eight unique model/method rows generated
  by `parse_table6.py`, each with `source_kind=FRESH_REMOTE_RERUN`, one shared
  `run_id`, and a relative `log` path.
- `table6_fresh_logs_sanitized/`: exactly the eight summary-referenced lm-eval
  logs. Each log must contain `AVERAGE` and all four configured tasks.
- `table6_fresh_training_logs_sanitized/`: exactly two sanitized training
  logs preserving each model's effective narrow-symmetric W4A4KV4 runtime
  configuration and OST rotation/smoothing training markers.
- `table6_fresh_tolerance_report.json` and `.md`: output from
  `compare_table6.py`, carrying the same `run_id`, all strict deltas/statuses,
  and either `PASS` or an explicit `PASS_WITH_AUTHOR_ACCEPTED_VARIANCE` status.
  All eight rows are compared with the archived Table 6 anchor; unexpected
  methods, including WP521, are rejected rather than admitted as reference-only.
- `table6_fresh_a800_environment_lock.txt`: the formal A800 package lock with
  a `run_id=<id>` record plus `datasets==4.8.5` and `lm_eval==0.4.12`; a
  preflight or A40 lock is not accepted here.
- `table6_fresh_checkpoint_manifest.json`: an object with the shared `run_id`
  and four records—`model.bin` and `qmodel.pt` for each model. Each record uses
  a relative evidence path and records lowercase SHA-256 plus byte size. The
  generated checkpoint payloads themselves are not redistributed.

After the evaluation processes have exited, generate the checkpoint manifest
without copying model data or recording the private run root:

```bash
python experiments/table6_overpacking/scripts/build_checkpoint_manifest.py \
  --run-dir <RUN_DIR> \
  --run-id table6-symmetric-a800-20260720-v2 \
  --out ../../results/rerun/table6/table6_fresh_checkpoint_manifest.json
```

The helper intentionally streams the four files only after evaluation has
finished so its shared-storage reads cannot perturb the measured run.

Stage the eight completed logs with the package's shared privacy rules. The
helper validates and reparses all source and sanitized logs, publishes the
output atomically, and refuses to overwrite an existing evidence directory:

```bash
python experiments/table6_overpacking/scripts/stage_fresh_logs.py \
  --run-dir <RUN_DIR> \
  --out-dir ../../results/rerun/table6/table6_fresh_logs_sanitized
```

The formal dataset revisions are frozen separately in
`results/table6/dataset_revisions.json` and `results/table6/dataset_revisions.md`. They record
the calibration/evaluation dataset, configuration, split, immutable cache
revision, cache fingerprint, `dataset_info.json` SHA-256, example count, and
lm-eval task version without exposing a host path.

Run `python scripts/verify_artifact.py` from the artifact root. The verifier
reparses all eight logs, checks the summary values, tolerance reports,
environment/checkpoint provenance, error markers, and private paths before it
prints `TABLE6_FRESH_PASS`.

## KLT eigendecomposition compatibility

The historical OSTQuant KLT helper called `torch.linalg.eigh` directly on a
CPU covariance tensor. PyTorch 2.5.1 with oneMKL can reject the 4096 by 4096
`SSYEVD` workspace before training begins. The packaged helper preserves that
historical expression as a source comment and performs the same real symmetric
eigendecomposition through CUDA when a GPU is available. The returned
eigenvectors are moved back to the original parameter device before the
unchanged Hadamard composition. See `results/table6/klt_cusolver_smoke.md` for the
isolated A40 verification.
