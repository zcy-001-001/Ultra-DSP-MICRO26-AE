# Table 6 remote model-load smoke

Evidence class: `REMOTE_SMOKE`

This smoke validates the exact environment and external model copies used by
the fresh Table 6 regeneration. It is not an accuracy result.

## Environment

- GPU: NVIDIA A40, 48 GB
- Driver: 575.57.08
- Python: 3.10.20
- PyTorch: 2.5.1+cu121
- Transformers: 4.57.6
- Tokenizers: 0.22.2
- Geoopt: 0.5.0
- `torchao`: intentionally absent because it is unused by OSTQuant and the
  available release required a newer PyTorch ABI

The task uses an isolated copy of the remote Conda environment. The shared
base environment was not modified. The complete package list is in
`a40_environment_lock.txt`.

## Input integrity

- Llama-2: 10/10 top-level files matched the source SHA-256 values.
- Llama-3: 14/14 top-level Hugging Face files matched the source SHA-256
  values. The unused `original/consolidated.00.pth` duplicate is not an input
  to this workflow.
- Full external-file hashes are recorded in `model_files_sha256.txt`; licensed
  model files are not redistributed.

## Tests

```text
SYMMETRIC_CONFIG_STATIC_PASS tables=6,7
SYMMETRIC_QUANTIZATION_PASS bits=3,4,5
WP521 unsigned raw local checks passed.
TOKENIZER_SMOKE_PASS class=LlamaTokenizerFast vocab_size=32000
TOKENIZER_SMOKE_PASS class=LlamaTokenizerFast vocab_size=128257
MODEL_LOAD_SMOKE_PASS llama2_7b
MODEL_LOAD_SMOKE_PASS llama3_8b
TABLE6_ENV_SMOKE_PASS models=2
```

The scheduled A40 smoke completed in 2 minutes 18 seconds with exit code 0.
Both models loaded every safetensors shard, their tokenizer, and the cached
WikiText-2 evaluation loader. Full lm-eval accuracy is intentionally left to
the long Table 6 regeneration.

An earlier preflight exposed an incompatible tokenizer/runtime combination
before evaluation began. No accuracy values were produced. The isolated
environment above resolved it and is the only environment admitted to the
formal run.
