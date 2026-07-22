# Table 6 dataset revision lock

This lock records the exact cached dataset revisions used by the formal A800
run. It contains no host path. The revision is recovered from the immutable
revision embedded in each cached Hugging Face parquet URI; the cache
fingerprint and SHA-256 of `dataset_info.json` provide independent checks.

Environment: `datasets==4.8.5`, `lm_eval==0.4.12`.

| Purpose | Dataset/config | Split | Revision / cache fingerprint | Examples | Task version |
|---|---|---|---|---:|---:|
| Calibration | `Salesforce/wikitext` / `wikitext-2-raw-v1` | train | `b08601e04326c79dfdd32d625aee71d232d685c3` | 36,718 | n/a |
| Evaluation | `allenai/ai2_arc` / `ARC-Easy` | test | `210d026faf9955653af8916fad021475a3f00453` | 2,376 | 1 |
| Evaluation | `Rowan/hellaswag` / `default` | validation | `218ec52e09a7e7462a5400043bb9a69a41d06b76` | 10,042 | 1 |
| Evaluation | `baber/piqa` / `default` | validation | `142f6d7367fd9877f0fb3b5734ea6a545f54cdd1` | 1,838 | 1 |
| Evaluation | `allenai/openbookqa` / `main` | test | `388097ea7776314e93a529163e0fea805b8a6454` | 500 | 1 |

The full JSON additionally freezes each builder/cache version, the
`dataset_info.json` SHA-256, and the four lm-eval task-YAML hashes. Both
completed model logs independently print task Version 1 with zero shots.

The formal run loaded four serialized WikiText token loaders rather than
rebuilding them online. Their exact SHA-256 values and byte sizes are therefore
also frozen in the JSON, together with 128 samples, sequence length 2048, and
seed 42. These loader hashes are the strongest evidence for the calibration
inputs actually consumed. The loader format does not embed its source dataset
revision, so the matching WikiText cache revision is supporting provenance,
not a cryptographic derivation claim.

These are provenance records; model weights and dataset payloads are not
redistributed.
