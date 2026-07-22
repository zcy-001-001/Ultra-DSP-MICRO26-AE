# Table 6 formal-source provenance

This record separates the byte hashes used by the formal run from the hashes
of the privacy-sanitized public artifact.

| Source | Formal execution SHA-256 | Public AE relation |
|---|---|---|
| Full-regeneration runner | `f31a8ba0fba272b4f07e86d60e37e3f2f87e2c94f5448f562205c3877261528b` | All eight Table 6 training/evaluation commands are unchanged. The maintained/public runner removes the later WP521 extension and its summary alias; four private path defaults are also placeholders in the public copy. |
| `main.py` | `2664c4c9746f1032ba08e3d5f92ee3c8adeb1e4b50450e9b1354ea30b528a308` | Byte-identical. |
| `quant/approx_linear.py` | `2166a171e39fcba89ef5cd88b689d66306001a128712799349f63953d7b67d3f` | Byte-identical. |
| `quant/ost_model_utils.py` | `2a3359e9b2514d506dcaf28963f692b60da658e8273068718e16056ac683d072` | Byte-identical. |

The formal job-start manifest directly records the runner, `main.py`, and
`quant/approx_linear.py` hashes. The KLT helper was not included in that
three-file job-start list, so its evidence is stated more narrowly: its remote
source modification time predates both formal job starts and both generated
W4A4 checkpoints; the same byte hash is present in the maintenance source and
public artifact; and the retained training/KLT logs exercise the CUDA
eigendecomposition without oneMKL or cuSOLVER failure.

The formal execution runner used the older combined scheduling script. Its
eight paper Table 6 commands, quantization flags, method order through DB-MixQ,
and skip conditions remain unchanged in the maintained runner. The only
functional correction removes the subsequent WP521 invocation and
WP521-inclusive summary alias: WP521 belongs to the separate Figure 12/rebuttal
archive, not Table 6. The public runner then differs from the maintained runner
only in default values for the environment, two model directories, and the
Hugging Face cache; those four private path defaults are placeholders. The
formal, maintained, and publication hashes remain different and visible instead
of being presented as byte-identical.
