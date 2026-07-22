# Minimal HLS Source Inventory

This directory contains the smallest identified source closure for regenerating
the three C-synthesis components that produced the selected Figure 14/15 HLS
reports. Generated RTL, IP, logs, object files, executables, bitstreams, model
weights, host applications, and the historical `synth/` tree are intentionally
excluded.

## Included files

| Relative path | Role |
|---|---|
| `src/config.h` | Llama-2-7B dimensions, 32 layers, and sequence-length configuration. |
| `src/typedefs.h` | Quantized weights, stream payloads, and accelerator data types. |
| `src/forward.h` | HLS operators, W4A8 matrix engine, quantization, normalization, and attention. |
| `src/forward.cpp` | Three top functions and transformer-layer scheduler. |
| `src/matmul_scheduler.h` | Matrix scheduler declarations used by the transformer component. |
| `src/tb_llama2_csim.cpp` | Testbench referenced by all three configuration files. |
| `llama2/hls_config1.cfg` | `initial_embedding_lookup` synthesis configuration. |
| `llama2/hls_config2.cfg` | `transformer_layer_pipeline` synthesis configuration. |
| `llama2/hls_config3.cfg` | `final_norm_classifier` synthesis configuration. |
| `run_csynth.sh` | Version-gated Vitis 2023.2 synthesis entry. |
| `collect_selected_reports.sh` | Copies only the 13 reports used by the simulator. |

The source also includes vendor headers such as `ap_int.h`, `hls_stream.h`,
`hls_math.h`, and `hls_half.h`. These are supplied by Vitis HLS and are not
redistributed here. The testbench conditionally includes `win.h` only on
Windows; the documented Linux C-synthesis flow does not require that file.

## Provenance and limits

`SOURCE_MANIFEST.sha256` records hashes of the copied upstream files. The first
line records the historical upstream entry-script hash for provenance only;
the script itself is replaced by the safer, rerunnable `run_csynth.sh`.

The historical frozen reports in `../inputs/raw_hls_reports/` identify Vitis
HLS 2024.1. This source package enables a clean 2023.2 rerun, but numerical
latency or report-format differences across tool versions must remain visible
and must not be silently calibrated away.

No standalone license file was present beside this remote source snapshot.
Before public AE release, the authors must include the applicable project
license or confirm that the top-level artifact license covers these files.
This packaging note is not a license grant.

The source snapshot is sufficient for C synthesis. Full C simulation or board
execution additionally requires model/checkpoint inputs, optional Windows
compatibility code, host software, platform installation, and hardware build
artifacts; those are outside Figures 14/15 and are not claimed here.
