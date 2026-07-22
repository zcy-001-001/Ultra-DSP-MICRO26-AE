#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
package_root="$(cd "${script_dir}/../../.." && pwd)"
build_dir="${1:-${script_dir}/build_2023_2}"
output_dir="${2:-${package_root}/results/figures14_15/rerun/raw_hls_reports_2023_2}"

mkdir -p "${output_dir}"

copy_report() {
    local component="$1"
    local filename="$2"
    local source_path="${build_dir}/${component}/hls/syn/report/${filename}"
    if [[ ! -f "${source_path}" ]]; then
        echo "ERROR: missing report ${source_path}" >&2
        exit 3
    fi
    cp "${source_path}" "${output_dir}/${filename}"
}

copy_report initial_embedding_lookup initial_embedding_lookup_csynth.rpt
copy_report transformer_layer_pipeline transformer_layer_scheduler_csynth.rpt
copy_report transformer_layer_pipeline compute_core_w4a8_Pipeline_compute_n_core_w4a8_group_loop_w4a8_csynth.rpt
copy_report transformer_layer_pipeline compute_mha_csynth.rpt
copy_report transformer_layer_pipeline prepare_attn_input_csynth.rpt
copy_report transformer_layer_pipeline prepare_wo_input_csynth.rpt
copy_report transformer_layer_pipeline update_residual_csynth.rpt
copy_report transformer_layer_pipeline prepare_ffn_input_csynth.rpt
copy_report transformer_layer_pipeline compute_swiglu_prepare_w2_csynth.rpt
copy_report transformer_layer_pipeline rmsnorm_4096_s_csynth.rpt
copy_report transformer_layer_pipeline quantize_ptr_4096_1_s_csynth.rpt
copy_report transformer_layer_pipeline quantize_ptr_11008_1_s_csynth.rpt
copy_report final_norm_classifier final_norm_classifier_csynth.rpt

echo "REPORT_COLLECTION_PASS output_dir=${output_dir}"
