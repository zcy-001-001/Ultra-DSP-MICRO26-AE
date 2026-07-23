#!/usr/bin/env bash

# Portable toolchain loader shared by every Table 3 implementation.
# Point these variables to the setup scripts installed on the reviewer host.
export MODEL_BASE_PATH="${MODEL_BASE_PATH:-$PWD}"

source_if_set() {
    local var_name="$1"
    local script_path="${!var_name:-}"

    if [[ -z "${script_path}" ]]; then
        return 0
    fi
    if [[ ! -f "${script_path}" ]]; then
        echo "Configured setup script not found: ${var_name}=${script_path}" >&2
        return 1
    fi
    # shellcheck disable=SC1090
    source "${script_path}"
}

status=0
source_if_set XILINX_VITIS_SETTINGS || status=1
source_if_set XILINX_VIVADO_SETTINGS || status=1
source_if_set XRT_SETUP || status=1

return "${status}" 2>/dev/null || exit "${status}"
