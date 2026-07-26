#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_BASE_PATH="${ROOT_DIR}"

if [[ -n "${VITIS_SETTINGS:-}" ]]; then
    # shellcheck disable=SC1090
    source "${VITIS_SETTINGS}"
fi
if [[ -n "${VIVADO_SETTINGS:-}" ]]; then
    # shellcheck disable=SC1090
    source "${VIVADO_SETTINGS}"
fi
if [[ -n "${XRT_SETUP:-}" ]]; then
    # shellcheck disable=SC1090
    source "${XRT_SETUP}"
fi

command -v v++ >/dev/null || {
    echo "v++ not found; set VITIS_SETTINGS or source Vitis 2023.2." >&2
    return 1
}
command -v vivado >/dev/null || {
    echo "vivado not found; set VIVADO_SETTINGS or source Vivado 2023.2." >&2
    return 1
}
: "${PLATFORM:?Set PLATFORM to the U55C .xpfm file}"
