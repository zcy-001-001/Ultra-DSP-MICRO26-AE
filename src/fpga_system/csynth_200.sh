#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_SH="${ROOT_DIR}/env.sh"
PLATFORM="${PLATFORM:-}"

if [[ -f "${ENV_SH}" ]]; then
    # shellcheck disable=SC1090
    source "${ENV_SH}"
fi

if [[ -z "${PLATFORM}" ]]; then
    echo "Set PLATFORM to the target U55C platform file before running synthesis." >&2
    exit 1
fi

mkdir -p "${ROOT_DIR}/synth_200"

v++ -c --mode hls \
    --platform "${PLATFORM}" \
    --config "${ROOT_DIR}/config/hls_200MHz.cfg" \
    --work_dir "${ROOT_DIR}/synth_200"
