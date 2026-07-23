#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_SH="${ROOT_DIR}/env.sh"
PLATFORM="${PLATFORM:-/opt/xilinx/platforms/xilinx_u55c_gen3x16_xdma_3_202210_1/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm}"
HLS_CFG="${ROOT_DIR}/config/hls.cfg"
OOC_IMPLEMENT_ROOT="${ROOT_DIR}/ooc_implement"
WORK_DIR="${OOC_IMPLEMENT_ROOT}/freq_200MHz"
LOG_DIR="${WORK_DIR}/logs"
PAUSE_ON_FAIL="${PAUSE_ON_FAIL:-1}"

is_sourced() {
    [[ "${BASH_SOURCE[0]}" != "$0" ]]
}

setup_env() {
    if [[ -f "${ENV_SH}" ]]; then
        local old_pwd
        old_pwd="$(pwd)"
        cd "${ROOT_DIR}" || return 1
        source "${ENV_SH}" || return 1
        cd "${old_pwd}" || return 1
    fi
}

run_ooc_impl() {
    rm -rf "${WORK_DIR}"
    mkdir -p "${LOG_DIR}" || return 1

    echo "[ooc-impl] csynth start 200MHz"
    (
        cd "${ROOT_DIR}" || exit 1
        v++ -c --mode hls \
            --platform "${PLATFORM}" \
            --config "${HLS_CFG}" \
            --work_dir "${WORK_DIR}"
    ) > "${LOG_DIR}/csynth.stdout.log" 2>&1
    local status=$?
    if (( status != 0 )); then
        echo "[ooc-impl] csynth fail 200MHz -> ${LOG_DIR}/csynth.stdout.log" >&2
        return "${status}"
    fi

    echo "[ooc-impl] impl start 200MHz"
    (
        cd "${ROOT_DIR}" || exit 1
        vitis-run --mode hls \
            --impl \
            --config "${HLS_CFG}" \
            --work_dir "${WORK_DIR}"
    ) > "${LOG_DIR}/ooc_implement.stdout.log" 2>&1
    status=$?
    if (( status != 0 )); then
        echo "[ooc-impl] impl fail 200MHz -> ${LOG_DIR}/ooc_implement.stdout.log" >&2
        return "${status}"
    fi

    echo "[ooc-impl] done 200MHz"
}

pause_on_failure() {
    local status="$1"

    if (( status == 0 )); then
        return 0
    fi

    echo "[ooc-impl] synthesis/implementation failed."
    echo "[ooc-impl] csynth log : ${LOG_DIR}/csynth.stdout.log"
    echo "[ooc-impl] impl log   : ${LOG_DIR}/ooc_implement.stdout.log"

    if [[ "${PAUSE_ON_FAIL}" == "1" && -t 0 && -t 1 ]]; then
        echo "[ooc-impl] press Enter to return to the shell."
        read -r _
    fi

    return "${status}"
}

main() {
    local status=0

    if ! setup_env; then
        status=1
        pause_on_failure "${status}"
        return "${status}"
    fi

    if ! run_ooc_impl; then
        status=$?
        pause_on_failure "${status}"
        return "${status}"
    fi

    pause_on_failure "${status}"
    return "${status}"
}

if is_sourced; then
    main "$@"
    return $?
else
    main "$@"
    exit $?
fi
