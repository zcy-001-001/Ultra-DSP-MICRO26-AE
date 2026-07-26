#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_SH="${ROOT_DIR}/env.sh"
PLATFORM="${PLATFORM:?Set PLATFORM to the U55C .xpfm file}"
HLS_CFG="${ROOT_DIR}/config/hls.cfg"
OOC_IMPLEMENT_ROOT="${ROOT_DIR}/ooc_implement"
WORK_DIR="${OOC_WORK_DIR:-${OOC_IMPLEMENT_ROOT}/freq_200MHz_work}"
HLS_WORK_DIR="${HLS_WORK_DIR:-${WORK_DIR}}"
LOG_DIR="${WORK_DIR}/logs"
RTL_DIR="${HLS_WORK_DIR}/hls/syn/verilog"
REPORT_DIR="${WORK_DIR}/ooc_impl_reports"
OOC_SYNTH_TCL="${ROOT_DIR}/ooc_synth.tcl"
OOC_SYNTH_XDC="${ROOT_DIR}/config/ooc_synth.xdc"
OOC_STAGE="${OOC_STAGE:-opt}"
SKIP_HLS="${SKIP_HLS:-0}"
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

run_ooc_implement() {
    rm -rf "${WORK_DIR}"
    mkdir -p "${LOG_DIR}" || return 1

    local status=0
    if [[ "${SKIP_HLS}" == "1" ]]; then
        if [[ ! -d "${RTL_DIR}" ]]; then
            echo "[ooc-impl] requested HLS RTL directory is missing: ${RTL_DIR}" >&2
            return 1
        fi
        echo "[ooc-impl] reuse HLS RTL from ${RTL_DIR}"
    else
        echo "[ooc-impl] csynth start 200MHz"
        (
            cd "${ROOT_DIR}" || exit 1
            v++ -c --mode hls \
                --platform "${PLATFORM}" \
                --config "${HLS_CFG}" \
                --work_dir "${HLS_WORK_DIR}"
        ) > "${LOG_DIR}/csynth.stdout.log" 2>&1
        status=$?
        if (( status != 0 )); then
            echo "[ooc-impl] csynth fail 200MHz -> ${LOG_DIR}/csynth.stdout.log" >&2
            return "${status}"
        fi
    fi

    echo "[ooc-impl] Vivado OOC start 200MHz; stop stage=${OOC_STAGE}"
    (
        cd "${ROOT_DIR}" || exit 1
        vivado -notrace -mode batch \
            -source "${OOC_SYNTH_TCL}" \
            -tclargs "${RTL_DIR}" "${REPORT_DIR}" "${OOC_SYNTH_XDC}" \
                "${OOC_STAGE}"
    ) > "${LOG_DIR}/ooc_implement.stdout.log" 2>&1
    status=$?
    if (( status != 0 )); then
        echo "[ooc-impl] implementation fail 200MHz -> ${LOG_DIR}/ooc_implement.stdout.log" >&2
        return "${status}"
    fi

    echo "[ooc-impl] done 200MHz through ${OOC_STAGE}; no later stage was run"
}

pause_on_failure() {
    local status="$1"

    if (( status == 0 )); then
        return 0
    fi

    echo "[ooc-impl] implementation failed."
    echo "[ooc-impl] csynth log  : ${LOG_DIR}/csynth.stdout.log"
    echo "[ooc-impl] Vivado log  : ${LOG_DIR}/ooc_implement.stdout.log"

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

    if run_ooc_implement; then
        status=0
    else
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
