#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_SH="${ROOT_DIR}/env.sh"
PLATFORM="${PLATFORM:-}"
IMPL_STRATEGY="${IMPL_STRATEGY:-Congestion_SpreadLogic_high}"
VIVADO_SYN_JOBS="${VIVADO_SYN_JOBS:-16}"
VIVADO_IMPL_JOBS="${VIVADO_IMPL_JOBS:-16}"
AUTO_CSYNTH="${AUTO_CSYNTH:-1}"
PAUSE_ON_FAIL="${PAUSE_ON_FAIL:-1}"
SYNTH_ROOT="${ROOT_DIR}/synth_200"
IMPLEMENT_ROOT="${ROOT_DIR}/implement_200"
SYNTH_LOG="${SYNTH_ROOT}/logs/csynth.stdout.log"
IMPLEMENT_LOG="${IMPLEMENT_ROOT}/logs/implement.stdout.log"
XO_PATH="${SYNTH_ROOT}/gemv_kernel.xo"

is_sourced() {
    [[ "${BASH_SOURCE[0]}" != "$0" ]]
}

setup_env() {
    if [[ -f "${ENV_SH}" ]]; then
        local old_pwd
        old_pwd="$(pwd)"
        cd "${ROOT_DIR}" || return 1
        # shellcheck disable=SC1090
        source "${ENV_SH}" || return 1
        cd "${old_pwd}" || return 1
    fi
}

require_platform() {
    if [[ -z "${PLATFORM}" ]]; then
        echo "[impl] set PLATFORM to the target U55C platform file before running implementation." >&2
        return 1
    fi
}

run_csynth() {
    mkdir -p "${SYNTH_ROOT}/logs" || return 1

    echo "[csynth] start"
    if ! (
        cd "${ROOT_DIR}" &&
        bash "${ROOT_DIR}/csynth_200.sh"
    ) > "${SYNTH_LOG}" 2>&1; then
        echo "[csynth] fail -> ${SYNTH_LOG}" >&2
        return 1
    fi
    echo "[csynth] done"
}

run_impl() {
    mkdir -p "${IMPLEMENT_ROOT}/logs" || return 1

    if [[ ! -f "${XO_PATH}" ]]; then
        echo "[impl] missing xo: ${XO_PATH}" >&2
        return 1
    fi

    echo "[impl] start"
    v++ -l -t hw \
        --platform "${PLATFORM}" \
        --config "${ROOT_DIR}/src/u55C.cfg" \
        --vivado.impl.strategies "${IMPL_STRATEGY}" \
        --vivado.synth.jobs "${VIVADO_SYN_JOBS}" \
        --vivado.impl.jobs "${VIVADO_IMPL_JOBS}" \
        "${XO_PATH}" \
        -o "${IMPLEMENT_ROOT}/GEMV.xclbin" \
        > "${IMPLEMENT_LOG}" 2>&1
    local status=$?
    if (( status != 0 )); then
        echo "[impl] fail -> ${IMPLEMENT_LOG}" >&2
        return "${status}"
    fi

    echo "[impl] done"
}

pause_on_failure() {
    local status="$1"

    if (( status == 0 )); then
        return 0
    fi

    echo "[impl] synthesis/implementation failed."
    echo "[impl] csynth log    : ${SYNTH_LOG}"
    echo "[impl] implement log : ${IMPLEMENT_LOG}"

    if [[ "${PAUSE_ON_FAIL}" == "1" && -t 0 && -t 1 ]]; then
        echo "[impl] press Enter to return to the shell."
        read -r _
    fi

    return "${status}"
}

xo_available() {
    [[ -f "${XO_PATH}" ]]
}

main() {
    local status=0

    if ! setup_env; then
        status=1
        pause_on_failure "${status}"
        return "${status}"
    fi

    if ! require_platform; then
        status=1
        pause_on_failure "${status}"
        return "${status}"
    fi

    if [[ "${AUTO_CSYNTH}" == "1" ]]; then
        if xo_available; then
            echo "[impl] found existing xo. skipping csynth: ${XO_PATH}"
        elif ! run_csynth; then
            status=1
            echo "[impl] csynth failed. implementation will not start." >&2
            pause_on_failure "${status}"
            return "${status}"
        fi
    fi

    if ! run_impl; then
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
