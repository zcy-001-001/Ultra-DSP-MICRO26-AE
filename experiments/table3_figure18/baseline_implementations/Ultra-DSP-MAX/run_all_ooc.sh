#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGETS=("$@")

if (( ${#TARGETS[@]} == 0 )); then
    TARGETS=(W3A4 W3A5 W4A4 W4A5 W5A5)
fi

for target in "${TARGETS[@]}"; do
    TARGET_DIR="${ROOT_DIR}/${target}"
    SCRIPT_PATH="${TARGET_DIR}/ooc_implement.sh"

    # The archived script required an executable bit and launched
    # `./ooc_implement.sh`. Use an explicit Bash invocation so a Windows clone
    # can run the published sources without preserving POSIX mode bits.
    if [[ ! -f "${SCRIPT_PATH}" ]]; then
        echo "[run-all-ooc] missing script: ${SCRIPT_PATH}" >&2
        exit 1
    fi

    echo "[run-all-ooc] start ${target}"
    (
        cd "${TARGET_DIR}" || exit 1
        bash "${SCRIPT_PATH}"
    )
    echo "[run-all-ooc] done ${target}"
done
