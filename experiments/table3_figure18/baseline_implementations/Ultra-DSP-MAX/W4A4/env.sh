export MODEL_BASE_PATH=$PWD
# The machine-local setup commands are retained conceptually as configurable
# inputs instead of hard-coded installation paths:
# source <VITIS_SETTINGS>; source <VIVADO_SETTINGS>; source <XRT_SETUP>
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/env.sh"
