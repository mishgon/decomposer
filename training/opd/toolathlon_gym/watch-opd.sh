#!/usr/bin/env bash
set -euo pipefail
script_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
cd "${OPD_ROOT:-$script_dir/../../..}"
exec .venv-rl/bin/python -m training.rl.toolathlon_gym.watch \
    --root "$PWD/artifacts/training/toolathlon_gym_opd" "$@"
