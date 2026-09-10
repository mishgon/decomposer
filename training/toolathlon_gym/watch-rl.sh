#!/usr/bin/env bash
set -euo pipefail
script_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
cd "${RL_ROOT:-$script_dir/../..}"
exec .venv-rl/bin/python -m training.toolathlon_gym.watch "$@"
