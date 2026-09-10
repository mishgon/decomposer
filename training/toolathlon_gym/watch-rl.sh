#!/usr/bin/env bash
set -euo pipefail
cd /home/matrosov/decomposer-rl
exec .venv-rl/bin/python -m training.toolathlon_gym.watch "$@"
