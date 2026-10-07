#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
uv_bin="${UV_BIN:-uv}"
gym_venv=gyms/wideseek/.venv
test -d "$gym_venv" || "$uv_bin" venv --python 3.12 "$gym_venv"
"$uv_bin" pip install --python "$gym_venv/bin/python" torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
"$uv_bin" pip install --python "$gym_venv/bin/python" -r gyms/wideseek/requirements.txt
git submodule update --init --depth 1 external/RLinf
echo 'Dependencies ready. Next: gyms/wideseek/.venv/bin/python -m gyms.wideseek.assets --root /large/disk/wideseek/assets'
