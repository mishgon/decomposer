#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv_bin="${UV_BIN:-uv}"
revision=483b8a009ba3a97563edee3a19887e4862b8094a
if [ ! -d external/verl ]; then
    git clone https://github.com/volcengine/verl.git external/verl
    git -C external/verl checkout "$revision"
fi
test "$(git -C external/verl rev-parse HEAD)" = "$revision"
for patch_file in "$PWD"/opd/patches/*.patch; do
    if ! git -C external/verl apply --recount --reverse --check "$patch_file" 2>/dev/null; then
        git -C external/verl apply --recount --check "$patch_file"
        git -C external/verl apply --recount "$patch_file"
    fi
done
test -d .venv-opd || "$uv_bin" venv --python 3.12 .venv-opd
"$uv_bin" pip install --python .venv-opd/bin/python --override opd/overrides.txt -r opd/requirements.lock
"$uv_bin" pip install --python .venv-opd/bin/python --no-deps -e external/verl -e .
PYTHONPATH="$PWD:$PWD/src:$PWD/external/verl" .venv-opd/bin/python -c \
    'import torch, vllm, verl; print(torch.__version__, vllm.__version__, verl.__version__)'
# Agent Server and vLLM have incompatible web-stack pins. Keep the CPU workers separate.
test -d .venv-workers || "$uv_bin" venv --python 3.12 .venv-workers
"$uv_bin" pip install --python .venv-workers/bin/python -e . 'langgraph-cli[inmem]>=0.4.31'
