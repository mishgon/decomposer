#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
uv_bin="${UV_BIN:-uv}"
verl_revision=483b8a009ba3a97563edee3a19887e4862b8094a
if [ ! -d external/verl ]; then
    git clone https://github.com/volcengine/verl.git external/verl
    git -C external/verl checkout "$verl_revision"
fi
test "$(git -C external/verl rev-parse HEAD)" = "$verl_revision"
padding_patch="$PWD/training/toolathlon_gym/patches/verl-sdpa-padding.patch"
if ! git -C external/verl apply --recount --reverse --check "$padding_patch" 2>/dev/null; then
    git -C external/verl apply --recount --check "$padding_patch"
    git -C external/verl apply --recount "$padding_patch"
fi
if [ ! -d .venv-rl ]; then
    "$uv_bin" venv --python 3.12 .venv-rl
fi
"$uv_bin" pip install --python .venv-rl/bin/python \
    --override training/toolathlon_gym/overrides.txt \
    -r training/toolathlon_gym/requirements.lock
"$uv_bin" pip install --python .venv-rl/bin/python --no-deps -e external/verl -e .
.venv-rl/bin/python -c 'import torch, transformers, vllm, verl; print(torch.__version__, transformers.__version__, vllm.__version__, verl.__version__)'
