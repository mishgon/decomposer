#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
uv_bin="${UV_BIN:-uv}"
test -d .venv || "$uv_bin" venv --python 3.12 .venv
"$uv_bin" pip install --python .venv/bin/python torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
"$uv_bin" pip install --python .venv/bin/python -r gyms/wideseek/requirements.txt
revision=64875d346d5cafb06c1112f563b20d2c6360bfae
if [ ! -d external/RLinf ]; then
    git clone --filter=blob:none --no-checkout https://github.com/RLinf/RLinf.git external/RLinf
    git -C external/RLinf sparse-checkout set examples/agent/tools/search_local_server_qdrant
    git -C external/RLinf checkout "$revision"
fi
test "$(git -C external/RLinf rev-parse HEAD)" = "$revision"
echo 'Dependencies ready. Next: python -m gyms.wideseek.assets --root /large/disk/wideseek/assets'
