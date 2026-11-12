#!/usr/bin/env bash
# One OCCtopus worker owns retrieval, baseline, trainer and rollout processes.
set -euo pipefail
source_root="$(cd "$(dirname "$0")/../../.." && pwd)"
state="${1:?Persistent OPD directory}"
gym="${2:?Prepared pinned Gym checkout}"
assets="${3:?Prepared Wikipedia assets}"
name="${4:?New run name}"
shift 4
test -f "$assets/assets.json"
: "${HF_TOKEN:?Bind the Hugging Face secret}"
: "${LLM_PROXY_MASTER_KEY:?Bind the lmrouter secret}"
: "${CLEARML_CONFIG_FILE:?Bind the ClearML config file}"
bash "$source_root/opd/wideseek/occtopus/setup.sh" "$state" "$gym"
state="$(realpath "$state")"
commit="$(git -c safe.directory="$source_root" -C "$source_root" rev-parse HEAD)"
code="$state/code/$commit"
cd "$code"
export UV_CACHE_DIR="$state/cache/uv" UV_PYTHON_INSTALL_DIR="$state/cache/python"
export HF_HOME="$state/cache/huggingface" XDG_CACHE_HOME="$state/cache" TMPDIR="$state/tmp"
export RAY_TMPDIR=/tmp RAY_USE_UVLOOP=0
export PATH="$code/.venv-opd/bin:$PATH"
export PYTHONPATH="$code/opd/wideseek/dns:$code:$code/src:$code/external/Gym:$code/external/verl"
export LLM_PROXY_URL=https://lmrouter.2a2i.org/v1
export CLEARML_API_HOST=https://clearml.2a2i.org/api
export CLEARML_WEB_HOST=https://clearml.2a2i.org
export CLEARML_FILES_HOST=https://clearml.2a2i.org/files
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY=localhost,127.0.0.1,lmrouter.2a2i.org,clearml.2a2i.org
export no_proxy="$NO_PROXY"
python - <<'PY'
import os
import httpx
from clearml.utilities.pyhocon import ConfigFactory
config = ConfigFactory.parse_file(os.environ['CLEARML_CONFIG_FILE'])
with httpx.Client(trust_env=False, timeout=30) as client:
    response = client.get(os.environ['LLM_PROXY_URL'] + '/models',
        headers={'Authorization': 'Bearer ' + os.environ['LLM_PROXY_MASTER_KEY']})
    response.raise_for_status()
    response = client.post(os.environ['CLEARML_API_HOST'] + '/auth.login',
        auth=(config.get_string('api.credentials.access_key'), config.get_string('api.credentials.secret_key')),
        json={'expiration_sec': 60})
    response.raise_for_status()
    if not response.json().get('data', {}).get('token'):
        raise RuntimeError('ClearML authentication returned no token')
print('lmrouter and ClearML authentication passed', flush=True)
PY
model="$state/models/decomposer-manager-sft-v1.2.0"
if [[ ! -f "$model/opd_initialization.json" ]]; then
    python -m opd.wideseek.prepare_model --gpu 0 --output "$model"
fi
exec python -u -m opd.wideseek.local --root "$state/runs/$name" \
    --assets "$assets" --source "$state/data/depth/tasks.jsonl" --model "$model" "$@"
