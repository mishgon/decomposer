#!/usr/bin/env bash
# One allocated GPU serves the manager; researchers and judge use lmrouter.
set -euo pipefail
source_root="$(cd "$(dirname "$0")/../../.." && pwd)"
root="${1:?New persistent run directory}"
shared="${2:?Prepared shared runtime and cache directory}"
pool="${3:?Frozen seen tasks.jsonl}"
models="${4:?Shared merged checkpoint directory}"
inference="${5:?Prepared vLLM Python executable}"
shift 5
: "${LLM_PROXY_MASTER_KEY:?Bind the lmrouter secret}"
: "${AGENTIC_RAG_API_KEY:?Bind the research API secret}"
: "${CLEARML_CONFIG_FILE:?Bind the ClearML config file}"
export PATH="$shared/system/bin:$PATH"
export UV_CACHE_DIR="$shared/cache/uv" UV_PYTHON_INSTALL_DIR="$shared/cache/python"
export HF_HOME="$shared/cache/huggingface" XDG_CACHE_HOME="$shared/cache"
export TMPDIR="$shared/tmp" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export NO_PROXY=127.0.0.1,localhost,lmrouter.2a2i.org,clearml.2a2i.org
export no_proxy="$NO_PROXY"
export VLLM_HOST=127.0.0.1 VLLM_HOST_IP=127.0.0.1 GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo NCCL_IB_DISABLE=1
export CLEARML_API_HOST=https://clearml.2a2i.org/api CLEARML_WEB_HOST=https://clearml.2a2i.org CLEARML_FILES_HOST=https://clearml.2a2i.org/files
mkdir -p "$root" "$TMPDIR"
exec > >(tee -a "$root/bootstrap.log") 2>&1
commit="$(git -c safe.directory="$source_root" -C "$source_root" rev-parse HEAD)"
code="$root/code/$commit"
if [[ ! -d "$code/.git" ]]; then
    mkdir -p "$root/code"
    git -c safe.directory="$source_root" -c safe.directory="$source_root/.git" clone --no-hardlinks "$source_root" "$code"
fi
cd "$code"
git checkout --detach "$commit"
export PYTHONPATH="$code/sft/dns:$code/src:$code"
if [[ ! -x "$root/workers/bin/python" ]]; then uv venv --python 3.12 "$root/workers"; fi
uv pip install --python "$root/workers/bin/python" -r gyms/wideseek/requirements.txt
"$inference" -c 'import vllm; assert vllm.__version__ == "0.19.0"; print("Verified vLLM", vllm.__version__)'
exec "$root/workers/bin/python" -u -m evals.wideseek.occtopus.seen \
    --root "$root" --data "$pool" --models "$models" --inference-python "$inference" "$@"
