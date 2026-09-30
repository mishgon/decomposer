#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
set -a
source "${LMROUTER_ENV:-$HOME/.local/share/environment/lmrouter.env}"
set +a
export LLM_PROXY_UNIX_SOCKET="${LLM_PROXY_UNIX_SOCKET:?Set the private model proxy socket}"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec "${COLLECTION_PYTHON:-$PWD/.venv/bin/python}" -m sft.toolathlon_gym.run \
    --model Qwen/Qwen3.8-Flash-Next-NVFP4 \
    --subagent-model Qwen/Qwen3.5-4B-unlooped \
    --subagent-api-model Qwen/Qwen3.5-4B-unlooped \
    --subagent-base-url https://lmrouter.2a2i.org/v1 \
    --image "${COLLECTION_IMAGE:?Set a validated image ID}" \
    --agent-timeout 2700 --episode-timeout 3600 --startup-timeout 600 "$@"
