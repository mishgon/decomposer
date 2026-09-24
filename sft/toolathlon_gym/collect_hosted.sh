#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
set -a
source "${LMROUTER_ENV:-$HOME/.local/share/environment/lmrouter.env}"
set +a
export LLM_PROXY_URL="${COLLECTION_ROUTER_URL:-https://lmrouter.2a2i.org/v1}"
export LLM_PROXY_UNIX_SOCKET="${LLM_PROXY_UNIX_SOCKET:?Set the private model proxy socket}"
export DECOMPOSER_PARSE_QWEN_XML=0
unset DECOMPOSER_VLLM_BASE_URL DECOMPOSER_MAX_TOKENS OPENROUTER_API_KEY
export VLLM_API_KEY="$LLM_PROXY_MASTER_KEY"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec "${COLLECTION_PYTHON:-$PWD/.venv/bin/python}" -m sft.toolathlon_gym.run \
    --model Qwen/Qwen3.8-Flash-Next-NVFP4 \
    --subagent-model Qwen/Qwen3.5-4B-unlooped \
    --subagent-api-model Qwen/Qwen3.5-4B-unlooped \
    --subagent-base-url "$LLM_PROXY_URL" \
    --image "${COLLECTION_IMAGE:?Set a validated image ID}" \
    --agent-timeout 2700 --episode-timeout 3600 --startup-timeout 600 "$@"
