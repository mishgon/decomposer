#!/usr/bin/env bash
# Sourced before Ray starts; credentials stay out of Hydra config and run metadata.
router_env="${LMROUTER_ENV:-$HOME/.local/share/environment/lmrouter.env}"
if [[ -r "$router_env" ]]; then
    source "$router_env"
fi
export SUBAGENT_URL="${SUBAGENT_URL:-${LLM_PROXY_URL:-}}"
: "${SUBAGENT_URL:?Configure SUBAGENT_URL or LLM_PROXY_URL for hosted subagents}"
SUBAGENT_URL="${SUBAGENT_URL%/}"
[[ "$SUBAGENT_URL" == */v1 ]] || SUBAGENT_URL="$SUBAGENT_URL/v1"
export SUBAGENT_MODEL="${SUBAGENT_MODEL:-Qwen/Qwen3.5-4B}"
export SUBAGENT_HOST="${SUBAGENT_HOST:-}"
export VLLM_API_KEY="${VLLM_API_KEY:-${LLM_PROXY_MASTER_KEY:-}}"
: "${VLLM_API_KEY:?Configure VLLM_API_KEY or LLM_PROXY_MASTER_KEY}"
