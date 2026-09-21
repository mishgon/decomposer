#!/usr/bin/env bash
# Source this on Hertz-2; never print credentials or enable shell tracing.
source "${LMROUTER_ENV:-$HOME/.local/share/environment/lmrouter.env}"
export LLM_PROXY_URL="${WS_MODEL_URL:-https://lmrouter.2a2i.org/v1}"
export WS_MODEL_HOST="${WS_MODEL_HOST:-lmrouter.2a2i.org:176.108.242.226}"
export WS_MODEL="${WS_MODEL:-Qwen/Qwen3.5-4B}"
: "${LLM_PROXY_MASTER_KEY:?Missing hosted model credential}"
export WS_ARTIFACT_ROOT="${WS_ARTIFACT_ROOT:-$PWD/artifacts/gyms/wideseek/runs}"
export WS_SEARCH_URL="${WS_SEARCH_URL:-http://127.0.0.1:18080}"
export LANGSMITH_TRACING=false
export LANGGRAPH_NO_BROWSER=true
