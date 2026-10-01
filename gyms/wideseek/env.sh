#!/usr/bin/env bash
# Source credentials and storage/service paths; model settings live in the registry.
source "${LMROUTER_ENV:-$HOME/.local/share/environment/lmrouter.env}"
export LLM_PROXY_MASTER_KEY="${LLM_PROXY_MASTER_KEY:?Missing hosted model credential}"
if [[ -n "${LLM_PROXY_UNIX_SOCKET:-}" ]]; then
    export LLM_PROXY_UNIX_SOCKET
    test -S "$LLM_PROXY_UNIX_SOCKET" || { echo "Model proxy socket is missing" >&2; return 1; }
fi
export WS_ARTIFACT_ROOT="${WS_ARTIFACT_ROOT:-$PWD/artifacts}"
export WS_SEARCH_URL="${WS_SEARCH_URL:-http://127.0.0.1:18080}"
export LANGSMITH_TRACING=false
export LANGGRAPH_NO_BROWSER=true
