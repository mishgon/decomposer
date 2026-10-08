#!/usr/bin/env bash
# Credentials come from the caller's configured lmrouter environment.
export LLM_PROXY_MASTER_KEY="${LLM_PROXY_MASTER_KEY:?Missing hosted model credential}"
if [[ -n "${LLM_PROXY_UNIX_SOCKET:-}" ]]; then
    export LLM_PROXY_UNIX_SOCKET
    test -S "$LLM_PROXY_UNIX_SOCKET" || { echo "Model proxy socket is missing" >&2; return 1; }
fi
export WS_SEARCH_URL="${WS_SEARCH_URL:-http://127.0.0.1:18080}"
export LANGSMITH_TRACING=false
export LANGGRAPH_NO_BROWSER=true
