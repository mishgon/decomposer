#!/usr/bin/env bash
# Sourced before Ray starts; credentials stay out of Hydra config and run metadata.
router_env="${LMROUTER_ENV:-$HOME/.local/share/environment/lmrouter.env}"
if [[ -r "$router_env" ]]; then
    source "$router_env"
fi
export LLM_PROXY_MASTER_KEY="${LLM_PROXY_MASTER_KEY:?Configure LLM_PROXY_MASTER_KEY}"
if [[ -n "${LLM_PROXY_UNIX_SOCKET:-}" ]]; then
    export LLM_PROXY_UNIX_SOCKET
    test -S "$LLM_PROXY_UNIX_SOCKET" || { echo "Model proxy socket is missing" >&2; return 1; }
fi
