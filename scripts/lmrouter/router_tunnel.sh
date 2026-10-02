#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
    echo "Usage: bash router_tunnel.sh SSH_ALIAS SSH_KEY [ROUTER_HOST] [LISTEN_PORT]" >&2
    exit 2
fi

router_host="${3:-lmrouter.2a2i.org}"
listen_port="${4:-18443}"
if [[ ! "$listen_port" =~ ^[0-9]{1,5}$ ]] || (( 10#$listen_port < 1 || 10#$listen_port > 65535 )); then
    echo "LISTEN_PORT must be between 1 and 65535" >&2
    exit 2
fi

relay=
child=
stop() {
    trap - INT TERM HUP
    if [[ -n "$child" ]]; then
        kill "$child" 2>/dev/null || true
        wait "$child" 2>/dev/null || true
    fi
    if [[ -n "$relay" ]]; then
        kill "$relay" 2>/dev/null || true
        wait "$relay" 2>/dev/null || true
    fi
    exit "${1:-0}"
}
trap stop INT TERM HUP
python3 "$(dirname "$0")/gateway_relay.py" --host "$router_host" &
relay=$!
delay=2
while true; do
    started=$SECONDS
    ssh -N -T -i "$2" -o IdentitiesOnly=yes -o IdentityAgent=none -o BatchMode=yes \
        -o StrictHostKeyChecking=yes -o ConnectTimeout=10 \
        -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
        -R "127.0.0.1:$listen_port:127.0.0.1:18445" "$1" &
    child=$!
    finished=
    wait -n -p finished "$relay" "$child" || true
    if [[ "$finished" == "$relay" ]]; then
        stop 1
    fi
    child=
    # Keep retries short after a healthy session, bounded after repeated failures.
    if (( SECONDS - started >= 60 )); then delay=2; fi
    sleep "$delay" &
    child=$!
    wait "$child" || true
    child=
    delay=$((delay < 15 ? delay * 2 : 30))
done
