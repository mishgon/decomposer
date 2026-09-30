#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 2 ]]; then
    echo "Usage: bash router_tunnel.sh USER@HERTZ SSH_KEY" >&2
    exit 2
fi

child=
stop() {
    trap - INT TERM
    if [[ -n "$child" ]]; then
        kill "$child" 2>/dev/null || true
        wait "$child" 2>/dev/null || true
    fi
    exit 0
}
trap stop INT TERM
delay=2
while true; do
    started=$SECONDS
    ssh -N -T -i "$2" -o IdentitiesOnly=yes -o IdentityAgent=none -o BatchMode=yes \
        -o StrictHostKeyChecking=yes -o ConnectTimeout=10 \
        -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
        -R 127.0.0.1:18443:176.108.242.226:443 "$1" &
    child=$!
    wait "$child" || true
    child=
    # Keep retries short after a healthy session, bounded after repeated failures.
    if (( SECONDS - started >= 60 )); then delay=2; fi
    sleep "$delay" &
    child=$!
    wait "$child" || true
    child=
    delay=$((delay < 15 ? delay * 2 : 30))
done
