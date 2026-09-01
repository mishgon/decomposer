#!/usr/bin/env bash
set -euo pipefail

# Kind's node containers must use a file-backed log driver under rootless
# Podman. The default journald relay can miss systemd's one-time ready line,
# causing intermittent "Multi-User System" startup failures.
if [[ "${1:-}" == "run" ]]; then
    shift
    exec /usr/local/bin/podman run --log-driver=k8s-file "$@"
fi

exec /usr/local/bin/podman "$@"
