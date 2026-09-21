#!/usr/bin/env bash
set -eu
REPO="${WIDESEEK_REPO:-$HOME/decomposer-wideseek}"
exec python3 "$REPO/scripts/watch_wideseek.py" \
  --root "$REPO/artifacts/gyms/wideseek/runs" "$@"
