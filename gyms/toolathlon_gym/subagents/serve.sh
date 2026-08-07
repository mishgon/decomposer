#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SUBAGENT_VENV="${SUBAGENT_VENV:-/opt/subagents}"

cd "$SCRIPT_DIR"
exec "$SUBAGENT_VENV/bin/langgraph" dev \
  --config langgraph.json \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-2024}" \
  --n-jobs-per-worker "${N_JOBS_PER_WORKER:-16}" \
  --no-browser \
  --no-reload
