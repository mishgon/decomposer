#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
set -a
source "${LMROUTER_ENV:-$HOME/.local/share/environment/lmrouter.env}"
set +a
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec "${COLLECTION_PYTHON:-$PWD/.venv/bin/python}" -m sft.toolathlon_gym.run \
    --image "${COLLECTION_IMAGE:?Set a validated image ID}" \
    --agent-timeout 2700 --episode-timeout 3600 --startup-timeout 600 "$@"
