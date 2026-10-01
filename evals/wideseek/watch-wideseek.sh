#!/usr/bin/env bash
set -euo pipefail
script_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
cd "${WIDESEEK_REPO:-$script_dir/../..}"
exec .venv/bin/python -m evals.wideseek.watch --root "$PWD/artifacts" "$@"
