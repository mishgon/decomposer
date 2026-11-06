#!/usr/bin/env bash
set -euo pipefail
script_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
cd "$script_dir/../.."
exec "${OPD_WATCH_PYTHON:-$PWD/.venv-opd/bin/python}" -m opd.synth.watch "$@"
