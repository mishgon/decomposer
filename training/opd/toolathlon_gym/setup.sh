#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
bash training/rl/toolathlon_gym/setup.sh
patch_file="$PWD/training/opd/toolathlon_gym/patches/verl-hosted-teacher.patch"
if ! git -C external/verl apply --recount --reverse --check "$patch_file" 2>/dev/null; then
    git -C external/verl apply --recount --check "$patch_file"
    git -C external/verl apply --recount "$patch_file"
fi
PYTHONPATH="$PWD:$PWD/src:$PWD/external/verl" .venv-rl/bin/python \
    -m unittest tests.test_opd_teacher tests.test_opd_recipe -q
