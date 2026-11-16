#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
git submodule update --init external/WideSearch
patch="$PWD/gyms/widesearch/upstream.patch"
if git -C external/WideSearch apply --reverse --check "$patch" 2>/dev/null; then
    exit 0
fi
git -C external/WideSearch apply --check "$patch"
git -C external/WideSearch apply "$patch"
