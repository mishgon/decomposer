#!/usr/bin/env bash
# Prepare a persistent checkout using OCCtopus-managed Git sources.
set -euo pipefail
source_root="$(cd "$(dirname "$0")/../../.." && pwd)"
state="${1:?Persistent OPD directory}"
gym="${2:?Prepared pinned Gym checkout}"
mkdir -p "$state"
state="$(realpath "$state")"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git build-essential curl iproute2 ca-certificates
python -m pip install --disable-pip-version-check uv==0.9.7
commit="$(git -c safe.directory="$source_root" -C "$source_root" rev-parse HEAD)"
code="$state/code/$commit"
exec 9>"$state/setup-$commit.lock"
flock 9
if [[ -f "$state/ready-$commit" ]]; then exit 0; fi
test -f "$gym/resources_servers/workplace_assistant/utils.py"
mkdir -p "$state/code" "$state/cache" "$state/tmp"
export UV_CACHE_DIR="$state/cache/uv" UV_PYTHON_INSTALL_DIR="$state/cache/python"
export HF_HOME="$state/cache/huggingface" XDG_CACHE_HOME="$state/cache"
export TMPDIR="$state/tmp"
if [[ ! -d "$code/.git" ]]; then git -c safe.directory="$source_root" clone --no-hardlinks "$source_root" "$code"; fi
cd "$code"
git checkout --detach "$commit"
gym_commit="$(git ls-tree HEAD external/Gym | awk '{print $3}')"
test "$(git -c safe.directory="$gym" -C "$gym" rev-parse HEAD)" = "$gym_commit"
if [[ ! -d external/Gym/.git ]]; then
    git -c safe.directory="$gym" clone --no-hardlinks "$gym" external/Gym
fi
git -C external/Gym checkout --detach "$gym_commit"
export MAX_JOBS=8 TORCH_CUDA_ARCH_LIST=9.0 FLASH_ATTN_CUDA_ARCHS=90
bash opd/setup.sh
bash gyms/wideseek/setup.sh
export PYTHONPATH="$code:$code/src:$code/external/Gym:$code/external/verl"
if [[ ! -f "$state/data/depth/manifest.json" ]]; then
    .venv-opd/bin/python -m gyms.wideseek.prepare --source depth --output "$state/data/depth"
fi
.venv-opd/bin/python -c 'from opd.workplace_assistant.prepare import validation_rows; assert len(validation_rows()) == 20'
touch "$state/ready-$commit"
echo "OPD environment ready: $commit"
