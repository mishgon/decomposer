#!/usr/bin/env bash
# Prepare a persistent checkout using OCCtopus-managed Git sources.
set -euo pipefail
source_root="$(cd "$(dirname "$0")/../../.." && pwd)"
state="${1:?Persistent OPD directory}"
gym="${2:?Prepared pinned Gym checkout}"
mkdir -p "$state"
state="$(realpath "$state")"
for tool in gcc g++ nvcc flock; do command -v "$tool" >/dev/null; done
exec 9>"$state/setup.lock"
flock 9
if [[ ! -x "$state/system/bin/git" || ! -x "$state/system/bin/curl" || ! -x "$state/system/bin/uv" ]]; then
    python - "$state" <<'PY'
import hashlib
import io
from pathlib import Path
import sys
import tarfile
import urllib.request
target = Path(sys.argv[1]) / 'bin/micromamba'
with urllib.request.urlopen('https://micro.mamba.pm/api/micromamba/linux-64/2.3.3', timeout=60) as response:
    data = response.read()
if hashlib.sha256(data).hexdigest() != 'e7274528ceb9c20d048a428d6c22d7e02e268f8ffb762c4c365422347c8b8ba2':
    raise RuntimeError('Micromamba archive checksum mismatch')
target.parent.mkdir(parents=True, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(data), mode='r:bz2') as archive:
    target.write_bytes(archive.extractfile('bin/micromamba').read())
target.chmod(0o755)
PY
    MAMBA_ROOT_PREFIX="$state/cache/mamba" "$state/bin/micromamba" create --no-rc -y \
        -p "$state/system" -c conda-forge git=2.56.1 curl=8.23.0 uv=0.9.7
fi
export PATH="$state/system/bin:$PATH"
commit="$(git -c safe.directory="$source_root" -C "$source_root" rev-parse HEAD)"
code="$state/code/$commit"
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
