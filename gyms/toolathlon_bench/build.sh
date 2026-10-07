#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

TOOLATHLON_BASE_IMAGE="${TOOLATHLON_BASE_IMAGE:-docker.io/lockon0927/toolathlon-task-image:1016beta}"
TOOLATHLON_BENCH_IMAGE="${TOOLATHLON_BENCH_IMAGE:-decomposer-toolathlon-bench:latest}"

# The legacy builder ignores Dockerfile.dockerignore. Install it at the context
# root so local credentials in external/toolathlon/configs never enter a layer.
DOCKERIGNORE="$PROJECT_ROOT/.dockerignore"
if [ -e "$DOCKERIGNORE" ]; then
    echo "Error: $DOCKERIGNORE exists; remove it before building." >&2
    exit 1
fi
cp "$SCRIPT_DIR/Dockerfile.dockerignore" "$DOCKERIGNORE"
trap 'rm -f -- "$DOCKERIGNORE"' EXIT

docker build \
  --file "$SCRIPT_DIR/Dockerfile" \
  --build-arg "TOOLATHLON_BASE_IMAGE=$TOOLATHLON_BASE_IMAGE" \
  --tag "$TOOLATHLON_BENCH_IMAGE" \
  "$PROJECT_ROOT"
