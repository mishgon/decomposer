#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

TOOLATHLON_BASE_IMAGE="${TOOLATHLON_BASE_IMAGE:-docker.io/lockon0927/toolathlon-task-image:1016beta}"
TOOLATHLON_BENCH_IMAGE="${TOOLATHLON_BENCH_IMAGE:-decomposer-toolathlon-bench:latest}"
DECOMPOSER_REVISION="$(git -C "$PROJECT_ROOT" rev-parse HEAD)"

PINNED="$(git -C "$PROJECT_ROOT" ls-tree HEAD external/toolathlon | awk '{print $3}')"
CHECKED_OUT="$(git -C "$PROJECT_ROOT/external/toolathlon" rev-parse HEAD 2>/dev/null || true)"
if [ -n "$PINNED" ] && [ "$PINNED" != "$CHECKED_OUT" ]; then
    echo "Error: external/toolathlon is at ${CHECKED_OUT:-<missing>}, but the repository pins ${PINNED}." >&2
    echo "Run 'git submodule update --init external/toolathlon' before building." >&2
    exit 1
fi

# buildx dropped the --dockerignore flag, so install the context filter at
# the context root for the duration of the build.
DOCKERIGNORE="$PROJECT_ROOT/.dockerignore"
RESTORE_DOCKERIGNORE=""
if [ -e "$DOCKERIGNORE" ]; then
    RESTORE_DOCKERIGNORE="$DOCKERIGNORE.bench-backup"
    mv "$DOCKERIGNORE" "$RESTORE_DOCKERIGNORE"
fi
cp "$SCRIPT_DIR/Dockerfile.dockerignore" "$DOCKERIGNORE"
trap 'rm -f -- "$DOCKERIGNORE"; [ -n "$RESTORE_DOCKERIGNORE" ] && mv "$RESTORE_DOCKERIGNORE" "$DOCKERIGNORE"' EXIT

docker build \
  --file "$SCRIPT_DIR/Dockerfile" \
  --build-arg "TOOLATHLON_BASE_IMAGE=$TOOLATHLON_BASE_IMAGE" \
  --build-arg "DECOMPOSER_REVISION=$DECOMPOSER_REVISION" \
  --build-arg "TOOLATHLON_REVISION=$CHECKED_OUT" \
  --tag "$TOOLATHLON_BENCH_IMAGE" \
  "$PROJECT_ROOT"
