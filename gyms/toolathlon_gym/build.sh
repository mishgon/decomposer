#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

TOOLATHLON_BASE_IMAGE="${TOOLATHLON_BASE_IMAGE:-toolathlon-pack:latest}"
TOOLATHLON_DECOMPOSER_IMAGE="${TOOLATHLON_DECOMPOSER_IMAGE:-decomposer-toolathlon:latest}"

docker build \
  --tag "$TOOLATHLON_BASE_IMAGE" \
  "$PROJECT_ROOT/external/toolathlon_gym"

docker build \
  --file "$SCRIPT_DIR/Dockerfile" \
  --build-arg "TOOLATHLON_BASE_IMAGE=$TOOLATHLON_BASE_IMAGE" \
  --tag "$TOOLATHLON_DECOMPOSER_IMAGE" \
  "$PROJECT_ROOT"
