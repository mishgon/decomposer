#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
case "${1:-}" in
  qdrant)
    : "${WS_ASSETS:?Set WS_ASSETS to the downloaded asset directory}"
    test -f "$WS_ASSETS/assets.json"
    export QDRANT__SERVICE__HOST=127.0.0.1 QDRANT__SERVICE__HTTP_PORT=16333 QDRANT__SERVICE__GRPC_PORT=16334
    export QDRANT__STORAGE__STORAGE_PATH="$WS_ASSETS/Wiki-2018-Corpus/qdrant/storage"
    export QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS=4
    chmod u+x "$WS_ASSETS/Wiki-2018-Corpus/qdrant/qdrant"
    cd "$WS_ASSETS/Wiki-2018-Corpus/qdrant"
    exec ./qdrant
    ;;
  retrieval)
    : "${WS_ASSETS:?Set WS_ASSETS}"
    export CUDA_VISIBLE_DEVICES=""
    exec .venv/bin/python -u -m gyms.wideseek.retrieval --assets "$WS_ASSETS"
    ;;
  workers)
    source gyms/wideseek/env.sh
    exec .venv/bin/langgraph dev --config gyms/wideseek/langgraph.json --host 127.0.0.1 \
      --port 18081 --no-browser --no-reload --n-jobs-per-worker 4
    ;;
  *) echo 'Usage: serve.sh {qdrant|retrieval|workers}' >&2; exit 2 ;;
esac
