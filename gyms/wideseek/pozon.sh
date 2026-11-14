#!/usr/bin/env bash
# Single RTX 5080 (16 GiB), WSL2 (30 GiB RAM).
set -euo pipefail
cd "$(dirname "$0")/../.."
export WS_ASSETS="${WS_ASSETS:-$HOME/projects/vkr-wideseek/assets}"
export WS_MODEL_BACKEND=local
export VLLM_HOST=127.0.0.1
export PATH="$HOME/.local/bin:$PATH"
case "${1:-}" in
  setup-vllm)
    test -d .venv-vllm || uv venv --python 3.12 .venv-vllm
    uv pip install --python .venv-vllm/bin/python \
      'https://github.com/vllm-project/vllm/releases/download/v0.24.0/vllm-0.24.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl' \
      --extra-index-url https://download.pytorch.org/whl/cu129 \
      --index-strategy unsafe-best-match
    ;;
  vllm)
    export CUDA_VISIBLE_DEVICES=0
    exec .venv-vllm/bin/vllm serve Qwen/Qwen3.5-4B \
      --host 127.0.0.1 --port 8024 --dtype bfloat16 \
      --max-model-len 16384 --max-num-seqs 1 --max-num-batched-tokens 2048 \
      --gpu-memory-utilization 0.85 --enforce-eager \
      --language-model-only --enable-prefix-caching --enable-chunked-prefill \
      --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
      --default-chat-template-kwargs '{"enable_thinking":false}'
    ;;
  pages)
    exec gyms/wideseek/.venv/bin/python -u -m gyms.wideseek.pages \
      "$WS_ASSETS/Wiki-2018-Corpus/wiki_webpages.jsonl"
    ;;
  qdrant)
    test -f "$WS_ASSETS/assets.json"
    docker run -d --name vkr-wideseek-qdrant --memory 8g --memory-swap 8g --cpus 4 \
      -p 127.0.0.1:16333:6333 -p 127.0.0.1:16334:6334 \
      -e QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS=4 \
      -v "$WS_ASSETS/Wiki-2018-Corpus/qdrant/storage:/qdrant/storage" \
      qdrant/qdrant:v1.16.2
    # Updating through the API lets Qdrant migrate the imported segments itself.
    for ((i=0; i<180; i++)); do
      if curl -fsS http://127.0.0.1:16333/readyz >/dev/null 2>&1; then
        exec curl --fail-with-body -X PATCH \
          http://127.0.0.1:16333/collections/wiki_collection_m32_cef512 \
          -H 'Content-Type: application/json' \
          -d '{"vectors":{"":{"on_disk":true}},"hnsw_config":{"on_disk":true},"optimizers_config":{"max_optimization_threads":1}}'
      fi
      sleep 2
    done
    echo 'Qdrant did not become ready; inspect docker logs vkr-wideseek-qdrant' >&2
    exit 1
    ;;
  retrieval|workers)
    exec bash gyms/wideseek/serve.sh "$1"
    ;;
  *) echo 'Usage: pozon.sh {setup-vllm|vllm|pages|qdrant|retrieval|workers}' >&2; exit 2 ;;
esac
