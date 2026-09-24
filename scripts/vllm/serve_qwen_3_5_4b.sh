#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
uv run --group dev \
  vllm serve Qwen/Qwen3.5-4B \
  --host 0.0.0.0 \
  --port "${PORT:-8024}" \
  --tensor-parallel-size 1 \
  --data-parallel-size 1 \
  --max-model-len 131072 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 192 \
  --attention-backend FLASH_ATTN \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --default-chat-template-kwargs '{"enable_thinking":false}'
