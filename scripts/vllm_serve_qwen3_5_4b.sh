#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
uv run --group dev \
  vllm serve Qwen/Qwen3.5-4B \
  --host 0.0.0.0 \
  --port "${PORT:-8022}" \
  --max-model-len 131072 \
  --max-num-seqs "${MAX_NUM_SEQS:-256}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.9}" \
  --language-model-only \
  --gdn-prefill-backend triton \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking":true}'
