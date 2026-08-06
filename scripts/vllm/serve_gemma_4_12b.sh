#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
uv run --group dev \
  vllm serve google/gemma-4-12B-it \
  --host 0.0.0.0 \
  --port "${PORT:-8022}" \
  --max-model-len 32768 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.9}" \
  --language-model-only \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4 \
  --default-chat-template-kwargs '{"enable_thinking":true}'
