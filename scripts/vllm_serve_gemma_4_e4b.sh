#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
uv run --group serve \
  vllm serve google/gemma-4-E4B-it \
  --host 0.0.0.0 \
  --port "${PORT:-8013}" \
  --max-model-len 65536 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.9}" \
  --limit-mm-per-prompt '{"image":0,"audio":0}' \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4 \
  --default-chat-template-kwargs '{"enable_thinking":true}'
