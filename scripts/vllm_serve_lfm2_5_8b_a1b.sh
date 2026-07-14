#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
uv run --group serve \
  vllm serve LiquidAI/LFM2.5-8B-A1B \
  --host 0.0.0.0 \
  --port "${PORT:-8014}" \
  --max-model-len 65536 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.9}" \
  --enable-auto-tool-choice \
  --tool-call-parser lfm2 \
  --reasoning-parser qwen3
