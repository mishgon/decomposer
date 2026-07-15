#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
uv run --group models \
  vllm serve LiquidAI/LFM2.5-1.2B-Instruct \
  --host 0.0.0.0 \
  --port "${PORT:-8017}" \
  --max-model-len 32768 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.9}" \
  --enable-auto-tool-choice \
  --tool-call-parser lfm2 
