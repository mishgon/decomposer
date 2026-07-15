#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}" \
uv run --group models \
  vllm serve Qwen/Qwen3.6-35B-A3B-FP8 \
  --host 0.0.0.0 \
  --port "${PORT:-8019}" \
  --max-model-len 65536 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.9}" \
  --language-model-only \
  --gdn-prefill-backend triton \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3
