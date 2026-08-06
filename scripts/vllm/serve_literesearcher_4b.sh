#!/usr/bin/env bash
set -euo pipefail

VLLM_USE_FLASHINFER_SAMPLER=0 \
VLLM_ATTENTION_BACKEND=FLASH_ATTN \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}" \
VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}" \
uv run --group dev \
  vllm serve simplex-ai-inc/LiteResearcher-4B \
  --host 0.0.0.0 \
  --port "${PORT:-8019}" \
  --max-model-len 512 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.1}" \
  --language-model-only \
  --gdn-prefill-backend triton \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser qwen3
