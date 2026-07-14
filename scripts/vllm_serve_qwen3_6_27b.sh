#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
uv run --group serve \
  vllm serve Qwen/Qwen3.6-27B \
  --host 0.0.0.0 \
  --port "${PORT:-8011}" \
  --max-model-len 65536 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.9}" \
  --language-model-only \
  --trust-remote-code \
  --gdn-prefill-backend triton \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --reasoning-config '{"reasoning_start_str":"<think>","reasoning_end_str":"</think>"}'
