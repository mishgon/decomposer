#!/usr/bin/env bash
set -euo pipefail

models=(
  Qwen/Qwen3.5-0.8B
  Qwen/Qwen3.5-2B
  Qwen/Qwen3.6-35B-A3B-FP8
  google/gemma-4-E2B-it
  google/gemma-4-26B-A4B-it
  LiquidAI/LFM2.5-1.2B-Instruct
  LiquidAI/LFM2.5-1.2B-Thinking
)

for model in "${models[@]}"; do
  printf '\nDownloading %s\n' "$model"
  if [[ "$model" == LiquidAI/* ]]; then
    HF_HUB_DISABLE_XET=1 uv tool run --from huggingface-hub hf download "$model"
  else
    uv tool run --from huggingface-hub hf download "$model"
  fi
done
