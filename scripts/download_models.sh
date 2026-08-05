#!/usr/bin/env bash
set -euo pipefail

models=(
  Qwen/Qwen3.5-0.8B
  Qwen/Qwen3.5-2B
  Qwen/Qwen3.5-4B
  Qwen/Qwen3.6-35B-A3B-FP8
  google/gemma-4-E2B-it
  google/gemma-4-E4B-it
  google/gemma-4-12B-it
  google/gemma-4-26B-A4B-it
)

for model in "${models[@]}"; do
  printf '\nDownloading %s\n' "$model"
  if [[ "$model" == LiquidAI/* ]]; then
    HF_HUB_DISABLE_XET=1 uv tool run --from huggingface-hub hf download "$model"
  else
    uv tool run --from huggingface-hub hf download "$model"
  fi
done
