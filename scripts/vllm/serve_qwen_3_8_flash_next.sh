#!/usr/bin/env bash
set -euo pipefail

# Run on hertz-2; weights are mounted read-only from the shared disk.
mkdir -p /mnt/share14T-1/goncharov/models/.vllm-cache
docker run --rm --name goncharov-qwen38-flash-next \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=0,1,2,3 \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e HF_HUB_OFFLINE=1 \
  --shm-size 16g \
  -p 127.0.0.1:8025:8000 \
  -v /mnt/share14T-1/danserebro/models/Qwen3.8-Flash-Next-FP8:/model:ro \
  -v /mnt/share14T-1/goncharov/models/.vllm-cache:/root/.cache \
  vllm/vllm-openai:qwen38-flash-next /model \
  --served-model-name Qwen/Qwen3.8-Flash-Next-FP8 \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 4 \
  --moe-backend triton \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32768 \
  --max-num-seqs 64 \
  --enable-prefix-caching \
  --no-enable-flashinfer-autotune \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3
