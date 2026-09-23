#!/usr/bin/env bash
set -euo pipefail

# Non-thinking Qwen3.5-4B subagent worker.
#
# --enable-prefix-caching matters more than it looks: the Decomposer re-sends a
# growing context every ReAct turn, so without it total prefill is quadratic in
# turn count. Subagents get a fresh context per spawn, so their shared prefix is
# only the system block plus tool schemas -- still worth caching, since the tau2
# schemas are large.
#
# Served WITHOUT a reasoning parser: the chat template disables thinking.

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
uv run --group dev \
  vllm serve Qwen/Qwen3.5-4B \
  --host 0.0.0.0 \
  --port "${PORT:-8025}" \
  --max-model-len "${MAX_MODEL_LEN:-128000}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.9}" \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --default-chat-template-kwargs '{"enable_thinking":false}'
