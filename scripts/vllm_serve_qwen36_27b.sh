#!/usr/bin/env bash
set -euo pipefail

uv run vllm serve Qwen/Qwen3.6-27B \
  --host 0.0.0.0 \
  --port 8011 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --gdn-prefill-backend triton \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --reasoning-config '{"reasoning_start_str":"<think>","reasoning_end_str":"</think>"}'
