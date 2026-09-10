#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="${SUBAGENT_GPU:-1}"
export TOKENIZERS_PARALLELISM=false
export PATH="$PWD/.venv-rl/bin:/usr/local/cuda/bin:$PATH"
exec .venv-rl/bin/vllm serve "${SUBAGENT_MODEL_PATH:-$HOME/models/Qwen3.5-4B}" \
    --served-model-name Qwen/Qwen3.5-4B \
    --host 0.0.0.0 --port "${SUBAGENT_PORT:-8025}" \
    --max-model-len 262144 --gpu-memory-utilization "${SUBAGENT_GPU_MEMORY_UTILIZATION:-0.65}" \
    --language-model-only --enable-prefix-caching --enforce-eager \
    --additional-config '{"gdn_prefill_backend":"triton"}' \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking":false}'
