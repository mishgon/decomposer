#!/usr/bin/env bash
set -euo pipefail

# Local-environment diagnostic for the Gemma-4-31B Toolathlon-Verified result
# reported by Ornith. This intentionally runs the repository's 55 valid local
# tasks; it is not the official 108-task hosted evaluation.

repo=${REPO:-/home/matrosov/decomposer-ta-ornith-repro}
artifacts=${ARTIFACTS_DIR:-/home/matrosov/decomposer-ta/artifacts/gyms/toolathlon/ornith-gemma31-official-profile-local}
model=${MODEL:-/home/matrosov/models/gemma-4-31B-it}
gpu=${GPU:-4}
port=${PORT:-8050}
wait_pid=${WAIT_PID:-}

if [[ -n "$wait_pid" ]]; then
  echo "Waiting for benchmark process $wait_pid to finish..."
  while kill -0 "$wait_pid" 2>/dev/null; do
    sleep 60
  done
fi

export PATH="$repo/.venv/bin:/home/matrosov/.local/bin:$PATH"
cd "$repo"

exec "$repo/.venv/bin/python" "$repo/gyms/toolathlon/run.py" \
  --all-valid \
  --allow-missing-task-credentials \
  -n 3 \
  --purpose evaluation \
  --agent-mode simple \
  --simple-agent-implementation toolathlon \
  --subagent-provider vllm \
  --subagent-model "$model" \
  --subagent-thinking \
  --native-generation-profile toolathlon-verified-128k \
  --subagent-port "$port" \
  --subagent-gpu "$gpu" \
  --vllm-max-model-len 262144 \
  --subagent-recursion-limit 410 \
  --vllm-data-parallel-size 1 \
  --vllm-gpu-memory-utilization 0.90 \
  --concurrency 10 \
  --container-slots 8 \
  --n-jobs-per-worker 1000 \
  --agent-timeout 5400 \
  --episode-timeout 6000 \
  --image decomposer-toolathlon-ornith-repro:eff2ea0 \
  --max-tool-output-chars 100000 \
  --bench-artifacts-dir "$artifacts"
