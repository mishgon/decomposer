#!/usr/bin/env bash
set -euo pipefail

repo=/home/matrosov/decomposer-ta
artifacts="$repo/artifacts/gyms/toolathlon"
decomposer_model=decomposer/qwen35-4b-sft-mixed-v3
tasks=(
  finalpool/find-alita-paper
  finalpool/canvas-art-quiz
  finalpool/identify-all-songs
  finalpool/git-bug-hunt
  finalpool/sales-accounting
  finalpool/train-ticket-plan
  finalpool/imagenet
  finalpool/canvas-new-students-notification
  finalpool/canvas-homework-grader-python
  finalpool/invoice-org
  finalpool/k8s-deployment-cleanup
  finalpool/canvas-art-manager
  finalpool/ppt-analysis
  finalpool/apply-phd-email
  finalpool/reimbursement-form-filler
  finalpool/cooking-guidance
  finalpool/woocommerce-update-cover
  finalpool/woocommerce-new-product
  finalpool/excel-data-transformation
  finalpool/meeting-assign
)

run_bench() {
  local subagent_model=$1
  "$repo/.venv/bin/python" "$repo/gyms/toolathlon/run.py" \
    --tasks "${tasks[@]}" \
    --score-against-all \
    -n 3 \
    --purpose evaluation \
    --agent-mode decomposer \
    --model "$decomposer_model" \
    --decomposer-provider vllm \
    --decomposer-base-url http://127.0.0.1:8040/v1 \
    --decomposer-prompt teacher \
    --no-decomposer-thinking \
    --subagent-provider vllm \
    --subagent-model "$subagent_model" \
    --no-subagent-thinking \
    --subagent-port 8030 \
    --subagent-gpu 3 \
    --vllm-max-model-len 256000 \
    --subagent-recursion-limit 410 \
    --vllm-data-parallel-size 1 \
    --vllm-gpu-memory-utilization 0.90 \
    --concurrency 16 \
    --container-slots 8 \
    --n-jobs-per-worker 1000 \
    --agent-timeout 2700 \
    --episode-timeout 6000 \
    --image decomposer-toolathlon-bench:latest \
    --max-tool-output-chars 100000 \
    --bench-artifacts-dir "$artifacts"
}

export PATH=/home/matrosov/.local/bin:$PATH
cd "$repo"
run_bench /home/matrosov/models/gemma-4-E2B-it
run_bench /home/matrosov/models/gemma-4-E4B-it
run_bench /home/matrosov/models/gemma-4-26B-A4B-it
