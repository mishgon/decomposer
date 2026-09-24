#!/usr/bin/env bash
# Two fixed smoke tasks, 8 attempts/task/epoch, 10 epochs (~20 updates).
set -euo pipefail
cd "$(dirname "$0")/.."
: "${POLICY_GPU:?Choose a free trainer GPU}"
: "${ROLLOUT_GPU:?Choose a free rollout GPU}"
: "${RL_ARTIFACTS:?Choose a fresh run directory}"
: "${RL_DATA:?Choose a fresh dataset directory}"
: "${RAY_TMPDIR:?Set a short Ray temp path on checkpoint storage}"
: "${RL_GYM_IMAGE:?Pin the baseline Gym image ID for this comparison}"
test ! -e "$RL_ARTIFACTS/run.json"
export SUBAGENT_MODEL="${SUBAGENT_MODEL:-Qwen/Qwen3.5-4B}"
export SUBAGENT_URL=https://lmrouter.2a2i.org/v1
export SUBAGENT_HOST=lmrouter.2a2i.org:176.108.242.226
export MODEL_PATH="${MODEL_PATH:-$HOME/models/decomposer-4b-sft}"
.venv-rl/bin/python -m rl.toolathlon_gym.task_profiles \
  --prepare "$RL_DATA" --profile smoke --groups-per-task 1
exec bash rl/toolathlon_gym/train.sh smoke \
  actor_rollout_ref.rollout.n=8 trainer.total_epochs=10 \
  ray_kwargs.ray_init.num_cpus=64 \
  transfer_queue.backend.SimpleStorage.num_data_storage_units=2 "$@"
