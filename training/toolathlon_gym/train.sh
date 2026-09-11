#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source training/toolathlon_gym/router.sh
export RL_ROOT="$PWD"
export MODEL_CHECKPOINT_LINK="${MODEL_PATH:-$HOME/models/decomposer-4b-sft}"
export MODEL_PATH
MODEL_PATH="$(readlink -f "$MODEL_CHECKPOINT_LINK")"
test -f "$MODEL_PATH/config.json" || { echo "Missing model config: $MODEL_PATH" >&2; exit 1; }
export RL_DATA="${RL_DATA:-$PWD/artifacts/training/toolathlon_gym/pilot-data}"
export RL_ARTIFACTS="${RL_ARTIFACTS:-$PWD/artifacts/training/toolathlon_gym/pilot}"
export CUDA_VISIBLE_DEVICES="${POLICY_GPU:-0}"
RL_GYM_IMAGE=$(podman image inspect --format '{{.Id}}' "${RL_GYM_IMAGE:-decomposer-toolathlon-rl:latest}")
export RL_GYM_IMAGE
export PYTHONPATH="$PWD:$PWD/src:$PWD/external/verl${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PATH="$PWD/.venv-rl/bin:/usr/local/cuda/bin:$PATH"
export TENSORBOARD_DIR="$RL_ARTIFACTS/tensorboard"
mkdir -p "$RL_ARTIFACTS"
checkpoint_root="${RL_CHECKPOINT_ROOT:-$HOME/.local/share/decomposer/rl-checkpoints}"
if [ -n "${RL_CHECKPOINT_ROOT:-}" ] || [ -e "$checkpoint_root" ] || [ -L "$checkpoint_root" ]; then
    test -d "$checkpoint_root" || { echo "Checkpoint storage unavailable: $checkpoint_root" >&2; exit 1; }
    if [ ! -e "$RL_ARTIFACTS/checkpoints" ] && [ ! -L "$RL_ARTIFACTS/checkpoints" ]; then
        checkpoint_dir="$checkpoint_root/$(basename "$RL_ARTIFACTS")/checkpoints"
        test ! -e "$checkpoint_dir" || { echo "Checkpoint destination already exists: $checkpoint_dir" >&2; exit 1; }
        mkdir -p "$checkpoint_dir"
        ln -s "$(readlink -f "$checkpoint_dir")" "$RL_ARTIFACTS/checkpoints"
    fi
fi
.venv-rl/bin/python -m training.toolathlon_gym.record_run --pid "$$" --directory "$RL_ARTIFACTS" "$@"
exec > >(tee -a "$RL_ARTIFACTS/trainer.log") 2>&1
exec .venv-rl/bin/python -m verl.trainer.main_ppo \
    --config-path "$PWD/training/toolathlon_gym" --config-name config "$@"
