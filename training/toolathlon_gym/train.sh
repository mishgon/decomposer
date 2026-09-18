#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
case "${1:-}" in
    full|overfit) export RL_CONFIG="$1"; shift ;;
    *) echo "Usage: $0 {full|overfit} [Hydra overrides...]" >&2; exit 2 ;;
esac
: "${RL_DATA:?Set RL_DATA to the prepared dataset (see README)}"
: "${RL_ARTIFACTS:?Set RL_ARTIFACTS to the run directory}"
test -f "$RL_DATA/train.parquet"
test -f "$RL_DATA/evaluation.parquet"
export RL_EPISODE_TIMEOUT="${RL_EPISODE_TIMEOUT:-2700}"
source training/toolathlon_gym/router.sh
export RL_ROOT="$PWD"
export MODEL_CHECKPOINT_LINK="${MODEL_PATH:-$HOME/models/decomposer-4b-sft}"
export MODEL_PATH
MODEL_PATH="$(readlink -f "$MODEL_CHECKPOINT_LINK")"
test -f "$MODEL_PATH/config.json" || { echo "Missing model config: $MODEL_PATH" >&2; exit 1; }
export RL_DATA RL_ARTIFACTS
# Dependency source scanning can emit thousands of invalid-escape warnings.
# Keep other SyntaxWarnings and all runtime warnings/errors visible.
export PYTHONWARNINGS="${PYTHONWARNINGS:+$PYTHONWARNINGS,}ignore:invalid escape sequence:SyntaxWarning"
export CUDA_VISIBLE_DEVICES="${POLICY_GPU:-0}"
trainer_module=verl.trainer.main_ppo
if [[ "$RL_CONFIG" == overfit ]]; then
    : "${ROLLOUT_GPU:?Fully async overfit requires a separate ROLLOUT_GPU}"
    [[ "$ROLLOUT_GPU" != "$CUDA_VISIBLE_DEVICES" ]] || { echo "Trainer and rollout GPUs must differ" >&2; exit 1; }
    export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES,$ROLLOUT_GPU"
    # Hertz-2 direct NCCL transport stalls in the isolated transfer test.
    # Scope the verified shared-memory fallback to this job only.
    export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
    trainer_module=verl.experimental.fully_async_policy.fully_async_main
fi
export RL_TRAINER_MODULE="$trainer_module"
RL_GYM_IMAGE=$(podman image inspect --format '{{.Id}}' "${RL_GYM_IMAGE:-decomposer-toolathlon-rl:latest}")
export RL_GYM_IMAGE
export PYTHONPATH="$PWD:$PWD/src:$PWD/external/verl${PYTHONPATH:+:$PYTHONPATH}"
if [[ "$RL_CONFIG" == overfit ]]; then
    .venv-rl/bin/python -c 'import torch; import verl.checkpoint_engine.nccl_checkpoint_engine; from verl.utils.attention_utils import unpad_input; unpad_input(torch.ones(1, 2, 1), torch.ones(1, 2, dtype=torch.long))'
fi
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
if [ -d "$checkpoint_root" ]; then
    export RAY_TMPDIR="${RAY_TMPDIR:-$(readlink -f "$checkpoint_root")/ray-tmp}"
    mkdir -p "$RAY_TMPDIR"
fi
.venv-rl/bin/python -m training.toolathlon_gym.record_run --pid "$$" --directory "$RL_ARTIFACTS" "$@"
exec > >(tee -a "$RL_ARTIFACTS/trainer.log") 2>&1
finish() {
    .venv-rl/bin/python -m training.toolathlon_gym.throughput_cleanup "$RL_ARTIFACTS" || true
    .venv-rl/bin/python -m training.toolathlon_gym.report \
        --run "$RL_ARTIFACTS" --data "$RL_DATA" || true
    .venv-rl/bin/python -m training.toolathlon_gym.select_checkpoints --run "$RL_ARTIFACTS" || true
}
trap finish EXIT
.venv-rl/bin/python -m "$trainer_module" \
    --config-path "$PWD/training/toolathlon_gym" --config-name "$RL_CONFIG" \
    'hydra.searchpath=[pkg://verl.trainer.config]' \
    "trainer.experiment_name=$(basename "$RL_ARTIFACTS")" "$@"
