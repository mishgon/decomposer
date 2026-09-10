#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${RL_DATA:?Set RL_DATA to the prepared 32-task dataset}"
: "${RL_ARTIFACTS:?Set a fresh RL_ARTIFACTS directory}"
export RL_EPISODE_TIMEOUT=2700
finish() {
    PYTHONPATH=src:. .venv-rl/bin/python -m training.toolathlon_gym.report \
        --run "$RL_ARTIFACTS" --data "$RL_DATA" --samples 4
    .venv-rl/bin/python -m training.toolathlon_gym.select_checkpoints --run "$RL_ARTIFACTS"
}
trap finish EXIT
bash training/toolathlon_gym/train.sh \
    trainer.experiment_name=train32 trainer.total_training_steps=4 \
    trainer.val_before_train=True trainer.test_freq=2 trainer.save_freq=2 \
    data.train_batch_size=8 actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.rollout.n=4 actor_rollout_ref.rollout.val_kwargs.n=4 "$@"
