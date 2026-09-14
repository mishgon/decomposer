#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${RL_DATA:?Set RL_DATA to the prepared eight-task dataset}"
: "${RL_ARTIFACTS:?Set a fresh RL_ARTIFACTS directory}"
export RL_EPISODE_TIMEOUT=2700
finish() {
    PYTHONPATH=src:. .venv-rl/bin/python -m training.toolathlon_gym.report \
        --run "$RL_ARTIFACTS" --data "$RL_DATA" --samples 5
    .venv-rl/bin/python -m training.toolathlon_gym.select_checkpoints --run "$RL_ARTIFACTS"
}
trap finish EXIT
bash training/toolathlon_gym/train.sh \
    trainer.experiment_name=train8-n5 trainer.total_training_steps=16 trainer.total_epochs=16 \
    trainer.val_before_train=True trainer.test_freq=8 trainer.save_freq=8 \
    data.train_batch_size=8 data.val_batch_size=8 actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.rollout.n=5 actor_rollout_ref.rollout.val_kwargs.n=5 \
    actor_rollout_ref.rollout.agent.num_workers=8 "$@"
