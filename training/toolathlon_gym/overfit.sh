#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export RL_DATA="${RL_DATA:-$PWD/artifacts/training/toolathlon_gym/overfit-data}"
export RL_ARTIFACTS="${RL_ARTIFACTS:-$PWD/artifacts/training/toolathlon_gym/overfit-4}"
export RL_EPISODE_TIMEOUT="${RL_EPISODE_TIMEOUT:-300}"
exec bash training/toolathlon_gym/train.sh \
  trainer.experiment_name=overfit-smoke trainer.total_training_steps=1 \
  trainer.val_before_train=False trainer.test_freq=1 trainer.save_freq=1 \
  data.train_batch_size=1 data.val_batch_size=1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=1 \
  actor_rollout_ref.actor.optim.lr=1e-4 \
  actor_rollout_ref.rollout.n=2 actor_rollout_ref.rollout.val_kwargs.n=2 "$@"
