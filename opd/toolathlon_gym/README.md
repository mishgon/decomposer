# Toolathlon Gym OPD

The SFT decomposer executes Gym tasks with hosted Qwen3.5-4B-unlooped
non-thinking subagents. Hosted Qwen3.8 Flash Next scores the exact student token
IDs. Only student-generated positions contribute to the distillation loss;
tool observations and subagent reports are masked. Native reward measures
quality but does not enter the OPD loss.

## Files and Recipe

`teacher.py` validates prompt logprobs and saves every teacher call.
`manager.py` connects this client to veRL. Both teacher and subagent clients
use `src/decomposer/models.py`, including its tunnel transport and two retries.
There are no OPD-specific model URL, IP, credential or sampling overrides.

`rl_recipe.yaml` links to `rl/toolathlon_gym/recipe.yaml`. OPD reuses RL's
environment adapter, task preparation, launch infrastructure and monitoring.
Its `full.yaml` changes only the objective and rollout schedule: sampled-token
k1 policy-gradient distillation, no task rewards, one rollout/task/epoch, ten
epochs, and two concurrent episodes. No teacher GPU is allocated. Smoke targets
20 updates on the same two tasks as RL. Evaluation uses eight attempts/task
before training and every eight updates; it reuses training tasks.

LoRA 32/64, LR 1.5e-5, two optimizer passes, 45-minute episodes and saving every
update remain unchanged. `full` uses the frozen 346-task pool. The independent
RL objective and the pristine Timur reference remain untouched.

## Setup and Launch

Configure the shared [lmrouter tunnel](../../README.md). Export the private
`LLM_PROXY_MASTER_KEY` and, when needed, `LLM_PROXY_UNIX_SOCKET`.

```bash
bash opd/toolathlon_gym/setup.sh
export POLICY_GPU=2 ROLLOUT_GPU=3  # choose two actually free GPUs
export RL_GYM_IMAGE=your-tested-gym-image
export RL_DATA="$PWD/artifacts/training/toolathlon_gym_opd/smoke-data"
export RL_ARTIFACTS="$PWD/artifacts/training/toolathlon_gym_opd/smoke-01"
bash opd/toolathlon_gym/train.sh smoke
```

The launcher prepares one training group/task if the dataset directory does not
exist. Use a fresh run directory. Setup applies an opt-in teacher-manager hook
to pinned veRL; it changes no RL behavior when distillation is disabled.

## Teacher Preflight

```bash
PYTHONPATH=src:. .venv-rl/bin/python -m opd.toolathlon_gym.teacher \
  --trace /path/to/episode/trace.json \
  --tokenizer /path/to/decomposer-4b-sft \
  --output artifacts/training/toolathlon_gym_opd/preflight/teacher-score.json
```

Every scored token must match both its numeric ID and decoded meaning. The
first token has no preceding context and is excluded. Alignment failures stay
errors, never zero scores. Prompt scoring requests one unused generated token;
its temperature does not change prompt logprobs. No chat-template re-rendering
or teacher reasoning is inserted into student trajectories.

## Monitoring

```bash
bash opd/toolathlon_gym/watch-opd.sh
bash opd/toolathlon_gym/watch-opd.sh --run /path/to/run --once
```

ClearML logs under `decomposer-toolathlon-gym-opd`. Teacher requests/responses
and failure types go into `teacher_calls/`, without authentication headers.
Episode traces, raw native scores, checkpoints and monitoring use the same
formats as RL. A saved update verifies execution, not learning or overfitting.
