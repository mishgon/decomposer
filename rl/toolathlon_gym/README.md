# Toolathlon Gym RL

veRL trains the decomposer. The shared Gym runs isolated tasks and scores the
result. Hosted Qwen3.5-4B-unlooped non-thinking subagents use the explicit profile
in `src/decomposer/models.py`. RL has no separate provider or sampling registry.

## Files

| File | Responsibility |
| --- | --- |
| `recipe.yaml` | Shared model, optimizer, rollout and logging settings |
| `full.yaml`, `cold-start.yaml`, `smoke.yaml` | Experiment labels; same recipe |
| `task_pools/*.json`, `task_profiles.py` | Frozen task selection and parquet preparation |
| `agent_loop.py` | Connect veRL policy generation to the Gym and Decomposer |
| `policy.py` | Keep raw policy tokens, tool-call parsing and observation masks |
| `train.sh`, `setup.sh` | Launch training and install pinned dependencies |
| `watch.py`, `report.py`, `select_checkpoints.py` | Progress, summaries, best/last checkpoints |

Reusable lifecycle and native reward parsing live in `gyms/toolathlon_gym`.
Evaluation uses `evals/toolathlon_gym/run.py`; off-policy collection uses
`sft/toolathlon_gym/run.py`. The retired pilot/stress launchers and duplicate
SFT-checkpoint evaluator are removed. Existing artifacts remain readable.

## Task Pools and Recipe

`full` contains 346 tasks with an observed intermediate native score, excluding
tasks with five scores >=0.9. `cold-start` contains 191 of those tasks with mean
runtime <=30 minutes and score range >0.1. `smoke` contains two fixed tasks with
partial rewards and observed full successes. These frozen pools are unchanged.

All profiles evaluate their training pool; these are fitting scores, not held-out
validation. Training uses native partial reward, not a thresholded pass label.

The recipe retains GRPO groups of 32, three groups/task/epoch (96 attempts), one
epoch, LoRA rank 32/alpha 64, LR 1.5e-5 and two PPO passes per group. Evaluation
runs eight attempts/task before training and every eight updates. Each episode
has a 45-minute agent timeout and recursion limit 410. Policy token budgets are
4096 prompt + 12288 response/observations. Policy sampling is temperature 0.7,
top-p 0.8, top-k 20, presence penalty 1.5; thinking and prefix caching stay off.

Fully async training needs separate trainer and rollout GPUs. Two concurrent
GRPO groups permit up to 64 episodes, not two episodes. Review total evaluation
cost before launching `full`: every panel covers the entire selected pool.

Sources: Timur's `tau2-gym` champion recipe at
`08cc4f6ab2ff34abbf50fdc87b62df6313eecb4b` and agentic-rag's `occ-train`
`colocate.sh` at `89d1812bcac1313319f66354fb37efbf7f8326da`. The unchanged Timur
reference is in `references/timur-champion.yaml`. Our async scheduling differs
from that reference; this refactor does not change either learning objective.

## Setup and Launch

Follow the repository [lmrouter tunnel instructions](../../README.md) once.
Set `LLM_PROXY_MASTER_KEY` privately. If using the relay, export
`LLM_PROXY_UNIX_SOCKET`; the launcher passes the socket and credentials into
workers without putting secrets in logs. `router.sh` can load
`~/.local/share/environment/lmrouter.env` or `LMROUTER_ENV`.

```bash
bash rl/toolathlon_gym/setup.sh
.venv-rl/bin/python -m rl.toolathlon_gym.task_profiles \
  --profile smoke --prepare artifacts/training/toolathlon_gym/smoke-data

RL_DATA="$PWD/artifacts/training/toolathlon_gym/smoke-data" \
RL_ARTIFACTS="$PWD/artifacts/training/toolathlon_gym/smoke-01" \
POLICY_GPU=2 ROLLOUT_GPU=3 RL_GYM_IMAGE=your-tested-gym-image \
  bash rl/toolathlon_gym/train.sh smoke ray_kwargs.ray_init.num_cpus=64
```

Choose actually free GPUs. `MODEL_PATH` defaults to `~/models/decomposer-4b-sft`
and resolves the symlink once at launch. Preparation refuses to overwrite a
dataset. The launcher checks its pool hash and task/group counts. Use tmux for
unattended runs; Hydra overrides follow the profile name.

## Artifacts and Monitoring

```bash
bash rl/toolathlon_gym/watch-rl.sh
bash rl/toolathlon_gym/watch-rl.sh --run /path/to/run --once
python scripts/plot_rl.py --run /path/to/run
```

Each run stores source/config provenance, policy tokens/logprobs/masks, worker
responses, raw native evaluations, trainer logs, TensorBoard and ClearML metrics.
Failed OPD calls also retain their request and error type. Checkpoints contain
model, optimizer and training state after every update. `best` means best
complete fixed-panel native reward; `last` means newest checkpoint. Configure
`RL_CHECKPOINT_ROOT` for external storage.

Remote runs stop before partial scoring. Failure to stop them prevents scoring.
Known invalid agent-output paths score zero; evaluator and provider failures
stay explicit errors. Cleanup removes only resources recorded for this run.
