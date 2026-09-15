# Toolathlon Gym RL

Two experiment configs, one launcher:

| Config | Task pool | Schedule |
|---|---|---|
| `full.yaml` | `rl_task_pool.json`: 194 tasks | One epoch, 24 updates |
| `overfit.yaml` | `overfit_partial8_pool.json`: eight selected graded-reward tasks | 16 updates |

`overfit.yaml` inherits `full.yaml`; only the name and training duration change.
`agent_loop.yaml` registers our environment adapter, not another experiment.
Both evaluate their training tasks, **not a holdout**. Rewards stay native partial
scores; the >=0.9 threshold selects tasks, not training rewards.

## Prepare and run

From the repository root on Hertz-2, prepare either dataset (no GPU needed):

```bash
.venv-rl/bin/python -m training.toolathlon_gym.prepare_pilot \
  --task-pool training/toolathlon_gym/rl_task_pool.json \
  --train-tasks 194 --evaluate-training-tasks \
  --output artifacts/training/toolathlon_gym/full-data

.venv-rl/bin/python -m training.toolathlon_gym.prepare_pilot \
  --task-pool training/toolathlon_gym/overfit_partial8_pool.json \
  --train-tasks 8 --evaluate-training-tasks \
  --output artifacts/training/toolathlon_gym/overfit-data
```

Preparation refuses to overwrite datasets and records task IDs, pool hash, and
Gym revision. The full pool contains tasks with 1–4 successes out of five historical
SFT attempts. The overfit pool freezes the user's eight selected graded tasks.

Choose a free policy GPU and a fresh run directory:

```bash
RL_DATA="$PWD/artifacts/training/toolathlon_gym/overfit-data" \
RL_ARTIFACTS="$PWD/artifacts/training/toolathlon_gym/overfit-01" \
POLICY_GPU=0 bash training/toolathlon_gym/train.sh overfit
```

For full training use `full-data`, a new run directory, and `train.sh full`.
Hydra overrides follow the config name. Use tmux for unattended training.

## Shared recipe and provenance

Seeds:
- Timur's `tau2-gym/training/configs/grpo/champion.yaml` at
  `08cc4f6ab2ff34abbf50fdc87b62df6313eecb4b`.
- Agentic-rag's `occ-train/experiments/templates/colocate.sh`, branch
  `ic/feat/train-verl09-agentloop`, at `89d1812bcac1313319f66354fb37efbf7f8326da`.
  Its old `configs/rl.yaml` is not the current recipe.

Shared reference settings: all-linear LoRA rank 32/alpha 64, LR 1.5e-5,
`seq-mean-token-mean` loss averaging. GRPO without standard-deviation normalization
follows Timur; RAG exposes this as an experiment choice. Task batch and optimizer
minibatch are eight (Timur), not RAG's 64/32.

Intentional Gym differences: keep five attempts/task, rather than eight/sixteen;
keep tested single-GPU synchronous updates with async AgentLoop generation,
4096 prompt + 12288 response/observation budget, eager vLLM, SDPA, and disabled
policy prefix caching. Do not copy RAG's multi-GPU pipeline or adapter-only saves:
we retain optimizer state. The old run used LoRA 16/32 and LR 1e-5; this is a
changed recipe, not an identical rerun.

Qwen non-thinking sampling remains temperature 0.7, top-p 0.8, top-k 20; the adapter
also applies presence penalty 1.5. Subagents remain hosted Qwen3.5-4B.
Episode timeout is 2700 seconds and recursion limit is 410. Infrastructure errors
stay explicit; retain raw evaluations because native check denominators can vary.

Training generates 40 episodes/update. Evaluation uses five attempts/task before
training and every eight updates. Checkpoint cadence is also eight updates.
Overfit: 640 training + 120 evaluation = 760 episodes.
Full: 960 training + 3,880 evaluation = 4,840 episodes.
Full is a larger schedule, not a smoke run; choose epochs and evaluation cadence
explicitly before launching if that budget is unsuitable.
veRL drops the last incomplete training batch: 194 tasks yield 24 batches/epoch,
with two shuffled tasks omitted that epoch. All 194 are evaluated each time.

## Infrastructure and outputs

One-time setup: `bash training/toolathlon_gym/setup.sh`, then:

```bash
podman build -f training/toolathlon_gym/Dockerfile -t decomposer-toolathlon-rl:latest .
```

The launcher resolves `~/models/decomposer-4b-sft` once and records model, config,
source revision/diff, image, endpoint, dataset and process identity. Override
`MODEL_PATH` for another checkpoint. `router.sh` sources
`~/.local/share/environment/lmrouter.env` (override with `LMROUTER_ENV`).
Credentials travel by environment, not logs. No local subagent service is launched.
Existing episode startup caching and owned-container cleanup stay intact.

All outputs live under `RL_ARTIFACTS`: traces, evaluations, model calls,
TensorBoard, resolved Hydra config, logs and checkpoints.
`RL_CHECKPOINT_ROOT` or `~/.local/share/decomposer/rl-checkpoints` can point to the
large volume. Checkpoints are linked, not copied; Ray temp storage defaults there.
Existing storage is never automatically moved or overwritten.

`~/watch-rl.sh` remains the general watcher; `--run /path/to/run --once` selects
a snapshot. ETA includes evaluation. Batch rewards alone do not prove learning.
On exit the launcher writes `reward_report.json` and labels `best` (highest
complete training-probe reward) and `last`. Incomplete panels have no aggregate.
The report reads the actual evaluation repetition count from the saved config.

The old one-task smoke, train8 and train32 launchers are retired. Saved runs,
checkpoints, analysis tools and throughput measurements are untouched.
This consolidation is config validation, not a GPU training/resume test.
