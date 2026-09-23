# Toolathlon Gym RL

Three experiment configs, one fully async recipe and launcher:

| Config | Task pool | Schedule |
|---|---|---|
| `full.yaml` | 346 observed partial-score tasks, excluding five >=90% scores | 96 attempts/task/epoch |
| `cold-start.yaml` | 191 full-pool tasks: mean time <=30 min, range >0.1 | Same recipe |
| `smoke.yaml` | Two partial-reward tasks with observed full successes | Same recipe |

`cold-start.yaml` and `smoke.yaml` inherit every training setting from `full.yaml`.
Only task selection and the experiment label differ. Frozen pools live in `task_pools/`.
Future smoke uses historical score means 0.371 and 0.55, population variances
0.209 and 0.154, and mean runtimes 27.4 and 12.2 minutes (including one timeout
per task). Prepare fresh datasets after this pool change; old runs are untouched.
The full pool includes 189 tasks with zero >=90% successes in five attempts.
Observed partial means at least one native score strictly between zero and one;
we cannot identify unobserved evaluator granularity from five all-zero attempts.
`agent_loop.yaml` registers our environment adapter, not another experiment.
All evaluate their training tasks, **not a holdout**. Rewards stay native partial
scores; the >=0.9 threshold selects tasks, not training rewards.

## Prepare and run

From the repository root on Hertz-2, prepare a dataset (no GPU needed):

```bash
.venv-rl/bin/python -m training.rl.toolathlon_gym.task_profiles \
  --profile smoke --prepare artifacts/training/toolathlon_gym/smoke-data-v2
```

Preparation refuses to overwrite datasets and records task IDs, pool hash, and
Gym revision. Three training rows per task produce three groups of 32 attempts per
epoch. Evaluation contains one row per task. Launcher validates pool hash and
actual parquet task multiplicities. Old pools remain historical provenance only.

Choose a free policy GPU and a fresh run directory:

```bash
RL_DATA="$PWD/artifacts/training/toolathlon_gym/smoke-data-v2" \
RL_ARTIFACTS="$PWD/artifacts/training/toolathlon_gym/smoke-01" \
POLICY_GPU=2 ROLLOUT_GPU=3 bash training/rl/toolathlon_gym/train.sh smoke \
  ray_kwargs.ray_init.num_cpus=64
```

For other profiles replace `smoke` in both preparation and launcher commands.
Hydra overrides follow the config name. Use tmux for unattended training.

## Shared recipe and provenance

Seeds:
- Timur's `tau2-gym/training/configs/grpo/champion.yaml` at
  `08cc4f6ab2ff34abbf50fdc87b62df6313eecb4b`.
  An unmodified copy is preserved in `references/timur-champion.yaml`;
  it is an upstream reference, not a runnable Gym config.
- Agentic-rag's `occ-train/experiments/templates/colocate.sh`, branch
  `ic/feat/train-verl09-agentloop`, at `89d1812bcac1313319f66354fb37efbf7f8326da`.
  Its old `configs/rl.yaml` is not the current recipe.

Shared reference settings: all-linear LoRA rank 32/alpha 64, LR 1.5e-5,
`seq-mean-token-mean` loss averaging. GRPO without standard-deviation normalization
follows Timur; RAG exposes this as an experiment choice. Our async learner takes
one completed 32-attempt group at a time, with two PPO epochs per group.

All profiles use one dataset epoch by default, 96 training attempts/task/epoch,
eight evaluation attempts/task, and separate trainer/rollout GPUs. Shared settings:
4096 prompt + 12288 response/observation budget, eager vLLM, SDPA, and disabled
policy prefix caching. Do not copy RAG's multi-GPU pipeline or adapter-only saves:
we retain optimizer state. The old run used LoRA 16/32 and LR 1e-5; this is a
changed recipe, not an identical rerun.

Qwen non-thinking sampling remains temperature 0.7, top-p 0.8, top-k 20; the adapter
also applies presence penalty 1.5. Subagents remain hosted Qwen3.5-4B.
Episode timeout is 2700 seconds and recursion limit is 410. Infrastructure errors
stay explicit; retain raw evaluations because native check denominators can vary.

Each group contains 32 episodes. Evaluation uses eight attempts/task before
training and every eight updates. Resumable checkpoints (model, optimizer, extra
state) are saved after **every update, before evaluation**. Automatic eviction is
disabled so an unevaluated latest checkpoint cannot delete the best evaluated one.
Best/last are labeled on launcher exit.
Smoke: 192 training + 16 baseline evaluation = 208 scheduled episodes.
Six groups stream to a dedicated trainer while a separate GPU generates.
Each completed group gets two PPO epochs (12 inner optimizer updates total).
With the unchanged evaluation interval of eight updates, one smoke epoch has
no post-training evaluation; use more epochs or an explicit evaluation override.
The watcher/global step counts weight versions, not inner optimizer updates.
Weights synchronize and checkpoints are saved after every group update;
evaluation runs before training and every eight weight versions. This differs from
Timur's batch-eight, one-PPO-epoch recipe; LR and LoRA settings are unchanged.
GRPO still computes advantages from complete 32-attempt groups: a lone fresh
rollout cannot supply that relative baseline.

All profiles use veRL's `experimental/fully_async_policy` with two active groups
(64 active training episodes, not a cap on nested subagent requests),
staleness threshold 1.0, partial-rollout continuation, and rollout log probabilities
as PPO's behavior-policy reference. Trajectories can span policy versions; their
token log probabilities and version range are retained. LoRA is merged only for
weight transfer to the rollout server; training still updates adapters.
The previous synchronous two-task recipe is preserved at git revision `5fb2370`.
On Hertz-2 the direct NCCL transport stalled even in a two-GPU tensor test;
the async launcher defaults to job-local `NCCL_P2P_DISABLE=1` and
`NCCL_IB_DISABLE=1` (shared-memory transport). Override only after checking
`python -m tests.toolathlon_gym.check_async_weights` on the selected GPUs.
`setup.sh` applies `patches/verl-sdpa-padding.patch`: the separated trainer's
batch conversion can use veRL's existing pure-PyTorch padding helpers when
FlashAttention is absent. This does not change the model's SDPA attention.
Full: 33,216 training episodes per epoch. Cold-start: 18,336.
Prepare fresh datasets with three groups/task; historical ten-group datasets
are not rewritten. OPD explicitly prepares one group/task and remains unchanged.
WARNING: evaluating the entire pool every eight groups is very expensive at scale:
full schedules 1,198,544 evaluation episodes; cold-start schedules 365,192.
The settings are intentionally identical, not automatically made cheaper for full.
Choose a shared fixed evaluation panel/cadence before launching these large profiles.
Full is a larger schedule, not a smoke run; choose epochs and evaluation cadence
explicitly before launching if that budget is unsuitable.
The generation batch size is one task group, so no task rows are dropped.

## Infrastructure and outputs

One-time setup: `bash training/rl/toolathlon_gym/setup.sh`, then:

```bash
podman build -f gyms/toolathlon_gym/Dockerfile -t decomposer-toolathlon:latest .
```

RL and trace collection share the Gym Dockerfile; there is no separate RL image
recipe. `RL_GYM_IMAGE` can pin an already-validated image ID for controlled reruns.
The launcher resolves `~/models/decomposer-4b-sft` once and records model, config,
source revision/diff, image, endpoint, dataset and process identity. Override
`MODEL_PATH` for another checkpoint. `router.sh` sources
`~/.local/share/environment/lmrouter.env` (override with `LMROUTER_ENV`).
Credentials travel by environment, not logs. No local subagent service is launched.
Existing episode startup caching and owned-container cleanup stay intact.
Task containers have a 16 GiB RAM limit and databases 2 GiB, with no additional
swap allowance. Python tools run in isolated process groups; timeout/cancellation
kills the group, including descendants. Output returned by the Python tool is
capped at 8,000 bytes per stream. On trainer exit the launcher also cleans up
containers recorded under that run, covering Ray-worker deaths that skip episode
finally blocks. A hard kill of the launcher or host crash still requires recovery
cleanup; no shell EXIT trap can handle SIGKILL.

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

## ClearML

The pinned SDK is included in the RL requirements. Keep credentials in
`~/clearml.conf` (mode 600), never in Hydra arguments or Git. On Hertz-2 the
verified API endpoint is `https://clearml.2a2i.org/api`, not port 8008.
Opt in with `'trainer.logger=[console,tensorboard,clearml]'` at launch.
Local TensorBoard, traces and checkpoints remain the source of truth.
The launcher defaults the experiment name to the run-directory basename so fresh
runs do not accidentally reuse an experiment named simply `overfit` or `full`.
The upstream logger requests continuation of the matching ClearML task; check task
identity when resuming from a different machine. Credentials must be configured
before launch. Enabling logging does not require ClearML Agent or remote scheduling.
