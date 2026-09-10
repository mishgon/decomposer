# Toolathlon-Gym RL

Working branch: `we_rl_toolathlon_gym`. Keep a separate checkout for RL.
This is the integration in progress. A real GRPO update and checkpoint save passed;
**reward improvement and checkpoint resume have not yet been verified**.
Existing collection and benchmark checkouts remain separate.

## First piece: reproducible task selection

Run from the repository root using ordinary Python; no CUDA dependencies:

```bash
python3 -m unittest discover -s tests -p test_toolathlon_gym_rl_data.py
python3 training/toolathlon_gym/prepare_data.py \
  --output artifacts/training/toolathlon_gym/task-split.json
```

The manifest includes every Gym task, its config hash and the Gym source revision.
It refuses to overwrite an existing manifest. `--validation-tasks N` reserves an
explicit, deterministic task-level holdout; the default reserves none. Repeated
rollouts must inherit their task's split. This is a task manifest, not veRL's final
prompt/parquet dataset. It does not assert that every task's infrastructure works.

## Setup and pilot

For the cheap **four-rollout overfitting smoke**, after setup and starting frozen
subagents, use:

```bash
.venv-rl/bin/python -m training.toolathlon_gym.prepare_overfit \
  --task wc-customer-order-gsheet-email \
  --output artifacts/training/toolathlon_gym/overfit-data
bash training/toolathlon_gym/overfit.sh
```

Two pre-update rollouts supply the GRPO group, followed by two post-update
rollouts on the same task. This deliberately uses a diagnostic learning rate of
`1e-4`. There is no separate baseline pass or held-out evaluation. Four rollouts
can verify the update and inspect reward change, not establish generalization
or statistically prove overfitting. The task was chosen using observed runtime
and reward variation; do not treat this checkpoint as unseen-task validation
on a panel containing that task. Use `report --samples 2` for this run.

The larger pilot below is a separate experiment, not a smoke test.

```bash
bash training/toolathlon_gym/setup.sh
podman build -f training/toolathlon_gym/Dockerfile -t decomposer-toolathlon-rl:latest .
.venv-rl/bin/python -m training.toolathlon_gym.check_environment \
  --output artifacts/training/toolathlon_gym/environment-check
.venv-rl/bin/python -m training.toolathlon_gym.prepare_pilot \
  --output artifacts/training/toolathlon_gym/pilot-data
.venv-rl/bin/python -m training.toolathlon_gym.check_evaluators \
  --split artifacts/training/toolathlon_gym/pilot-data/split.json \
  --output artifacts/training/toolathlon_gym/evaluator-check
SUBAGENT_GPU=1 bash training/toolathlon_gym/serve_subagents.sh
# In another shell, after the endpoint on port 8025 is ready:
POLICY_GPU=0 bash training/toolathlon_gym/train.sh
```

The pilot starts from the original `Qwen3.5-4B`, not the previous SFT checkpoint.
Only its rank-16 LoRA parameters train. Subagents use separately served, unchanged
Qwen3.5-4B weights. The default is one policy GPU plus one frozen-subagent GPU.
The 16 training / 8 validation tasks are hash-selected from local-fixture tasks,
without consulting historical rewards. Each validation uses three rollouts per
task; its tasks are never included in gradient updates. This is a small Gym pilot,
not a whole-Gym performance claim.

`evaluation.parquet` additionally includes a fixed four-task training probe,
reported separately from held-out validation. Both panels are measured before
training and every four updates, with three rollouts per task. This avoids
mistaking changes in training-batch difficulty for reward improvement. There are
four environment workers sharing the same two GPUs.

Known native-environment limitations: the email service defaults to
`user@example.com`, although some tasks require another sender. Some native
evaluators skip downstream checks after missing outputs, changing their partial
score denominator. Preserve and inspect raw checks alongside scalar reward;
neither limitation is evidence of a policy improvement.

The small RL image extends an existing `decomposer-toolathlon:latest` Gym image.
It connects the upstream WooCommerce/Notion PostgreSQL adapters and keeps Canvas
account-user queries on its PostgreSQL client; otherwise those requests attempt
HTTP endpoints despite a local database.
The environment check exercises real MCP calls and verifies isolation between two
copies of the same task, including Canvas account-user reads. It allocates no GPU. Build the base Gym image with
`gyms/toolathlon_gym/build.sh` if it is not already available.

Smoke training budgets are 4K initial prompt plus 12K response/observation tokens;
the frozen subagent server supports 256K context. The Decomposer and subagent
recursion limits are 410. The episode wall-clock cap is 30 minutes. Budget-limited
episodes are scored on partial state; unknown evaluator formats and infrastructure
errors fail explicitly instead of receiving fabricated zero rewards.

Generation uses Qwen's [non-thinking general-task recommendations](https://huggingface.co/Qwen/Qwen3.5-4B):
temperature 0.7, top-p 0.8, top-k 20, presence penalty 1.5. Frozen-subagent prefix
caching is enabled; trainable-policy caching is disabled across weight updates.
Raw token IDs/logprobs are retained. Tool and middleware observations have zero
loss mask. On new user feedback Qwen's full template would remove the previous
empty thinking scaffold; the RL sequence retains its actual generation prefix
instead of rewriting past policy tokens. No thinking text is generated in this mode.

`~/watch-rl.sh` on Hertz-2 shows trainer liveness, episode counts and TensorBoard
reward metrics. `--once` prints a single snapshot. The compact dashboard includes
fixed-panel baseline/latest rewards, a batch-reward sparkline, gradient/clipping
warnings and timeouts. “Cooking” means alive, not proven to be learning. Its ETA
covers the whole scheduled run, including validation; until update timings
exist it is explicitly a low-confidence episode-throughput extrapolation.
The same inspector handles smoke tests, subset runs and full training: task counts
and train/evaluation overlap come from the configured datasets, while rollout
counts come from the saved training schedule. It does not infer these from run names.
By default it selects the newest recorded run. Use `--run /path/to/run` to inspect
a specific run, or `--root /path/to/runs` to select another collection of runs.
Missing historical dataset metadata is shown as unknown rather than guessed.
Logs, native evaluations, model outputs and checkpoints
live under `artifacts/training/toolathlon_gym/`.

Compare complete panels and retain per-task rewards:

```bash
.venv-rl/bin/python -m training.toolathlon_gym.report \
  --run artifacts/training/toolathlon_gym/pilot \
  --data artifacts/training/toolathlon_gym/pilot-data
```

This writes `reward_report.json` in the run directory. Incomplete or duplicate
panels have no aggregate reward or delta; optimizer rollouts are not mixed into
the fixed training probe. A positive delta is an observed estimate, not by itself
proof of statistical significance or freedom from native-evaluator artifacts.

## Remaining verification

1. Verify checkpoint resume/export (save and a nonzero-gradient update passed).
2. Measure training-probe and held-out reward before and after training.
3. Audit any improvement against raw evaluations, not only scalar rewards.

No GPU allocation is made by task preparation. The model paths remain under
`~/models` by default; override `MODEL_PATH` and `SUBAGENT_MODEL_PATH` as needed.
Do not duplicate weights here. Put new training artifacts
under `artifacts/training/toolathlon_gym/<run_id>/`.
See [the integration audit](../../docs/toolathlon_gym_verl_plan.md) for the design
and reference revisions.
