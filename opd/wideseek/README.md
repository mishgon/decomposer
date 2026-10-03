# WideSeek OPD

The SFT Decomposer-4B checkpoint runs WideSeek tasks with hosted thinking
Qwen3.5-4B-unlooped subagents. Hosted Qwen3.8 Flash Next scores the student's
exact token IDs. Teacher and subagent clients use the shared model registry.
Only student-generated tokens contribute to the sampled-token k1 distillation
loss. Tool observations and subagent reports have zero loss weight. Native
item-F1 is logged for evaluation; it does not enter the training objective.

`agent_loop.py` injects the veRL policy into `gyms/wideseek/run.py`. The Gym
continues to own task tools, raw logs, judging and worker cleanup. Shared OPD
token and teacher adapters live directly in `opd/`. This stage imports no RL
or Toolathlon code.

## Setup

Prepare [WideSeek's cached corpus and retrieval services](../../gyms/wideseek/README.md)
and [lmrouter access](../../README.md). Reuse a running retrieval service on
port 18080. On the training machine, install the pinned training environment:

```bash
bash opd/setup.sh
source gyms/wideseek/env.sh
```

Start a separate worker process in this checkout so its artifact-root access
matches the trainer. It shares the existing retrieval service and has its own
port and LangGraph storage. It does not load a local subagent model:

```bash
source gyms/wideseek/env.sh
.venv/bin/langgraph dev --config gyms/wideseek/langgraph.json \
  --host 127.0.0.1 --port 18082 --no-browser --no-reload --n-jobs-per-worker 4
```

Use the Gym's `.venv` for this service and `.venv-opd` for training. Start it in
a terminal or tmux session and keep its logs under `artifacts/`.
Export the same `WS_ARTIFACT_ROOT` for workers and training if using another
artifact location. Corpus/index files stay in their existing cache.

## Prepare Tasks and Run

Freeze two tasks into a new dataset directory. Defaults are `width-00001` and
`width-00003`, the two normal finishes in the latest four-task collection smoke.
Training and evaluation use these same tasks; this is an overfit probe.
Use `--tasks ID ...` for another selection. Reference answers remain in the
Gym task file and are excluded from the policy's parquet inputs.

```bash
.venv-opd/bin/python -m opd.wideseek.prepare \
  --source artifacts/gyms/wideseek/data/width/tasks.jsonl \
  --output artifacts/opd/wideseek/data/two-tasks

.venv-opd/bin/python -m opd.wideseek.run --config smoke \
  --data artifacts/opd/wideseek/data/two-tasks \
  --output artifacts/opd/wideseek/runs/smoke-01 \
  --model "$HOME/models/decomposer-4b-sft" --gpus 1 2
```

Choose two free GPUs: one trains the student, the other serves its rollouts.
The teacher and subagents use lmrouter. Before allocating GPUs, the launcher
checks retrieval, the worker graph, a real subagent response and teacher token
alignment. `--preflight-only` performs those checks and exits; use a separate
output directory for a later training launch. The resolved checkpoint path,
configuration, task hashes, source hashes and Git revision are recorded.

`smoke` performs one rollout and one optimizer update, without a baseline or
periodic evaluation. It checks execution rather than learning. `full` uses the
same loss and optimizer for ten epochs, one rollout/task/epoch, concurrency two,
baseline evaluation and evaluation every eight updates with eight attempts/task.
With the two-task dataset this schedules 20 updates. Hydra overrides can follow
the command, for example `trainer.total_epochs=2`.

Both use LoRA rank 32/alpha 64, LR 1.5e-5, two optimizer passes/update,
microbatch and minibatch size one, and a 45-minute episode timeout. The student
training sequence budget is 4096 prompt plus 12288 response/observation tokens,
matching the existing OPD recipe. Hosted subagents keep their provider context
limit. Checkpoints include model, optimizer and extra state after every update.
Resume with the identical command plus `--resume`; changed source, data or
configuration is rejected. No full-split collection is scheduled here.

## Artifacts and Verification

`trainer.log` and TensorBoard scalars record loss, gradients and timing. ClearML
uses project `decomposer-wideseek-opd` and the run directory name. Configure
ClearML on the training host before launch, or explicitly override
`trainer.logger=[console,tensorboard]`. `run.json` records status and exit code.

Each `episodes/<id>/trajectory.json` saves exact policy IDs, generated-token
masks, rollout logprobs and policy versions. Its nested Gym directory contains
raw model/tool/judge logs and archived worker states. `teacher_calls/` preserves
teacher requests and returned token scores without authentication headers.
All failures keep their artifacts. A native judge or cleanup failure remains an
error; it does not silently become a zero training score.

```bash
PYTHONPATH=src:.:external/verl OPD_TOKENIZER="$HOME/models/decomposer-4b-sft" \
  .venv-opd/bin/pytest tests/test_opd*.py tests/test_wideseek*.py tests/test_core.py -q
```

The training-environment tests exercise the real veRL masked loss and token
alignment. A completed optimizer update and checkpoint reload are required
before calling the GPU training path verified. The vendored native scorer is
unchanged from the Gym; its upstream scoring behavior is retained.
