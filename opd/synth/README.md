# Synthetic OPD Experiment

This branch tests the existing sampled-token, fully asynchronous veRL OPD
pipeline on a simpler environment. It imports the shared policy bridge, hosted
teacher client, pinned dependency lock and veRL patches from `we_opd_wideseek_r1`.
The teacher registry key is updated to main's `lmrouter/` naming.
There is no separate RL implementation and no task reward mixed into OPD loss.

## Order

1. Validate the deterministic reference plan and worker isolation with tests.
2. Run [stock Qwen3.5-4B evaluation](../../evals/synth/README.md).
3. Inspect its success rate; then launch OPD from the **same stock checkpoint**.

```bash
bash opd/setup.sh
PYTHONPATH=src:. .venv-opd/bin/python -m opd.synth.run \
  --baseline artifacts/evals/synth/stock-qwen4b \
  --model "$HOME/models/Qwen3.5-4B" \
  --output artifacts/opd/synth/stock-qwen4b \
  --gpus 0 1
```

The launcher requires a completed stock-model baseline. It checks the installed
veRL patches and teacher prompt-logprob alignment before launching training.
Hosted chat completion alone is insufficient: OPD needs compatible token IDs,
decoded token meaning and prompt logprobs. A failed preflight aborts training.
The hosted teacher uses the model registry's authenticated connection. No
OpenRouter fallback or automatic paid-provider substitution is implemented.

Setup creates a separate CPU-only `.venv-workers` for Agent Server because its
web dependencies conflict with the pinned vLLM training environment. The common
server launcher accepts an optional Python executable; the core orchestration
loop is unchanged. `--worker-python` can select an existing compatible environment.

## Fixed Recipe

| Setting | Value |
|---|---|
| Student | Stock Qwen3.5-4B, non-thinking |
| Teacher | Hosted Qwen3.8 Flash Next, non-thinking |
| Workers | Deterministic COPY/CHECK executors |
| Train / held-out templates | 16 / 8, disjoint filenames |
| Training sampling | 8 attempts/task/epoch, 4 epochs |
| Scheduled rollouts / update batches | 512 / approximately 64 |
| Minibatch / PPO epochs | 8 episodes / 2 |
| LoRA | Rank 32, alpha 64, all linear layers |
| Learning rate | 1.5e-5 |
| Rollout concurrency | 8 |
| Validation | 4 train-probe + 8 held-out tasks, 4 attempts each |
| Validation interval | Baseline, every 8 updates, final |
| Checkpoints | After every update, including optimizer state |
| GPUs | 1 training + 1 rollout, hosted teacher |
| Logging | Console, TensorBoard, ClearML |

The rollout response budget is 8192 tokens across the trajectory, including
observations (masked out of the training loss). There is no 8K per-worker model
cap because workers do not sample tokens. Scores measure correct file states;
parallelism and copy count are separate diagnostics. Compare fixed-panel native
scores and pass rates alongside loss. The training tasks are not held-out evidence.

Artifacts include `trainer.log`, `run.json`, prepared data, per-episode traces and
token masks, teacher requests/logprobs, checkpoints, TensorBoard and ClearML
metrics. This first launcher starts fresh runs only; checkpoints retain the
state needed to add explicit resume handling without overwriting prior results.
