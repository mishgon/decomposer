# Synthetic OPD Experiment

This experiment runs sampled-token, fully asynchronous veRL OPD on a deterministic
file-swap environment. It uses the shared policy bridge, hosted teacher client,
pinned dependency lock and veRL patches under `opd/`.
Task rewards are measured separately and do not enter the distillation loss.

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

The launcher requires a completed stock-model baseline that records the same
resolved checkpoint path. Keep checkpoint directories immutable. It checks the installed
veRL patches and teacher prompt-logprob alignment before launching training.
Hosted chat completion alone is insufficient: OPD needs compatible token IDs,
decoded token meaning and prompt logprobs. A failed preflight aborts training.
The hosted teacher uses the model registry's authenticated connection. No
OpenRouter fallback or automatic paid-provider substitution is implemented.

For more GPUs, use `--gpus 0 1 2 3 --training-gpus 2` (two training and two rollout
GPUs). These launchers target a single node.

For one GPU, replace `--gpus 0 1` with `--gpu 0`. This selects
`single_gpu.yaml`: veRL colocates rollout on the training GPU and alternates
generation with optimizer updates. The teacher remains hosted. This mode uses
the same OPD recipe but does not overlap GPU generation with training. The final
one-GPU scheduling correction has not been verified through a successful update;
the verified MLSpace configuration uses two H100s.

Start with a two-update smoke test by adding `rollout.total_rollout_steps=16`.
For a fresh 128-update run, use `trainer.total_epochs=8` and
`rollout.total_rollout_steps=1024`. This trainer derives the update count from
the rollout budget and batch size.

Setup creates a separate CPU-only `.venv-workers` for Agent Server because its
web dependencies conflict with the pinned vLLM training environment. The common
server launcher accepts an optional Python executable; the core orchestration
loop is unchanged. `--worker-python` can select an existing compatible environment.

## Fixed Recipe

| Setting | Value |
|---|---|
| Student | Stock Qwen3.5-4B, non-thinking (no Decomposer SFT) |
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
metrics. To resume, pass `--resume-from /previous/run/checkpoints/global_step_15`
and a new `--output` directory. veRL restores model, optimizer and dataloader
state; previous artifacts remain untouched. Keep the model and training recipe
unchanged when resuming.

Known transient worker transport errors retry the episode in a fresh workspace
with capped exponential backoff and jitter until recovery or cancellation. Failed
attempts retain their traces and traceback but do not enter the training batch or
validation average. Expected task failures retain their scores; unsafe cleanup and
unclassified infrastructure errors stop the run. Hosted model requests use the
shared lmrouter outage recovery.

## MLSpace Run

The committed experiment in `mlspace/experiments.py` runs stock Qwen3.5-4B on
two SR008 H100s, one for training and one for rollout. It verifies teacher prompt
logprobs, ClearML authentication and S3 access, reuses a completed stock baseline
or runs the 32-episode baseline, then starts 512 training rollouts (64 updates).
The baseline can also be submitted independently with `run_eval`; that launcher
uses the experiment's GPU allocation even though baseline inference uses GPU 0.
The selected instance bundles 24 CPU cores and 488 GB host RAM. These are
allocation limits, not measured training requirements. Each Ray controller
reserves one CPU, leaving capacity for GPU and agent workers.

From a checkout on shared storage, using its configured `mls` Python environment:

```bash
python -m opd.synth.mlspace.run_train_jobs --dry \
  --author-name YOUR_NAME --author-telegram YOUR_TELEGRAM \
  --model /shared/models/Qwen3.5-4B \
  --clearml-config ~/.config/clearml/clearml.conf
```

Use the same arguments with `opd.synth.mlspace.run_eval` for a baseline-only job.
Identity arguments are required. Only credential file paths enter the payload;
the job reads their contents inside the container. `--lmrouter-key-file` defaults
to `~/.config/decomposer/lmrouter.key`, and `--s3-config` to `~/.s3cfg`.
Override these paths when the container and submission host have different homes.
All supplied paths must be accessible inside the job.

Artifacts default to `decomposer-artifacts` beside the checkout, resolved to an
absolute path. Use `--artifacts-root` to select another shared-NFS directory outside
Git. `code/<commit>` holds staged source and runtimes; `sanity/<experiment>` holds
submission metadata, baseline traces and training artifacts; `cache/huggingface`
holds the HF cache. The launcher does not override `HOME`.

Set `num_gpus` and `training_gpus` in the experiment registry, or override them
with `--num-gpus` and `--training-gpus`. For allocations beyond the existing
one- and two-GPU mappings, pass `--instance-type` from `mls job instance_types`.
The instance must provide the requested GPU count. One GPU alternates training
and rollout; otherwise training GPUs come first and rollout uses the rest.
Batch sizes and tensor parallelism must remain compatible with the chosen split.

Review, commit and push changes before submission. Real submissions refuse a dirty
checkout and stage the exact commit. Provision that staged checkout with
`bash opd/setup.sh` before submitting without `--dry`. The OPD requirements
lock and separate worker environment take precedence over the general repo
`uv.lock`; do not reuse the shared vLLM 0.24 environment for this recipe.
Completed and active jobs are skipped. Failed output directories are preserved;
use a new committed experiment name for another attempt.

The temporary B200 hostname workaround maps lmrouter and ClearML names to
`176.108.242.226` through Python's `socket.getaddrinfo`. A job-local
`sitecustomize.py` on `PYTHONPATH` applies it before SDK imports, including child
Python processes. URLs retain their hostnames for SNI and certificate verification;
proxies are bypassed. No system hosts file or elevated privileges are needed.

### Live Terminal Watcher

`opd/synth/watch-opd.sh` restores the Hertz-2 OPD watcher's compact layout and
last-eight-point sparklines (original implementation at commit `7f02353`). It
reads MLSpace job status, baseline manifests, saved checkpoints and TensorBoard
scalars from the shared artifacts directory. It never changes jobs or training
artifacts and does not read credential files.

On `occ-cds-1`, run `./watch-opd.sh` from the alternate home. Use `--once` for a
single snapshot, `--interval 30` for slower refreshes, or
`--run stock-qwen4b-single-h100-smoke-20261030` to select an experiment.
Ctrl-C stops only the watcher. The default refresh interval is 15 seconds plus
the time spent querying MLSpace; its status query has a 15-second timeout.

Baseline ETA uses observed wall throughput after three completed episodes.
Update ETA needs two logged update timestamps and covers remaining updates;
final validation and teardown are excluded. Queue ETA and whole-run ETA remain
unknown where there is no measured basis. Failed or unavailable job status
suppresses countdowns even if an old manifest still says running. Sparkline
scales are independent, and reward means native partial score, not pass rate.

The watcher runs from a separate committed source snapshot with
`OPD_WATCH_PYTHON` pointing to a provisioned OPD environment (TensorBoard and
PyYAML). Installing/updating it does not change the source of a running job.

## Workplace-20 Run

The `stock-qwen4b-synth-workplace20-twoh100-cpus-20261102` experiment starts from
stock Qwen3.5-4B on two H100s: 512 synth rollouts, approximately 64 eight-episode
updates, LoRA rank 32, learning rate 1.5e-5. Checkpoints are saved each update;
completed MLSpace checkpoints are uploaded to S3, size-verified, then offloaded
from NFS. Model checkpoints contain
only LoRA adapter weights; optimizer, scheduler, RNG and dataloader state remain
available for resuming against the same base model.
Baseline, every eighth update, and final validation use pass@1: 12 synth probes
and 20 frozen Workplace tasks, reported under separate data sources.
Workplace rewards never enter the training dataset or distillation objective.

The Workplace coordinator uses current checkpoint weights through the veRL token
bridge. Workers use hosted Qwen3.5-4B, non-thinking, with native Gym
tools and grading. The validation trajectory has the training runtime's 8192-token
response budget and 300-second episode timeout. Stock GPU inference and token
budgets differ from the hosted selection run; baseline validation measures this.

`gyms/workplace_assistant/validation20.json` contains only the 20 task IDs.
Preparation resolves them from the validation split of
`nvidia/Nemotron-RL-agent-workplace_assistant`, pinned to revision
`f674d9bba17820d90f14eefe9156fdc018a0db60`, and embeds the prompts and grading
references in `evaluation.parquet`. Preparation needs Hugging Face access or a
cached dataset; validation workers use the prepared rows without downloading data.
Compare rewards against each run's step-zero validation.
The panel contains native multi-record CRM/project-management tasks. The teacher
used multiple worker runs on these tasks, while the student usually used one.
They test target coverage and coordination, but do not reproduce synth's swap
dependency. The demo gym is excluded from validation and remains unchanged.

The staged checkout must include pinned `external/Gym` source and CSV fixtures.
The runtime uses pandas 2.2.3, matching calibration. Credentials and DNS handling
follow the MLSpace setup above.

Ray uses standard asyncio (`RAY_USE_UVLOOP=0`): uvloop bypasses the temporary
`socket.getaddrinfo` hostname mapping and failed during Workplace validation.
Prior failed runs and their baseline artifacts remain available.

## MLSpace Checkpoint Offload

MLSpace jobs set `OPD_CHECKPOINT_S3_URI` to
`s3://airi-occ-artifacts/training/artifacts/decomposer-<experiment>`. Other launches
keep their normal checkpoint behavior. Checkpoint payloads temporarily use the
shared-NFS run directory. Override the S3 parent with `--checkpoint-s3-root`.

After all actor and rollout state is saved locally, a background thread runs the
team's `s3cmd sync` workflow using the existing `~/.s3cfg`. Training continues
during hashing, upload, verification and cleanup. Only one checkpoint can await
offload: before writing another checkpoint, the trainer waits for the previous
upload to finish. Local saving and any such wait still block training.

Every payload file's size must match the S3 listing before deletion.
Upload/verification failures surface at the next training-step boundary, save,
or shutdown, with local data intact. The trainer drains pending uploads on normal
exit and Python exceptions; a forcibly killed process cannot finish an upload.
Successful offloads preserve `evaluation/` and
`OFFLOADED_TO_S3.json`, which records the URI, timestamp and removed-file manifest.
The marker is written before deletion so interrupted cleanup remains explicit.
S3 checkpoints are retained; at most one completed checkpoint awaits offload locally.

The job rejects an occupied experiment prefix and checks S3 write/read/delete
access with a unique tiny probe before training. No credentials are included in the job description or environment.

Restore a checkpoint before passing it to `--resume-from`:

```bash
UV_BIN="$(command -v uv)" python -m opd.checkpoint_offload restore /absolute/path/to/global_step_8
```

Restore fetches missing or mismatched payloads individually into temporary files,
verifies sizes and SHA-256 hashes when present, and replaces files atomically.
It removes the marker only on success. Partially restored checkpoints remain marked.
This follows team ADR 0008's NFS → S3 → verified offload convention, with
automatic offload at the completed-checkpoint boundary for these MLSpace jobs.
Full training resume from an S3-restored checkpoint still requires an end-to-end
GPU check; the restore utility is covered by file-integrity tests.

## Shared Actor Optimization

The actor uses veRL's Torch fused output head, matching the verified WideSeek
configuration with the same Qwen3.5-4B architecture and sampled-token distillation
loss. Activation checkpointing, microbatch size, and the existing linear-attention
fallback remain unchanged. The FLA overlay and disabling activation checkpointing
remain experimental and are not part of this MR.

The default validation uses synthetic probes and the selected Workplace-20 panel.
Optional demo validation is documented under `opd/decomposer_demo/`. The demo gym
itself is already on main and is unchanged by this feature.

## Failure Handling and Checkpoint Integrity

Workplace validation and optional OPD demo validation use the shared episode-failure
classifier. Worker recursion, time, and model-call budgets are zero-reward task
outcomes. Known transient worker failures trigger the retry guard. Permanent
infrastructure errors and invalid token alignment remain fatal, even when another
worker or the coordinator also exhausted a budget. Raw worker errors
remain in the saved result. The standalone demo gym and its scoring are unchanged.

Before starting training, probe the configured Workplace worker profile:
`lmrouter/qwen_3_5_4b_non_thinking`. The MLSpace entrypoint does
this before baseline inference; direct synth launches do it before creating the
training output. An unavailable worker is an explicit preflight failure. Compare validation against
the new run’s step-zero scores, using the same worker throughout the run.

Checkpoint offload refuses any occupied destination unless the local offload marker
already identifies the completed upload. This prevents same-sized old objects from
being mistaken for the new checkpoint. Use a fresh experiment destination after an
interrupted upload without a marker; the local payload is preserved. Do not run two
writers against the same checkpoint prefix.

New markers record local SHA-256 hashes. Restore downloads missing, truncated, or
hash-mismatched payloads into temporary files, verifies them, and atomically replaces
the destination. It keeps the marker until every payload passes verification and
leaves local evaluation artifacts alone. Older size-only markers remain readable,
but cannot detect same-sized content corruption. Upload verification still relies
on s3cmd transfer integrity and remote sizes; it does not perform a full readback.
