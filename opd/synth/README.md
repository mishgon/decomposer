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

Worker transport errors retry the entire episode in a fresh workspace, at most
twice. Failed attempts retain their traces and traceback but do not enter the
training batch or validation average. Cleanup failures, model outcomes and
exhausted retries are not silently converted into successful episodes.

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

From a checkout on `occ-cds-1`, using its configured `mls` Python environment:

```bash
python -m opd.synth.mlspace.run_train_jobs --dry
python -m opd.synth.mlspace.run_eval --dry
```

Review and commit changes before submission. Real submissions refuse a dirty
checkout and stage the exact commit. Provision that staged checkout with
`bash opd/setup.sh` before submitting without `--dry`. The OPD requirements
lock and separate worker environment take precedence over the general repo
`uv.lock`; do not reuse the shared vLLM 0.24 environment for this recipe.

All paths are under
`/mnt/shared_ru.ml.SZ-5_000264/matrosov/decomposer-synth-artifacts`:
`code/<commit>` holds source and isolated runtimes; `sanity/<experiment>` holds
submission metadata, preflight results, baseline traces and training artifacts.
Completed and active jobs are skipped. Failed output directories are preserved;
use a new committed experiment name for another attempt.

The job reads `~/.config/decomposer/lmrouter.key` and
`~/.config/clearml/matrosov-cds.conf` from the configured alternate home.
Credentials are loaded inside the container, never placed in the job payload.
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
bridge. Workers remain hosted Qwen3.5-4B-unlooped, non-thinking, with native Gym
tools and grading. The validation trajectory has the training runtime's 8192-token
response budget and 300-second episode timeout. Stock GPU inference and token
budgets differ from the hosted selection run; baseline validation measures this.

`gyms/workplace_assistant/validation20.json` freezes the tasks; the adjacent
selection file records provenance and measured teacher 20/20, student 2/20. Those
are selection scores, not independent confirmation or guaranteed future scores.
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
`s3://airi-occ-artifacts/training/artifacts/matrosov-<experiment>`. Other launches
keep their normal checkpoint behavior. Checkpoint payloads temporarily use the
existing shared-NFS run directory, never the model cache under `/home/jovyan`.

After all actor and rollout state is saved, the trainer runs the team's
`s3cmd sync` workflow using the existing `~/.s3cfg`. Every payload file's size
must match the S3 listing before deletion. Upload/verification failures stop
training with local data intact. Successful offloads preserve `evaluation/` and
`OFFLOADED_TO_S3.json`, which records the URI, timestamp and removed-file manifest.
The marker is written before deletion so interrupted cleanup remains explicit.
S3 checkpoints are retained; local heavy files do not accumulate between updates.

The job checks S3 write/read/delete access with a unique tiny probe before
training. No credentials are included in the job description or environment.

Restore a checkpoint before passing it to `--resume-from`:

```bash
UV_BIN=/home/jovyan/.local/bin/uv python -m opd.checkpoint_offload restore /absolute/path/to/global_step_8
```

Restore uses `s3cmd get --recursive --skip-existing`, verifies sizes, and removes
the marker only on success. Partially restored checkpoints remain marked.
This follows team ADR 0008's NFS → S3 → verified offload convention, with
automatic offload at the completed-checkpoint boundary for these MLSpace jobs.

## Verified Run and Review Status

The two-H100 job `lm-mpi-job-b68faccc-e6cb-4fb6-862e-198da877db85` runs the
immutable source snapshot `aa4e0729ce29b05957e44224fbbac8fcc4f1a9e3`. On
2026-11-02, direct checkpoint inspection confirmed a real first update:
248 of 346 LoRA-B tensors were nonzero, all 692 saved model tensors were finite,
and optimizer state recorded two steps, matching the two PPO epochs. The model
payload was 312,239,659 bytes (297.8 MiB), with optimizer state saved separately.
The next update also completed and synchronized to the rollout worker.

The first checkpoint's S3 object sizes matched its saved payload. Local cleanup
left only `OFFLOADED_TO_S3.json` in `global_step_1`; the checkpoint tracker advanced.
This verifies save and offload on an actual training run. End-to-end training
resume from an S3-restored checkpoint remains unverified. Offload verification
checks file sizes, not content hashes.

This run's step-zero rewards were 0.0625 on the synth train probe, 0.28125 on
synth heldout tasks, and 0.50 on Workplace-20. These are the baseline for this
run, not the hosted panel-selection scores. A successful optimizer update does
not establish learning or generalisation. Compare subsequent Workplace-20
validation against this run's own baseline; the small, selected panel and one
sampled attempt per task limit what that comparison can establish.

The final review ran the focused OPD, synth, checkpoint and Workplace suite:
44 tests passed and five environment-dependent checks skipped. Running the
policy suite with the cached Qwen tokenizer then passed all 11 tests, covering
the four tokenizer skips. A separate real Agent Server smoke check passed with
`.venv-workers`: six-copy swap, parallel delegation and clean worker shutdown.
These checks used the provisioned runtime without modifying the training job.

The demo gym and shared in-process workers are also included in
[MR !18](https://gitlab.2a2i.org/optimal-cognitive-core/decomposer/-/merge_requests/18).
Wait for that MR to merge, then rebase this branch onto main and review the
remaining OPD diff before opening its MR. Keep the demo implementation from
!18 when resolving overlap. No running job needs to be restarted for this cleanup.

## Shared Actor Optimization

The actor uses veRL's Torch fused output head, matching the verified WideSeek
configuration with the same Qwen3.5-4B architecture and sampled-token distillation
loss. Activation checkpointing, microbatch size, and the existing linear-attention
fallback remain unchanged. The FLA overlay and disabling activation checkpointing
remain experimental and are not part of this MR.

The recorded isolated-head replay used 25.3 / 31.8 GiB at 8K / 16K instead of
45.2 / 73.1 GiB, with gradient cosine >0.9998 and relative error <1.7%. Replay
times of 9.76 / 20.09 seconds do not establish a speed gain over the baseline;
the demonstrated gain is memory. Existing launched experiments keep their
original staged configurations.

This branch is stacked on the frozen demo MR !18. Its demo gym, standalone
evaluation, shared local workers, and demo tests are unchanged. Optional OPD demo
adapter documentation and tests live under `opd/decomposer_demo/` and
`tests/test_opd_demo_validation.py`. The default learning validation remains the
synthetic probes and the selected Workplace-20 panel.
