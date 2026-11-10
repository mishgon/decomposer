# WideSeek R1 OPD

This port carries the finalized synth OPD pipeline into WideSeek. The shared
policy bridge, hosted teacher, veRL patches and dependency lock come from
`syntch-opd`, including native FlashAttention2, pinned Workplace validation data,
and policy-timeout recovery. The launcher also
requires live retrieval, worker and teacher preflight before training.

## Recipe

- Decomposer Manager Qwen3.5-4B SFT v1.2.0 non-thinking coordinator; hosted Qwen3.8 Flash Next teacher.
- One training GPU and one rollout GPU; controller CPU reservations of one each.
- LoRA rank 32, alpha 64, all linear layers; learning rate 1.5e-5.
- Full deduplicated WideSeek source pool, excluding the frozen holdout, shuffled
  with seed 42 and one row per training question. Tasks rotate without replacement
  until the pool wraps. Cap: 512 rollouts / 64 batches of eight
  episodes, two PPO epochs per batch.
- Eight concurrent rollouts, two agent-loop workers, staleness threshold one.
- A 32,768-token total sequence budget (8,192 prompt + 24,576 response tokens) and a 900-second (15-minute) episode budget.
- Validation before training, every eight updates and at the end: four train
  probes, eight question-disjoint WideSeek tasks, and the unchanged Workplace-20
  OOD panel, each with one sampled attempt.
- LoRA-only checkpoints after every update, with optimizer and resume state.
  Hertz-2 keeps them on the shared disk; only MLSpace enables S3 offloading.

The shared training engine stays unchanged. WideSeek uses the current gym's hosted thinking
Qwen3.5-4B researchers, fixed Wiki-2018 search/access tools and native scorer.
Workplace continues using its fixed non-thinking workers. Neither task reward
nor Workplace reward enters the distillation objective.

Preparation accepts one source split per run. It uses `validation_panel.json`
for width and `validation_panel_depth.json` for depth, checking both task IDs and
question hashes. Each panel freezes four train probes and eight held-out questions.
The depth panel was selected from the unique source questions with seed 42, without
filtering by observed scores. All other
unique questions enter the training pool. References stay in the task store for
judging; they never enter the coordinator prompt. Select depth with `--source artifacts/gyms/wideseek/data/depth/tasks.jsonl`.
Workplace preparation resolves the same 20 IDs from the pinned Hugging Face
validation dataset and embeds complete grading tasks in the evaluation Parquet.
The small validation panels are diagnostic, not representative benchmarks.

## SFT Initialization

`prepare_model.py` reads `~/.config/huggingface/token` on Hertz-2 without printing
it. Keep that file mode 600. It pins the release to
`70f7732e58f25be7787d7ff75761ef85fe90c98e` and runs the release's GPU merge script
in a separate uv environment. The script must reproduce the published merged
weight SHA-256 before the model can be used. The pinned OPD environment is unchanged.

```bash
python -m opd.wideseek.prepare_model --gpu 0 \
  --output /mnt/share14T-2/matrosov/wideseek/models/decomposer-manager-sft-v1.2.0
```

OPD starts a fresh rank-32 adapter on the verified merged SFT model. It does not
resume an old optimizer or attach the SFT adapter to the stock rollout model.

## Hertz-2 Storage

Use `/mnt/share14T-2/matrosov/wideseek`, not the system-disk home. Existing assets:

- `assets/`: Wiki-2018 corpus, Qdrant index and E5 encoder.
- `artifacts/gyms/wideseek/data/width/tasks.jsonl`: normalized width tasks.
- `artifacts/gyms/wideseek/data/{depth,hybrid}/`: other preserved sources.

Keep new code, virtual environments, model downloads, caches and run output on
this mount. Existing collections and checkpoints are historical evidence; do
not overwrite them. Hertz-2 currently has H200 NVL GPUs. Choose two idle GPUs
under the host's allocation rules, checking again immediately before launch.

Provision the pinned OPD environment with `bash opd/setup.sh`.
OPD Python must include development headers (`Python.h`) for Triton's CUDA
helper compilation. On bare metal, create `.venv-opd` with a uv-managed Python
3.12 rather than a system Python without development packages. The launcher
checks for this header before starting services.

The separate CPU retrieval environment comes from `bash gyms/wideseek/setup.sh`; it needs
pinned `external/RLinf`. Set `HF_HOME`, `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`,
`XDG_CACHE_HOME` and `TMPDIR` to directories on the large disk before setup.
Keep `RAY_TMPDIR` short, for example `/mnt/share14T-2/matrosov/ray`: Ray appends
session and socket names, and Linux limits Unix socket paths to 107 bytes.
Supply lmrouter and ClearML credentials through the existing secret mechanism.
If using the DNS workaround, prepend `opd/synth/mlspace/dns` to `PYTHONPATH`
and bypass proxies for lmrouter/ClearML. The bare-metal launcher also prepends
`opd/wideseek/dns` for child processes, selecting standard asyncio so Agent
Server workers honor the socket-based DNS override. No credentials belong in Git.

After review, tests, commit and service checks, the direct launch is:

```bash
PYTHONPATH=src:.:external/Gym:external/verl python -m opd.wideseek.local \
  --root /mnt/share14T-2/matrosov/wideseek/runs/NEW_EXPERIMENT \
  --assets /mnt/share14T-2/matrosov/wideseek/assets \
  --source /mnt/share14T-2/matrosov/wideseek/artifacts/gyms/wideseek/data/width/tasks.jsonl \
  --model /mnt/share14T-2/matrosov/wideseek/models/decomposer-manager-sft-v1.2.0 --gpus 0 1
```

The launcher requires a clean committed checkout, checks the selected GPUs and
local ports, starts Qdrant and CPU retrieval, runs a fresh 32-episode starting-checkpoint
baseline, then starts OPD. It owns only its subprocesses. The baseline uses eight
held-out WideSeek questions with four attempts each. Training checks the source
hash and starting-checkpoint path against the baseline. To retry training after a successful baseline, pass
`--baseline /absolute/path/to/previous/run/baseline`; the new run links to it
without modifying its results. Existing synth baselines cannot be reused here.

The dashboard reads the same metrics and checkpoints:

```bash
python -m opd.wideseek.watch --root /mnt/share14T-2/matrosov/wideseek/runs \
  --run NEW_EXPERIMENT
```

## Restored Actor Kernel Benchmark

The default enables veRL's Torch fused output head while retaining the existing
linear-attention fallback and activation checkpointing. In the isolated-head
check it used 25.3 / 31.8 GiB at 8K / 16K instead of 45.2 / 73.1 GiB, with
gradient cosine >0.9998 and relative error <1.7%. Its 9.76 / 20.09 second replay
times do not establish a speed gain over the baseline; the demonstrated gain
is memory. FLA and disabling activation checkpointing remain explicitly gated.

A two-example microbatch replay with the fused head took 12.02 / 29.55 seconds
at 8K / 16K, with peaks of 31.7 / 44.7 GiB. Per example, this is about
1.58 / 1.34 times faster than the original fallback. Gradient cosine exceeded
0.9998 and relative error stayed below 1.6%. Both examples duplicate the same
trajectory, so this does not verify loss weighting across different episodes.
The default microbatch remains one pending that check; the optimizer batch remains
eight. FLA stays opt-in until its full-model gradient discrepancy is explained.

The restored kernel bundle is isolated from the pinned OPD virtual environment:
Torch 2.10.0/CUDA 12.8, Transformers 5.5.4, Triton 3.7.1, FLA 0.5.2 and
causal-conv1d 1.7.0. `fast_runtime.py` records and checks bundle file hashes and
checks actual imports before a trainer starts. It selects the Triton FLA path
with `FLA_TILELANG=0`. The default virtual environment is not upgraded.

The November 3 replay used one H200, the current FP32-storage/BF16-compute FSDP
actor with rank-32 LoRA, fixed saved WideSeek tokens, and fixed signed loss weights.
Each measurement excluded one warmup pass and took the median of two passes.
The 16K capacity case repeated the saved response to reach the configured maximum.
These are forward/backward measurements, not complete optimizer steps or end-to-end
OPD throughput; they exclude rollout, teacher calls, synchronization and saving.

| Variant | 8K seconds | 8K peak GiB | 16K seconds | 16K peak GiB |
|---|---:|---:|---:|---:|
| Existing fallback | 9.47 | 45.2 | 19.84 | 73.1 |
| FLA + causal convolution | 1.42 | 45.2 | 2.06 | 73.1 |
| Above + fused output head | 1.45 | 24.2 | 2.23 | 25.7 |
| Above, no activation checkpointing | 1.18 | 70.1 | 1.72 | 117.5 |

The fused-head and activation-checkpointing changes agree with the same-FLA
comparison at cosine >0.9998 and relative gradient error <1.8%. The FLA operator
agrees with FP32 recurrence math at cosine >0.9999 and relative gradient error
<0.7% on the tested tensors. However, the full-model signed-loss replay differs
from the old fallback by 22–36% in relative gradient error (cosine 0.976 / 0.938).
An explicit-FP32 recurrence diagnostic did not resolve that difference. The
full-model equivalence check failed; the operator checks do not supersede it.

Evidence remains on Hertz-2 under `wideseek/benchmarks/`:
`actor-fast-0b1c1c6-20261103`, `actor-reference-e6b159d-20261103`, and
`operator-numerics-20261103.log`. Raw gradients and log probabilities are retained.
Head-only and two-example results are in `actor-head-5bd203a-20261103` and
`actor-micro2-4bd87c9-20261103` respectively.
Run `~/watch-bench.sh` there for the benchmark dashboard. `watch-opd.sh` is unchanged.

Restore the preserved artifacts into a new bundle, then select the candidate
recipe with `OPD_FAST_RUNTIME=/absolute/path/to/bundle`:

```bash
python -m opd.wideseek.fast_runtime \
  --restore-from /mnt/share14T-2/matrosov/synth/kernel-ablation \
  --output /large/disk/new-kernel-bundle
```

The candidate is `fast.yaml`; it preserves the learning rate, batch sizes, loss,
sampling, task budgets and validation. Its 117.5 GiB measured peak without
activation checkpointing requires H200-class memory; it is not an H100 recipe.

## Adapter Reload Benchmark (Gated)

The November 3 receiver-only check used the WideSeek step-64 adapter on one H200
with the pinned vLLM worker extension. Full BF16 payload loading took 0.422 s;
FP32 adapter loading took 0.107 s (medians of three warm measurements). Payloads
were 9,319,730,176 and 311,951,360 bytes respectively. These timings include
host-to-GPU copies and vLLM reload, but exclude actor export, merging, NCCL and
cache reset. They are not end-to-end synchronization timings.

The output check did not pass: adapter versus merged mean absolute prompt-token
log-probability difference was 0.500 (maximum 5.090), while the merged reference
matched base exactly. Zero adapter matched base exactly; reloading the same
adapter differed by 0.030 on average. The reference construction or reload path
needs investigation before interpreting these differences as model numerics.
No production synchronization setting was changed.

The current async sender also calls `get_per_tensor_param()` without
`base_sync_done=True` and discards the returned PEFT metadata. The receiver does
not supply that metadata to vLLM. Changing `lora.merge` alone is insufficient.
Evidence: Hertz-2 `wideseek/benchmarks/sync-receiver-730b761-20261103/result.json`;
benchmark code: `benchmark_sync.py` at commit `730b761`.

Both `opd.wideseek.run` and `opd.wideseek.local` accept `--gpus` with any
single-node GPU list and `--training-gpus` (default 1). For example,
`--gpus 4 5 6 7 --training-gpus 2` assigns two GPUs to training and two to rollout.
A single GPU alternates the two phases; that WideSeek layout has not been GPU-tested.
Batch sizes and tensor parallelism must fit the selected split. MLSpace accepts
`--num-gpus` and `--training-gpus`, with `--instance-type` for allocations outside
the default mapping.

## MLSpace Wrapper

MLSpace remains optional. Both launchers use the same allocation, model,
absolute NFS roots, author tags, DNS workaround, credentials and S3 convention
as the final synth launch. The experiment name distinguishes its artifacts.
The retrieval URL must be reachable from inside the job; submission-host
localhost is not shared with a GPU container.

```bash
python -m opd.wideseek.mlspace.run_train_jobs --dry \
  --author-name YOUR_NAME --author-telegram YOUR_TELEGRAM \
  --model /shared/models/Qwen3.5-4B \
  --clearml-config ~/.config/clearml/clearml.conf \
  --source /shared/data/wideseek/tasks.jsonl \
  --search-url http://RETRIEVAL_HOST:18080
```

Use the same arguments with `opd.wideseek.mlspace.run_eval` for a baseline-only
job. Paths, identity and storage options match the [synth launcher](../synth/README.md).
Data and model paths must be visible inside the job. Dry runs submit nothing.
No WideSeek MLSpace job has been submitted.
