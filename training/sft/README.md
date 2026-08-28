# Decomposer supervised fine-tuning

This workflow converts valid Decomposer rollouts into tool-calling
conversations and performs full-parameter SFT with TRL. The loss covers only
Decomposer outputs. Benchmark prompts, tool definitions, tool responses, and
subagent reports remain visible as context but receive label `-100`.

## Install

From the repository root:

```bash
uv sync --group train
```

The root and `external/Gym` environments remain separate.

## Prepare the dataset

Dataset releases are defined by strict, checked-in build specifications. Build
the original all-subagent Workplace source pair with:

```bash
uv run --group train python -m gyms.workplace_assistant.prepare sft \
  --dataset workplace-all-v3 \
  --output-root /mnt/shared_ru.ml.SZ-5_000264/sukhorukov/decomposer_artifacts/datasets/sft
```

Build the 26B-A4B non-thinking source pair with:

```bash
uv run --group train python -m gyms.workplace_assistant.prepare sft \
  --dataset workplace-26b-nonthinking-v3 \
  --output-root /mnt/shared_ru.ml.SZ-5_000264/sukhorukov/decomposer_artifacts/datasets/sft
```

Both specifications use exact reward `1.0`, prompt-fixed validation fraction
`0.1`, and seed `42`. All teacher variants of one prompt are assigned to the
same split. The builder requires a clean Git worktree and refuses to replace an
existing `<dataset-id>/<version>` directory.

### Qwen Workplace + Toolathlon final all-reward release

The final mixed Qwen release intentionally keeps both full- and non-full-reward
traces. Its terminal Toolathlon run contains 404 completed episodes and 99
failed episodes with no trace/result pair. Import only that run into a
hash-namespaced immutable location:

```bash
.venv/bin/python -m data.sft.import_toolathlon \
  --archive /mnt/shared_ru.ml.SZ-5_000264/sukhorukov/traces_full.tar.gz \
  --archive-prefix matrosov/decomposer-qwen/artifacts/gyms/toolathlon_gym \
  --run-id 20260826T122838Z-84ae95f3 \
  --expected-sha256 493c24c4f8853230c0b7557b905ce2027138522ee6da06d5d2e71d0cade94753 \
  --output-root /mnt/shared_ru.ml.SZ-5_000264/sukhorukov/decomposer_artifacts/evaluation/data/toolathlon_gym/imports/snapshots/493c24c4
```

Build the immutable mixed release after committing the preparation code and
specification:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m data.sft.prepare \
  --spec data/sft/specs/decomposer_mixed_deepseek_qwen35_4b_nonthinking_v1_32k.yaml \
  --output-root /mnt/shared_ru.ml.SZ-5_000264/sukhorukov/decomposer_artifacts/datasets/sft \
  --source toolathlon-deepseek-v4-flash-0731-qwen35-4b-nonthinking-n1=/mnt/shared_ru.ml.SZ-5_000264/sukhorukov/decomposer_artifacts/evaluation/data/toolathlon_gym/imports/snapshots/493c24c4/20260826T122838Z-84ae95f3
```

Preparation starts from 1,659 candidates: 404 completed Toolathlon episodes
plus one of three Workplace rollouts for each of 1,255 tasks, selected by a
stable seed-42 hash before validation. Structural validation retains 375
Toolathlon traces and 1,243 Workplace traces. The 32K token limit excludes 17
Toolathlon traces, producing 1,601 records: 1,441 train and 160 validation.
Toolathlon contributes 358 records (60 reward `1`, 298 reward `0`), while
Workplace contributes 1,243 (953 reward `1`, 290 reward `0`). Reward value
alone never excludes a trace, and no trace is truncated. Toolathlon adapter v3
accepts terminal `completed_with_errors` manifests, selects only completed
episodes, and records the 99 failed episodes separately. Because this archive
uses the legacy trace format, the canonical policy interface supplies the tool
schema.

Submit the final run from the base Qwen3.5-4B checkpoint on four GPUs with
high priority:

```bash
/mnt/shared_ru.ml.SZ-5_000264/sukhorukov/.venv-mls/bin/python \
  -m training.sft.run_train_jobs \
  --filter qwen35-4b-nonthinking-mixed-v1-final-493c24c4-404-32k-full-4gpu \
  --priority high
```

The run evaluates once per epoch and uses early-stopping patience one.

#### Filtered Workplace reward-1 and Toolathlon pass-or-quality variant

The filtered comparison reuses exactly the same pinned sources and the same
seed-42 one-of-three Workplace sampling as the all-rewards release. Workplace
then keeps only reward-1 traces. Toolathlon always keeps a binary native pass;
for binary failures it keeps traces whose available check ratio is strictly
greater than `0.9`, or traces whose evaluator does not expose check counts.
Check ratios support both native result schemas: `total_passed / total_checks`
and `passed / (passed + failed)`.

Build the separate immutable release from a clean committed checkout:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m data.sft.prepare \
  --spec data/sft/specs/decomposer_mixed_deepseek_qwen35_4b_nonthinking_v1_filtered_pass_quality_32k.yaml \
  --output-root /mnt/shared_ru.ml.SZ-5_000264/sukhorukov/decomposer_artifacts/datasets/sft \
  --source toolathlon-deepseek-v4-flash-0731-qwen35-4b-nonthinking-n1=/mnt/shared_ru.ml.SZ-5_000264/sukhorukov/decomposer_artifacts/evaluation/data/toolathlon_gym/imports/snapshots/493c24c4/20260826T122838Z-84ae95f3
```

The expected 32K release has 1,239 records: 953 Workplace traces and 286
Toolathlon traces, split into 1,116 train and 123 validation records. The
deterministic prompt-fixed split is recomputed after source filtering. Before
structural and length filtering, the Toolathlon policy keeps 314 of 404
completed traces: 68 binary passes, 21 binary failures above the quality
threshold, and 225 binary failures without check counts. It excludes 90
binary failures at or below the threshold. Structural validation excludes 16
of the selected traces and the 32K limit excludes another 12.

Submit a separate run from the base Qwen3.5-4B checkpoint on four GPUs with
high priority:

```bash
/mnt/shared_ru.ml.SZ-5_000264/sukhorukov/.venv-mls/bin/python \
  -m training.sft.run_train_jobs \
  --filter qwen35-4b-nonthinking-mixed-v1-final-493c24c4-404-filtered-pass-qgt90-32k-full-4gpu \
  --priority high
```

This run uses the same optimization settings as the all-rewards comparison:
32K inputs, four GPUs, five epochs at most, per-epoch evaluation, and
early-stopping patience one. It has an independent output directory and does
not resume or overwrite the all-rewards run.

#### Pinned partial Toolathlon snapshot

The snapshot-specific release
`v1-partial-3983f605-327-32k` is an immutable experiment input built from
Toolathlon archive SHA-256
`3983f60540f1887befbc2654db8a8d7b169c397a88d62d1017625cd480386f5f`.
It uses the 327 trace/result pairs available for run
`20260826T122838Z-84ae95f3`; it does not claim that the 503-task source run is
complete. Import it into a hash-namespaced location so that a later completed
archive for the same run ID cannot collide with it:

```bash
.venv/bin/python -m data.sft.import_toolathlon \
  --archive /mnt/shared_ru.ml.SZ-5_000264/sukhorukov/traces.tar.gz \
  --archive-prefix matrosov/decomposer-qwen/artifacts/gyms/toolathlon_gym \
  --run-id 20260826T122838Z-84ae95f3 \
  --expected-sha256 3983f60540f1887befbc2654db8a8d7b169c397a88d62d1017625cd480386f5f \
  --output-root /mnt/shared_ru.ml.SZ-5_000264/sukhorukov/decomposer_artifacts/evaluation/data/toolathlon_gym/imports/snapshots/3983f605
```

After committing the preparation implementation, build the release:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m data.sft.prepare \
  --spec data/sft/specs/decomposer_mixed_deepseek_qwen35_4b_nonthinking_v1_partial_3983f605_327_32k.yaml \
  --output-root /mnt/shared_ru.ml.SZ-5_000264/sukhorukov/decomposer_artifacts/datasets/sft \
  --source toolathlon-deepseek-v4-flash-0731-qwen35-4b-nonthinking-n1=/mnt/shared_ru.ml.SZ-5_000264/sukhorukov/decomposer_artifacts/evaluation/data/toolathlon_gym/imports/snapshots/3983f605/20260826T122838Z-84ae95f3
```

This build has 1,532 retained traces: 1,379 train and 153 validation. The
train split contains 1,118 Workplace and 261 Toolathlon traces; validation
contains 125 and 28 respectively. Preparation excludes 36 malformed traces
and 14 Toolathlon traces over 32K without truncating any record. The longest
retained train trace has 32,394 Qwen tokens.

Run the longest-trace, one-step smoke at high priority before the full job:

```bash
/mnt/shared_ru.ml.SZ-5_000264/sukhorukov/.venv-mls/bin/python \
  -m training.sft.run_train_jobs \
  --sanity-check \
  --filter qwen35-4b-nonthinking-mixed-v1-partial-3983f605-327-32k-smoke-4gpu \
  --priority high
```

Submit the full five-epoch run only after that smoke succeeds:

```bash
/mnt/shared_ru.ml.SZ-5_000264/sukhorukov/.venv-mls/bin/python \
  -m training.sft.run_train_jobs \
  --filter qwen35-4b-nonthinking-mixed-v1-partial-3983f605-327-32k-full-4gpu \
  --priority high
```

NeMo-Gym adapter version 3 preserves valid parallel delegation turns. When a
teacher emits several `spawn_subagent` calls in one message, preparation pairs
each call with its result by call ID and writes ordered assistant/tool pairs
with one call per assistant message. Shared visible content and hidden teacher
reasoning are retained only on the first pair. The manifest records the number
of traces, messages, and calls normalized this way. Mixed `wait`/spawn batches
remain invalid because `wait` must be emitted alone.

The historical adapter-v1 all-subagent source pair contains 2,497 rollouts.
Canonical validation excludes 467 non-success rewards, 13 invalid tool-call
traces, and seven traces with multiple calls in one assistant message. Its v3
release therefore has 1,815 train and 195 validation traces. This intentionally
drops 20 malformed traces that the original permissive dataset included.

The historical adapter-v1 frozen 26B-A4B source pair contains 2,508 rollouts.
Preparation excludes 398 non-success rewards, two invalid tool-call traces, and
six traces that emit multiple calls in one assistant message. The resulting
prompt-fixed split has 1,886 train and 216 validation traces. The GLM source
ended terminally failed after 1,253 of 1,255 rollouts with two sidecar failures;
its valid traces are included intentionally, and the v3 manifest pins the exact
source hashes and counts. A completed GLM rerun must produce a new dataset
version.

One v3 training trace has 35,044 Gemma tokens. The v3 training configs set
`data.exclude_overlength: true`, so it is recorded and excluded before TRL sees
the dataset. The effective 32K split is therefore 1,885 train and 216
validation traces; no trace is truncated.

The partial DeepSeek-manager/E4B-thinking-subagent v1 release contains 934
successful traces: 839 train and 95 validation. With teacher reasoning removed,
the longest traces have 23,132 train tokens and 13,906 validation tokens. The
8K E4B configs retain 827 train and 92 validation traces, explicitly record the
12 train and three validation exclusions, and never truncate a trace.

The new `v2-8k` and `v2-32k` releases materialize those length policies during
preparation. They require Gemma-4 E4B non-thinking token metadata on every row;
there is no fallback for a missing count, tokenizer revision, or template hash.
Training freshly tokenizes each row to construct assistant masks, verifies the
stored counts exactly, filters against `training.max_length` before TRL is
constructed, and keeps TRL's `max_length` at the same value. It never uses
truncation to repair an overlength conversation.

Prepare both releases before selecting a v2 training experiment:

```bash
uv run --group train python -m gyms.workplace_assistant.prepare sft \
  --dataset workplace-deepseek-e4b-thinking-v2-8k
uv run --group train python -m gyms.workplace_assistant.prepare sft \
  --dataset workplace-deepseek-e4b-thinking-v2-32k
```

The default v2 run remains 8K. Its smoke and full MLSpace experiments are:

```bash
uv run --with-requirements training/sft/requirements-mlspace.txt \
  python -m training.sft.run_train_jobs --sanity-check \
  --filter gemma4-e4b-nonthinking-deepseek-e4b-v2-8k-smoke-4gpu

uv run --with-requirements training/sft/requirements-mlspace.txt \
  python -m training.sft.run_train_jobs \
  --filter gemma4-e4b-nonthinking-deepseek-e4b-v2-8k-full-4gpu
```

The registered `v2-32k` experiment uses the 32K release and
`training.max_length: 32768`. It is available for local or MLSpace execution,
but its four-GPU memory envelope has not been smoke-tested and should not be
submitted as a full run until a dedicated 32K smoke configuration succeeds.

The output manifest records source hashes, reason-coded filtering counts,
split keys, the system-prompt hash, the tool-schema hash, and generated-file
hashes, canonical schema version, preparation Git revision, and portable
dataset fingerprint. Training accepts manifest v3 only and verifies this
metadata before loading a model.

## Train

Four-GPU E2B with fused CE and global batch eight:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run --group train \
  torchrun --standalone --nproc-per-node=4 \
  -m training.sft.train \
  --config training/sft/configs/gemma4_e2b_nonthinking_4gpu_liger_workplace_26b_v3.yaml
```

Four-GPU E4B with fused CE and global batch four:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run --group train \
  torchrun --standalone --nproc-per-node=4 \
  -m training.sft.train \
  --config training/sft/configs/gemma4_e4b_nonthinking_4gpu_liger_workplace_26b_v3.yaml
```

For the 8K DeepSeek/E4B v1 run, first launch the one-step smoke experiment into
the separate sanity artifact root:

```bash
uv run --with-requirements training/sft/requirements-mlspace.txt \
  python -m training.sft.run_train_jobs \
  --sanity-check \
  --filter gemma4-e4b-nonthinking-deepseek-e4b-v1-8k-smoke-4gpu
```

After the smoke run succeeds, launch the full experiment:

```bash
uv run --with-requirements training/sft/requirements-mlspace.txt \
  python -m training.sft.run_train_jobs \
  --filter gemma4-e4b-nonthinking-deepseek-e4b-v1-8k-full-4gpu
```

Both use the student prompt, omit teacher reasoning, exclude traces longer than
8,192 tokens, and train with four GPUs at global batch four without gradient
accumulation. The smoke run tokenizes the complete release, selects the four
longest retained train traces so every rank exercises one in the first global
batch, evaluates after its single optimizer step, and exports a complete final
checkpoint.

The retained full-run configs are:

```text
training/sft/configs/gemma4_e2b_nonthinking_4gpu_liger_workplace_26b_v3.yaml
training/sft/configs/gemma4_e4b_nonthinking_4gpu_liger_workplace_26b_v3.yaml
```

Both configs use unquantized full-parameter training, BF16, FSDP2 with
activation checkpointing, 32,768-token contexts, no packing, and batch one per
GPU. Configs request `training.global_batch_size`; startup validates exact
divisibility and derives gradient accumulation as global batch divided by world
size and per-device batch. The E2B config requests global batch eight, so its
four-GPU run derives two accumulation steps and uses a `2e-5` learning rate.
The E4B config requests global batch four, so
four GPUs derive no gradient accumulation; its learning rate is linearly scaled
from `2e-5` to `1e-5`. Carrying FSDP
gradients into a second microbatch left insufficient room for Liger's fused-CE
weight-gradient temporary. The E4B MLSpace experiment also
sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; varying sequence lengths
otherwise fragmented reserved memory. Liger 0.8.1's outer Gemma-4 fused linear
cross-entropy avoids materializing the full token-by-vocabulary logits tensor,
although its BF16 gradient accumulation can still need a temporary 2.5 GiB
FP32 vocabulary projection. The E4B config leaves RMSNorm, GeGLU, RoPE,
attention, and every other kernel native so the loss path is the only changed
implementation. Prepared traces are tokenized before model loading. Standard
configs fail if any trace would be truncated or has an empty assistant mask.
The v3 configs explicitly record and exclude overlength traces, then enforce
the same no-truncation invariant on the effective dataset. Each rank loads the
checkpoint into CPU memory before FSDP2 shards it;
Accelerate 1.14's RAM-efficient loader is incompatible with Gemma-4's persistent
buffer tensors. Both full-run configs have a five-epoch ceiling and stop after
two consecutive epoch evaluations without any decrease in validation loss.

Both full-run configs disable only the cuDNN SDPA backend because the first full E2B
run hit a cuDNN `mha_graph` execution failure during attention recomputation in
backward. PyTorch's flash, memory-efficient, and math SDPA backends remain
enabled as fallbacks. The resolved state of every SDPA backend is written to
`resolved_config.json`, connected to ClearML, and printed in the local console
log.

Liger fused CE is enabled for both retained full runs. Version
0.8.1 officially patches `Gemma4ForConditionalGeneration`, including the outer
multimodal forward used by the E4B-it checkpoint. When enabled, the trainer
replaces TRL's incompatible chunked-NLL setting with Liger's fused linear
cross-entropy. Fused CE does not return full logits, so token accuracy remains
available but entropy is intentionally absent from train/eval logs. The package
is a Python wheel and compiles kernels with Triton at runtime; `nvcc` is not
required.

To train the teachers' available hidden reasoning, use a distinct output
directory and a 65,536-token context:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run --group train \
  torchrun --standalone --nproc-per-node=4 \
  -m training.sft.train \
  --config training/sft/configs/gemma4_e2b_nonthinking_4gpu_liger_workplace_26b_v3.yaml \
  --include-reasoning \
  --max-length 65536 \
  --output-dir /mnt/shared_ru.ml.SZ-5_000264/sukhorukov/decomposer_artifacts/training/sft/checkpoints/gemma4-e2b-thinking
```

Internal epoch checkpoints use FSDP's sharded state-dict format so saving does
not gather the full model onto rank 0; this script reconstructs the annotated
training template when resuming. E2B checkpoints are written about every 229
optimizer steps; E4B global-batch-4 checkpoints are written about every 458
steps. On the effective v3 dataset, the corresponding intervals are about 236
and 472 steps per epoch.
`save_total_limit: 2` retains the best and latest recovery points. After
training, the temporary final shards
are merged on CPU and the exported `final/` model restores Gemma's canonical
template for inference. Future exports also contain `generation_config.json`
with Gemma-4's tokenizer EOS, `<turn|>`, and `<|tool_response>` stop IDs; this
preserves the pretrained `[1, 106, 50]` multi-EOS behavior for Transformers and
serving runtimes. `training_summary.json` records the completed epoch and step,
best validation loss and checkpoint, resolved batch settings, exported stop
IDs, and whether early stopping fired.

Gemma-4 KV-shared layers intentionally omit unused `k_norm` weights after SFT,
but vLLM 0.24 still requires those parameters during checkpoint validation.
Create a zero-copy compatibility view for vLLM without changing `final/`:

```bash
uv run --group train python -m training.sft.vllm_compat \
  --source /path/to/run/final \
  --output /path/to/run/final-vllm
```

The derived directory symlinks the original model tensor and adds identity
`k_norm` tensors only for the KV-shared layers. Transformers consumers should
continue to use `final/`; vLLM consumers should use `final-vllm/`.

## ClearML

ClearML is disabled by default. Configure the self-hosted server without adding
credentials to the repository:

```bash
export CLEARML_API_HOST=https://api.example.internal
export CLEARML_WEB_HOST=https://app.example.internal
export CLEARML_FILES_HOST=https://files.example.internal
export CLEARML_API_ACCESS_KEY=...
export CLEARML_API_SECRET_KEY=...
```

Then add `--clearml` to a direct training command. MLSpace SFT launchers enable
it automatically and pass only this private config-file path:

```bash
export CLEARML_CONFIG_FILE=/mnt/shared_ru.ml.SZ-5_000264/sukhorukov/.secrets/clearml.conf
chmod 600 "$CLEARML_CONFIG_FILE"
```

Tasks use project `decomposer` and tags `sft`, `workplace-assistant`, plus the
actual student model (`gemma-4-E2B-it` or `gemma-4-E4B-it`). Each invocation
creates a new ClearML task. The task logs resolved
hyperparameters, source/version information, the data manifest, console output,
losses, learning rate, gradient norm, token accuracy, and throughput. Every
numeric metric has an independent `train/<metric>` or `eval/<metric>` scalar
plot. `train/weight_norm` is the global L2 norm of trainable parameters after
the optimizer update; it is reduced directly over FSDP shards at the first
training log, every `clearml.weight_norm_interval_steps` optimizer steps, and
the final log. The checked-in interval of 10 keeps its expected overhead below
0.5%. Large checkpoint upload is disabled by default; checkpoints remain on
NFS. Set `clearml.log_model: true` only when checkpoint upload is desired.

## Real smoke run

The smoke config loads the real E2B checkpoint, uses two GPUs and two prepared
train and validation examples, performs one optimizer step plus evaluation,
and exports a full checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run --group train \
  torchrun --standalone --nproc-per-node=2 \
  -m training.sft.train \
  --config training/sft/configs/gemma4_e2b_smoke.yaml
```

## MLSpace jobs

The stable artifact path is a symlink into shared NFS:

```text
/home/jovyan/decomposer-artifacts
  -> /mnt/shared_ru.ml.SZ-5_000264/sukhorukov/decomposer_artifacts
```

Training outputs live under `training/sft/jobs/`, sanity outputs under
`training/sft/jobs_sanity/`, staged clean Git snapshots under `code/`, and
shared training environments under `venvs/sft/<uv-lock-hash>/`. The launcher
syncs only the base and `train` dependency group. It does not require `nvcc`;
Gemma-4 Liger experiments use the pinned Python wheel and Triton JIT.

Experiments are registered in `training.sft.experiments`. The `mls` submitter is
kept in an isolated environment because its Click 8.1.8 pin conflicts with the
Hugging Face stack's Click 8.4.2 requirement. Preview every payload without
staging code, creating persistent environments, or submitting jobs:

```bash
uv run --no-project \
  --with-requirements training/sft/requirements-mlspace.txt \
  python -m training.sft.run_train_jobs --dry
uv run --no-project \
  --with-requirements training/sft/requirements-mlspace.txt \
  python -m training.sft.run_eval --dry
```

Resume an incomplete run only by explicit request. The launcher selects the
highest numeric `checkpoint-N` that has a completed `trainer_state.json`:

```bash
uv run --no-project \
  --with-requirements training/sft/requirements-mlspace.txt \
  python -m training.sft.run_train_jobs \
  --filter gemma4-e2b-nonthinking --resume-latest
```

Job descriptions contain only the `#sukhorukov` tag. Real submission refuses a
dirty tracked worktree, stages the committed source by Git hash, skips completed
runs via `training_summary.json`, and deduplicates Pending/Running jobs by their
normalized descriptions. Combined `torchrun` output is also persisted as
`console.log` in each run directory with pipefail-safe exit propagation; save an
MLSpace API log snapshot as `mlspace.log` after the job reaches terminal state.
