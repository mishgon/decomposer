# Decomposer supervised fine-tuning

This workflow converts successful Decomposer rollouts into Gemma-4 tool-calling
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
uv run --group train python -m data.sft.prepare \
  --spec data/sft/specs/workplace_all_v3.yaml \
  --output-root /mnt/shared_ru.ml.SZ-5_000264/sukhorukov/decomposer_artifacts/datasets/sft
```

Build the 26B-A4B non-thinking source pair with:

```bash
uv run --group train python -m data.sft.prepare \
  --spec data/sft/specs/workplace_26b_nonthinking_v3.yaml \
  --output-root /mnt/shared_ru.ml.SZ-5_000264/sukhorukov/decomposer_artifacts/datasets/sft
```

Both specifications use exact reward `1.0`, prompt-fixed validation fraction
`0.1`, and seed `42`. All teacher variants of one prompt are assigned to the
same split. The builder requires a clean Git worktree and refuses to replace an
existing `<dataset-id>/<version>` directory.

The all-subagent source pair contains 2,497 rollouts. Canonical validation
excludes 467 non-success rewards, 13 invalid tool-call traces, and seven traces
with multiple calls in one assistant message. Its v3 release therefore has
1,815 train and 195 validation traces. This intentionally drops 20 malformed
traces that the original permissive dataset included.

The frozen 26B-A4B source pair contains 2,508 rollouts. Preparation excludes 398
non-success rewards, two invalid tool-call traces, and six traces that emit
multiple calls in one assistant message. The resulting prompt-fixed split has
1,886 train and 216 validation traces. The GLM source ended terminally failed
after 1,253 of 1,255 rollouts with two sidecar failures; its valid traces are
included intentionally, and the v3 manifest pins the exact source hashes and
counts. A completed GLM rerun must produce a new dataset version.

One v3 training trace has 35,044 Gemma tokens. The v3 training configs set
`data.exclude_overlength: true`, so it is recorded and excluded before TRL sees
the dataset. The effective 32K split is therefore 1,885 train and 216
validation traces; no trace is truncated.

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
