# Decomposer supervised fine-tuning

This package turns successful Decomposer teacher traces into a versioned dataset
release and trains a Qwen3.5 student manager on it, with LoRA or full SFT. The
loss covers only the manager's own turns; the task, tool schemas, tool results
and subagent summaries are context.

## Pipeline

1. **Collect traces.** tau2 and Workplace traces are ordinary gym runs with a
   teacher experiment (`gyms/tau2_gym/README.md`,
   `gyms/workplace_assistant/README.md`). Toolathlon and WideSeek have SFT
   collectors that repeat tasks until they have enough successes
   (`sft/toolathlon_gym/README.md`, `sft/wideseek/README.md`).
2. **Snapshot each run** into an immutable, digest-named copy of the files its
   adapter reads:
   `uv run --group train python -m sft.snapshots --adapter <adapter> --source <run dir>`.
3. **Write a build specification** in `sft/specs/` that names the snapshots,
   and commit it.
4. **Build the release:**
   `.venv/bin/python -m sft.prepare --spec sft/specs/<spec>.yaml --output-root /mnt/share14T-2/sukhorukov/decomposer_artifacts/datasets/sft`.
5. **Train:** `sft/train_qwen35_h200.sh <config> <gpus>` on Hertz-2. The run
   exports the merged model to `<output_dir>/final/`, which `evals/<gym>`
   evaluates (`evals/README.md`).

## Install

From the repository root:

```bash
uv sync --group train
```

The root and `external/Gym` environments remain separate.

## Records

A release holds `train.jsonl`, `validation.jsonl` and `manifest.json`. Each line
is one canonical record of one successful trace:

- **Messages.** The system prompt, then the task and the manager's turns with
  their tool results, ending in the final answer. The system prompt is the one
  the manager saw: the Decomposer prompt rendered with the 60-second run budget,
  as `create_decomposer_agent` sends it (`sft.chat_tools.decomposer_system_prompt`).
  The manifest records its SHA-256.
- **Tools.** The tool schemas the source's manager saw, so every gym keeps its
  own agent types. The manager's tools are exactly `new(agent_type_id)`,
  `fork(agent_id)`, `run(agent_id, prompt)` and `wait()`; each source uses one
  schema, whose hash the manifest records (`tool_schema_sha256`).
- **Turns as emitted.** An assistant message with several tool calls stays one
  message, and its results follow in the order the harness returned them, so
  the student learns the same turns that OPD and RL later sample. Mistakes the
  core answered stay, with the core's answer as their result: calls of unknown
  tools or agents, malformed arguments, refused parallel calls
  (`PARALLEL_WAIT_CALL_ERROR`, `PARALLEL_FORK_RUN_CALL_ERROR` and
  `PARALLEL_RUN_CALL_ERROR` in `decomposer.prompts`), answers that are empty or
  come before every run was collected, and a third assignment of the same task.
  The core follows the last two with an injected user message
  (`EARLY_RESPONSE_ERROR`, `EMPTY_RESPONSE_ERROR`,
  `DECOMPOSER_GRACEFUL_SHUTDOWN_REQUEST`). A trace must still be well formed:
  one task, every call answered, and a final text answer.
- **Reasoning.** The student trains non-thinking. Records keep the teacher's
  reasoning (`teacher_reasoning`) as metadata, and training never renders it.
- **Metadata.** `source` (adapter, source and task IDs, teacher), `outcome`
  (reward) and `attributes` (category and adapter-specific fields).

Releases do not depend on a model: they hold messages and tool schemas, with no
token counts and no length limit. Training tokenizes them and decides what to do
with long records (see [Length](#length)).

**Selection.** A release takes every successful trace: reward 1 for tau2 and
Workplace (`selection.policy: exact_reward`), and the collector's own success
rule for Toolathlon and WideSeek (`collector_qualifies`). Traces that are not
well formed are excluded with a reason code, and the manifest counts every
exclusion per source.

**Split.** A task goes to validation when the hash of `split.seed` and its group
ID falls under `split.validation_fraction` (`0.1`, seed `42`). All rollouts of a
task land on the same side, and a task's side depends only on its own ID: adding
data never moves an existing task. The group ID is the task prompt's hash for
tau2 and Workplace and the task name for Toolathlon and WideSeek.

**Manifest.** It records the specification, the snapshots and their files,
reason-coded exclusion counts, the split and its validation task IDs, record
counts by teacher, gym and category, the system-prompt and tool-schema hashes,
the files' sizes and hashes, the preparation Git revision, and a portable
dataset fingerprint. Training accepts manifest format 3 only and verifies it
before loading a model.

## Build a release

```bash
.venv/bin/python -m sft.prepare --spec sft/specs/<spec>.yaml \
  --output-root /mnt/share14T-2/sukhorukov/decomposer_artifacts/datasets/sft
```

The builder refuses to replace an existing `<dataset-id>/<version>` directory.
A specification (`spec_version: 4`) names the dataset and version, the sources
by snapshot digest, the selection and the split:

```yaml
spec_version: 4
dataset:
  id: decomposer-manager-sft
  version: <version>
policy:
  id: decomposer-default
sources:
  - id: tau2-qwen38-flash-nonthinking-broad-v1-n15
    adapter: nemo_gym
    snapshot: sha256:<hex>
    benchmark: tau2_gym
    environment: tau2_gym
    partition: train
    teacher: qwen38-flash-non-thinking
    expected_native_rollouts: <rollouts>
    expected_candidates: <rollouts>
selection:
  policy: exact_reward
  success_reward: 1.0
split:
  strategy: prompt_fixed
  seed: 42
  validation_fraction: 0.1
```

`expected_native_rollouts` pins how many rollouts the source holds; a different
count stops the build. Toolathlon and WideSeek sources add their own fields (see
below).

### Snapshots, releases and versioning

Build specifications (`spec_version: 4`) name every source by a snapshot
digest. A snapshot is an immutable copy of exactly the files
the source's adapter reads:

```bash
uv run --group train python -m sft.snapshots --adapter nemo_gym \
  --source /path/to/native/run
```

The command prints the reference to put in the spec, `snapshot: sha256:<hex>`.

- **Layout.** A snapshot lives in
  `/mnt/share14T-2/sukhorukov/decomposer_artifacts/datasets/sft/snapshots/<adapter>/<first 16 hex>/`
  (`--output-root` changes the root). It keeps the files at their original
  relative paths, plus `snapshot.json` with each file's size and hash and the
  origin path and host.
- **Digest.** It covers only the adapter name and the file contents. Renaming or
  moving the original source does not change it, and snapshotting unchanged files
  again reuses the existing snapshot.
- **Exact copies.** Files are copied byte for byte, so a snapshot holds whatever
  the traces hold, including endpoints and any keys a task exposed. Snapshots
  stay local and are never published.
- **Build.** `python -m sft.prepare` finds each snapshot under `--snapshot-root`
  (the root above by default) and verifies every file before its adapter reads
  it. The manifest records each source's `snapshot`, and the digests are part of
  the release fingerprint.
- **Git.** The builder records `git rev-parse HEAD`. It refuses to build unless
  tracked files are clean, `sft/` and `src/` have no untracked files, and the
  spec itself is committed.
- **Training.** `data.expected_fingerprint` in a training config pins the exact
  release; training refuses any other.

Adding data never changes a snapshot or a release: new traces for a gym become a
new snapshot, used by a new spec file with a new dataset version. A new gym needs
an adapter in `sft/adapters/registry.py` (`ADAPTERS`, `ADAPTER_VERSIONS`) and the
list of files it reads in `SNAPSHOT_FILES`. Bump an adapter's `ADAPTER_VERSION`
whenever its output changes. Snapshots and releases are self-contained
directories, so either can later be uploaded unchanged, for example to a Hugging
Face dataset repository, and pinned by revision.

**Dataset versions.** Releases carry a semantic version, MAJOR.MINOR.PATCH,
decided when each release is cut:

- **MAJOR.** A change that breaks training or comparisons: a new record format,
  a different system prompt, tools or core interface, a different teacher or
  harness setup, or a reassigned train/validation split.
- **MINOR.** More data that stays compatible: new snapshots, finished
  collections, a new gym, or optional new fields.
- **PATCH.** Fixes without new data, such as dropping or correcting a few
  records.

A new spec names the dataset `decomposer-manager-sft` and sets `dataset.version`
to that version, so the release directory
(`…/datasets/sft/decomposer-manager-sft/<version>/`), the manifest and the
Hugging Face folder share one name. A published version never changes: any
change, however small, becomes a new version. A minor release must keep every
existing record on its side of the split, which the split does as long as
`split.seed` and `split.validation_fraction` stay the same (see
[Records](#records)).

**Publishing.** Releases are published unchanged to the private Hugging Face
dataset `decomposer-datasets/decomposer-manager-sft`: one folder per version
(`<version>/` with `train.jsonl`, `validation.jsonl` and `manifest.json`) and
one tag per version (`v<version>`). Access is by membership in the
`decomposer-datasets` organization. The dataset card's source is
`sft/hf/decomposer-manager-sft.md`; with each release, update its versions table
and changelog, and upload it as the repository's `README.md` in the same commit
as the data. Stage the card and the version folder in one directory, then:

```bash
.venv/bin/hf upload decomposer-datasets/decomposer-manager-sft <stage-dir> . \
  --repo-type dataset --commit-message "Add <version>"
.venv/bin/hf repos tag create decomposer-datasets/decomposer-manager-sft \
  v<version> --repo-type dataset
```

Check the uploaded files' sha256 against `prepared_files` in the manifest.

### Toolathlon collections

The Toolathlon adapter reads a collection written by `sft/toolathlon_gym`
(`trace_format: toolathlon_langgraph_v1`), snapshotted with
`--adapter toolathlon_gym`. The snapshot takes every finished episode: its
`trace.json` and `runtime.json`, and its evaluation `result.json`. Episodes
still running have no evaluation yet and are left out, so a running collection
can be snapshotted.

- **Selection.** `selection.policy: collector_qualifies` keeps exactly the
  traces the collector counts as successes
  (`sft/toolathlon_gym/scheduler.LaunchOutcome.qualifies`): the agent finished,
  and the task passed strictly or its check fraction exceeds
  `selection.success_threshold` (0.9 for the collector). The record's reward is
  1 for a strict pass and the check fraction otherwise.
- **Tools.** The traces store no tool schemas. The source declares the subagent
  types the collection offered (`native_subagent_types`, from
  `gyms/toolathlon_gym/agents.py`), and the adapter rebuilds the manager's tools
  from them with the Decomposer core. A trace that creates an undeclared type
  stops the build.
- **Messages.** The first message must equal the task in `runtime.json`.
  Mistakes the core answered stay, as for the NeMo Gym adapter.

```yaml
- id: toolathlon-qwen38-flash-nonthinking-coverage
  adapter: toolathlon_gym
  snapshot: sha256:<hex>
  benchmark: toolathlon_gym
  environment: toolathlon_gym
  partition: train
  teacher: qwen38-flash-non-thinking
  trace_format: toolathlon_langgraph_v1
  native_subagent_types:
    - id: qwen_3_5_4b_thinking
      description: Qwen3.5-4B thinking agent equipped with all the available tools.
  selection:
    policy: collector_qualifies
    success_threshold: 0.9
  expected_native_rollouts: <finished episodes>
  expected_candidates: <finished episodes>
```

### WideSeek collections

The WideSeek adapter (`adapter: wideseek`, `trace_format: wideseek_langgraph_v2`)
reads a collection written by `sft/wideseek`, snapshotted with
`--adapter wideseek`. Each attempt's `result.json` names the execution it scored.
The snapshot takes, per finished attempt, that result, the execution's
`trace.json` and its model-call log `model_calls.jsonl`. The log holds tools and
the system prompt only on each thread's first call, so the manager's first call
holds the ones the manager saw. Restarted executions that no result names are
left out. Collections written before the gym logged `model_calls.jsonl`
(`1994bb3`, 2026-10-07) predate the current core prompt, which training
requires, so the adapter does not read them.

- **Selection.** `collector_qualifies` applies the collector's rule
  (`gyms/wideseek/metrics.qualifies`): a normal finish, a scored
  evaluation, no cleanup errors, and a score of 1 or above `success_threshold`.
  The record's reward is the score.
- **Tools and prompt.** Records keep the logged tools. The logged system prompt
  must be the teacher prompt, or the build stops.
- **Messages.** The first message must equal the task in the logged call. Mistakes
  the core answered stay.

```yaml
- id: wideseek-qwen38-flash-nonthinking-coverage
  adapter: wideseek
  snapshot: sha256:<hex>
  benchmark: wideseek
  environment: wideseek
  partition: train
  teacher: qwen38-flash-non-thinking
  trace_format: wideseek_langgraph_v2
  selection:
    policy: collector_qualifies
    success_threshold: 0.9
  expected_native_rollouts: <finished attempts>
  expected_candidates: <finished attempts>
```

### Release 1.0.0 of decomposer-manager-sft

Release 1.0.0 has four sources. Each has the Qwen3.8-Flash non-thinking manager
and Qwen3.5-4B-unlooped thinking subagents, collected with the old core prompt.
It was built on Hertz-2 at commit `53a88d8`, the only commit that reproduces it
and the one that holds its specification:

```bash
.venv/bin/python -m sft.prepare \
  --spec sft/specs/decomposer_manager_sft_1.0.0.yaml \
  --output-root /mnt/share14T-2/sukhorukov/decomposer_artifacts/datasets/sft
```

| Source | Snapshot | Rollouts | Selected | Rule |
|---|---|---:|---:|---|
| tau2 | `170d2e3d…` | 4,843 | 2,948 | reward 1 |
| Workplace | `c36c909e…` | 1,255 | 988 | reward 1 |
| Toolathlon | `c9ea4cfe…` | 1,095 | 214 | collector: pass or check fraction > 0.9 |
| WideSeek | `c160625c…` | 253 | 70 | collector: score 1 or > 0.9 |

That gives 4,220 records (3,797 train, 423 validation) with fingerprint
`fb6d86f6…c0a25556`. Each gym keeps its own tool schema. With the Qwen3.5
template the release has 38.4M tokens, 8.86M of them supervised.

| Gym | Median tokens | Max | Supervised share | Cut at 16K | Kept at 16K | Cut at 32K | Kept at 32K |
|---|---:|---:|---:|---:|---:|---:|---:|
| tau2 | 6.8K | 65.5K | 54% | 93 | 97% | 4 | 99% |
| Workplace | 6.0K | 30.7K | 13% | 6 | 99% | 0 | 100% |
| Toolathlon | 29.3K | 64.3K | 27% | 191 | 44% | 84 | 80% |
| WideSeek | 20.5K | 54.8K | 6% | 45 | 55% | 17 | 88% |
| All | | | | 335 | 80% | 105 | 93% |

"Kept" is the share of supervised tokens left after cutting from the end; no
record is left without any. The Toolathlon snapshot holds the LLM proxy master
key (the task containers exposed it, and subagents listed their environment), so
the snapshot must not be published.

Version 1.0.0 is published unchanged on Hugging Face as
`decomposer-datasets/decomposer-manager-sft` (folder `1.0.0/`, tag `v1.0.0`).
The published files hold no proxy-key value. In the Toolathlon snapshot the key
appears inside subagent runs (149 episodes), which records leave out, and in
`wait` results of 3 episodes, none of which qualified; one record names the
variable without a value. After joining the organization, download it with:

```bash
hf download decomposer-datasets/decomposer-manager-sft --repo-type dataset \
  --revision v1.0.0 --local-dir <dir>
```

## Train

Training runs on Hertz-2 through `sft/train_qwen35_h200.sh`, which sets the
environment from `docs/sft_qwen35_h200_benchmark.md` and refuses any listed GPU
that holds memory:

```bash
cd ~/decomposer_sft
nohup sft/train_qwen35_h200.sh <config> 4,5,6,7 > ~/<run>.log 2>&1 &
```

Arguments after the GPUs go to `python -m sft.train` and override the config
(for example `--max-steps`, `--max-length`, `--output-dir` or
`--resume-from-checkpoint`; see `--help`).

**Checks at start.** Before loading the model, training refuses to run unless:

- the process count equals `run.expected_world_size` (default 2);
- the model is Qwen3.5 (`Qwen3_5ForConditionalGeneration`), which the template
  patch, the LoRA targets and the kernels assume;
- the manifest is format 3, its fingerprint matches its contents and, when set,
  `data.expected_fingerprint`, and the files match their recorded sizes, hashes
  and record counts;
- the manifest's system-prompt hash and every record's first message equal the
  prompt the current core renders;
- the output directory is empty, unless the run resumes or sets
  `run.overwrite_output_dir`;
- with `run.required_runtime_profile` set, the pinned kernels are the ones loaded.

**Config.** A training config is a YAML file with these sections:

| Section | Holds |
|---|---|
| `model` | `name_or_path`, `revision`, `dtype`, `attn_implementation`, `trust_remote_code`, `qwen35_native_gva`, `freeze_modules` |
| `data` | `train_file`, `validation_file`, `manifest_file`, `expected_fingerprint`, `num_proc`; `max_train_samples`/`max_eval_samples` for smokes and `longest_train_samples`/`stratified_train_samples` for benchmarks |
| `training` | TRL `SFTConfig` fields, with `global_batch_size` instead of `gradient_accumulation_steps` (derived from the world size and per-device batch) and `max_length` |
| `lora` | `r`, `alpha`, `dropout`; without it, training is full SFT |
| `clearml` | `enabled`, `project`, `task`, `tags`, `weight_norm_interval_steps`, `log_model`, `output_uri` |
| `run` | `expected_world_size`, `early_stopping` (`patience`, `threshold`), `resume_from_checkpoint`, `overwrite_output_dir`, `disable_cudnn_sdpa`, `required_runtime_profile`, `benchmark` |

No config is checked in. Start from the release 1.0.0 LoRA config, which lives
in the parent of the commit that removed the old configs:

```bash
git show "$(git log --diff-filter=D -1 --format=%h -- sft/configs)^:sft/configs/qwen35_4b_unloop_nonthinking_manager_sft_1.0.0_lora_32k_4gpu.yaml"
```

Point its `data` section at the new release and its `expected_fingerprint`.
Fractional `eval_steps`/`save_steps` (for example `0.0667`, a fifth of an epoch
in a 3-epoch run) need no recount when the release grows.

The release 1.0.0 runs used the best cell of the v3 grid: 32K and global batch
8. LoRA (`r: 32`, `alpha: 64`, `dropout: 0.05`) trained at learning rate 2e-4;
full SFT took the benchmark's full recipe (fp32 weights, frozen `model.visual`)
at learning rate 2e-5. Both ran 3 epochs, evaluated and saved every 0.2 epoch
(95 steps), kept the checkpoint with the lowest validation loss and stopped
after three evaluations without improvement. A run resumed with
`--resume-from-checkpoint <output>/checkpoint-<step>` (passed after the GPUs to
`sft/train_qwen35_h200.sh`) keeps its early-stopping count.

**Evaluation.** Validation loss is reported for the whole split as
`eval_all_loss`, which picks the checkpoint, and per gym as `eval_<gym>_loss`
when the split mixes gyms. ClearML plots each one separately and also all of
them together in `eval/loss_by_dataset`. A config's
`metric_for_best_model: eval_loss` means `eval_all_loss`.

**Outputs.** A run writes into `training.output_dir`:

- `resolved_config.json`: the config after defaults and overrides, plus runtime
  facts (resolved batch, token statistics, cut records, kernels);
- `data_manifest.json`: a copy of the release manifest;
- `training_chat_template.jinja`: the chat template with the assistant-mask
  markers that training used;
- `checkpoint-<step>/`: Trainer checkpoints; `save_total_limit: 2` keeps the best
  and the latest;
- `final/`: the model evaluation serves, in bf16, with the tokenizer, the
  canonical chat template and `generation_config.json` (the tokenizer's EOS,
  then the checkpoint's own EOS IDs);
- `final-adapter/`: the trained LoRA adapter, for LoRA runs;
- `training_summary.json`: the completed epoch and step, best validation loss and
  checkpoint, resolved batch, token statistics, cut records, the generation
  config, and whether early stopping fired (`benchmark_summary.json` in
  benchmark mode).

`sft/grid.py` runs a learning-rate × global-batch grid over one config, one run
per GPU group at a time, and waits for each group's GPUs to be free:

```bash
nohup .venv/bin/python -m sft.grid \
  --config <config> \
  --learning-rates 5e-5 1e-4 2e-4 --global-batch-sizes 8 16 32 \
  --gpu-groups 0,1,2,3 4,5,6,7 --output-root <grid root> > grid.log 2>&1 &
.venv/bin/python -m sft.grid --report <grid root>
```

Each cell `<grid root>/lr<lr>-gb<batch>/` holds its config, `train.log` and the
run in `run/`. A cell whose `run/` exists is skipped, so a grid resumes after an
interruption, and a symlinked `run/` reuses an earlier run as a cell. The report
lists each cell's best epoch with its overall and per-gym validation loss and
saves them to `summary.json`.

### Length

`training.max_length` sets the longest sequence a run trains on; it defaults to
16,384 (`max_length: null` means no limit). A record over it is cut from the end:
TRL keeps its first `max_length` tokens, which hold the system prompt, the tools
and the task, and drops a record left with no supervised token.
`training_summary.json` lists the cut records under `overlength_truncations`,
each with the supervised tokens it keeps.

Training tokenizes each record once, at start-up, with the model's training
template. That pass checks the assistant mask and measures lengths, and TRL
trains on its token IDs without tokenizing again.

### LoRA

A top-level `lora:` section (`r`, `alpha`, `dropout`) trains LoRA adapters
instead of all weights; without it, training is full SFT. LoRA is defined for
Qwen3.5 and runs on DDP (`training.fsdp: false`), with the base model in bf16
(`model.dtype: bfloat16`): the base stays frozen, and PEFT keeps the adapters in
fp32. Adapters go on the language model's linear projections
(`self_attn.{q,k,v,o}_proj`, `linear_attn.{in_proj_qkv,in_proj_z,out_proj}`,
`mlp.{gate,up,down}_proj`); the vision tower, `lm_head`, the short convolutions
and the tiny `in_proj_a/b` gate projections get none. Because the whole base is
frozen, `ddp_find_unused_parameters: false` works. The run saves the adapter to
`final-adapter/` and the merged bf16 model to `final/`, so evaluation and serving
use `final/` for both modes.

Under bf16 autocast (`training.bf16: true`), the trainer turns off PEFT's cast of
each adapted layer's input to the fp32 adapter dtype: autocast runs the adapter
matmuls in bf16 anyway, so the cast only copied every adapted activation there and
back. The adapters and their optimizer state stay fp32.

The recommended settings:

```yaml
lora:
  r: 32
  alpha: 64
  dropout: 0.05
```

These are the settings of the 3-epoch release 1.0.0 runs. Dropout costs about 4%
of throughput (`docs/sft_qwen35_h200_benchmark.md`); a single-epoch run can set
it to 0, since one pass leaves little to overfit.

### Kernels, SDPA and FSDP

`training.use_liger_kernel: true` enables Liger fused CE, and the trainer then
replaces TRL's incompatible chunked-NLL setting with Liger's fused linear
cross-entropy. Fused CE does not return full logits, so token accuracy remains
available but entropy is intentionally absent from train/eval logs. The package
is a Python wheel and compiles kernels with Triton at runtime; `nvcc` is not
required.

`run.disable_cudnn_sdpa: true` disables only the cuDNN SDPA backend, which once
failed with a cuDNN `mha_graph` error during attention recomputation in
backward. PyTorch's flash, memory-efficient, and math SDPA backends remain
enabled as fallbacks. The resolved state of every SDPA backend is written to
`resolved_config.json`, connected to ClearML, and printed in the local console
log.

Full SFT with FSDP (`training.fsdp`, for GPUs that cannot hold the whole model,
such as MLSpace's H100s) saves internal checkpoints in FSDP's sharded
state-dict format, so saving does not gather the full model onto rank 0, and
merges the final shards on CPU after training. `save_total_limit: 2` retains the
best and latest recovery points. The exported `final/` restores the model's
canonical chat template for inference, and its `generation_config.json` lists
the tokenizer's EOS followed by the checkpoint's own EOS IDs.
`training_summary.json` records the completed epoch and step, best validation
loss and checkpoint, resolved batch settings, the generation config, and whether
early stopping fired.

Full SFT keeps fp32 master weights (fp32 DDP weights, or FSDP's upcast), so the
export rewrites `final/` in bf16, with a bf16 `config.json`, on both paths.
Evaluation serves bf16 anyway (`--dtype bfloat16`), and resuming uses the Trainer
checkpoints, which are unchanged.

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
export CLEARML_CONFIG_FILE=/home/sukhorukov/.secrets/clearml.conf
chmod 600 "$CLEARML_CONFIG_FILE"
```

Tasks take their project, name and tags from the config's `clearml` section.
Each invocation creates a new ClearML task. The task logs resolved
hyperparameters, source/version information, the data manifest, console output,
losses, learning rate, gradient norm, token accuracy, and throughput. Every
numeric metric has an independent `train/<metric>` or `eval/<metric>` scalar
plot. `train/weight_norm` is the global L2 norm of trainable parameters after
the optimizer update; it is reduced directly over FSDP shards at the first
training log, every `clearml.weight_norm_interval_steps` optimizer steps, and
the final log. The checked-in interval of 10 keeps its expected overhead below
0.5%. Large checkpoint upload is disabled by default; checkpoints remain on
NFS. Set `clearml.log_model: true` only when checkpoint upload is desired.

## MLSpace jobs

The stable artifact path is a symlink into shared NFS:

```text
/home/jovyan/decomposer-artifacts
  -> /home/sukhorukov/decomposer_artifacts
```

Training outputs live under `sft/jobs/`, sanity outputs under
`sft/jobs_sanity/`, staged clean Git snapshots under `code/`, and
shared training environments under `venvs/sft/<uv-lock-hash>/`. The launcher
syncs only the base and `train` dependency group. It does not require `nvcc`:
Liger is a pinned Python wheel whose kernels Triton compiles at runtime.

Experiments are registered in `sft.experiments`. The `mls` submitter is
kept in an isolated environment because its Click 8.1.8 pin conflicts with the
Hugging Face stack's Click 8.4.2 requirement. Preview every payload without
staging code, creating persistent environments, or submitting jobs:

```bash
uv run --no-project \
  --with-requirements sft/requirements-mlspace.txt \
  python -m sft.run_train_jobs --dry
uv run --no-project \
  --with-requirements sft/requirements-mlspace.txt \
  python -m sft.run_eval --dry
```

Resume an incomplete run only by explicit request. The launcher selects the
highest numeric `checkpoint-N` that has a completed `trainer_state.json`:

```bash
uv run --no-project \
  --with-requirements sft/requirements-mlspace.txt \
  python -m sft.run_train_jobs \
  --filter <experiment> --resume-latest
```

Job descriptions contain only the `#sukhorukov` tag. Real submission refuses a
dirty tracked worktree, stages the committed source by Git hash, skips completed
runs via `training_summary.json`, and deduplicates Pending/Running jobs by their
normalized descriptions. Combined `torchrun` output is also persisted as
`console.log` in each run directory with pipefail-safe exit propagation; save an
MLSpace API log snapshot as `mlspace.log` after the job reaches terminal state.

## History

Until 2026-10-13 this package also held Gemma-4 support, build specifications and
training configs of every release up to 1.0.0, a collection-time length limit, a
per-task trace cap, and options to exclude or refuse long records. None of them
works with or is needed by the current code; they live in the parents of the
commits that removed them (`git log --diff-filter=D -- sft`). The release 1.0.0
specification lives at commit `53a88d8`, the only commit that reproduces it.
