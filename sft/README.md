# Decomposer supervised fine-tuning

This workflow converts valid Decomposer rollouts into tool-calling
conversations and performs full-parameter SFT with TRL. The loss covers only
Decomposer outputs. Benchmark prompts, tool definitions, tool responses, and
subagent responses remain visible as context but receive label `-100`.

Preparation accepts only trajectories of the current Decomposer core, whose
manager uses the `new`, `fork`, `run`, and `wait` tools with the argument names
of the renamed core (2026-10-05): `new(agent_type_id)`, `fork(agent_id)` and
`run(agent_id, prompt)`. Traces of the retired `spawn_subagent`/`wait` core are
excluded with reason `excluded_legacy_tool_interface`; they are never converted.
Traces recorded before the rename (`subagent_type_id`, `subagent_id`) are
rejected as invalid tool calls. The canonical tool schemas come from
`sft/chat_tools.py`, which builds them from the live core.

A specification without `policy.subagent_types` keeps each record's native tool
schemas instead, so every gym keeps its own agent types. Each source must use
one schema; the manifest records its hash per source (`tool_schema_sha256`) and
lists the dataset's hashes (`content.tool_schema_sha256s`). The
NeMo Gym adapter (tau2, Workplace) keeps successful rollouts in which the
manager made a mistake the core answered: a call of an unknown tool, malformed
arguments, or an answer before collecting every run, which the core follows with
an injected user message (`EARLY_RESPONSE_ERROR`, `EMPTY_RESPONSE_ERROR`). The
trace must still be well formed: one task, every call answered, and a final
text answer. Current Toolathlon and WideSeek collections are read the same way
(see below); older Toolathlon imports and the GAIA2 adapter keep the strict checks.

## Legacy specs and configs

Every build specification in `sft/specs/`, `sft/gaia2/specs/` and
`sft/workplace_assistant/specs/`, and every training config in `sft/configs/`,
that predates the `new`/`fork`/`run`/`wait` core targets the retired
`spawn_subagent`/`wait` core. Their
datasets and checkpoints were built at `gaia2-eval` commit `2b1bda8`; rebuild
or validate them only from that commit. The current pipeline rejects their
source traces, and the `teacher` prompt profile now resolves to a different
prompt, so their manifests no longer validate against
`data.expected_system_prompt_profile: teacher`. The files stay in place as
experiment records, and the release walkthroughs below describe those
historical builds.

Specifications with a `tokenization:` block predate tokenizer-free releases
(2026-10-08) and no longer load; rebuild them only from the commit that built
them. Their releases still train: training ignores the token counts stored in
them.

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
uv run --group train python -m sft.workplace_assistant.prepare \
  --dataset workplace-all-v3 \
  --output-root /home/sukhorukov/decomposer_artifacts/datasets/sft
```

Build the 26B-A4B non-thinking source pair with:

```bash
uv run --group train python -m sft.workplace_assistant.prepare \
  --dataset workplace-26b-nonthinking-v3 \
  --output-root /home/sukhorukov/decomposer_artifacts/datasets/sft
```

Both specifications use exact reward `1.0`, prompt-fixed validation fraction
`0.1`, and seed `42`. All teacher variants of one prompt are assigned to the
same split. The builder requires a clean Git worktree and refuses to replace an
existing `<dataset-id>/<version>` directory.

New build specifications choose `policy.system_prompt_profile: student` or
`teacher`; a specification without a profile uses `teacher`, and the legacy
`policy.system_prompt: decomposer_default` keeps `student`. `teacher` is the
core's orchestration prompt (`decomposer.prompts.DECOMPOSER_SYSTEM_PROMPT`),
and `student` is the legacy one-line manager prompt
(`decomposer.prompt_profiles.DECOMPOSER_STUDENT_SYSTEM_PROMPT`). The builder
inserts that exact prompt before tokenization and records its profile and
SHA-256 in the immutable manifest. Training configs may set
`data.expected_system_prompt_profile` to fail if the selected release uses a
different prompt. Hidden teacher reasoning remains controlled separately by
`data.include_reasoning`.

Releases do not depend on a model: they hold messages and tool schemas, with no
token counts and no length limit. Training tokenizes every record with the
model's own chat template and decides what to do with long ones (see
[Length](#length)).

### Snapshots, releases and versioning

New build specifications (`spec_version: 4`) name every source by a snapshot
digest instead of a path. A snapshot is an immutable copy of exactly the files
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
Face dataset repository, and pinned by revision. Specifications with
`spec_version` 1 to 3 keep their paths and build as before.

### Toolathlon collections

The Toolathlon adapter reads a collection written by `sft/toolathlon_gym`
(`trace_format: toolathlon_langgraph_v1`), snapshotted with
`--adapter toolathlon_gym`. The snapshot takes every finished episode: its
`trace.json` and `runtime.json`, and its evaluation `result.json`. Episodes still running have no evaluation yet and are left out,
so a running collection can be snapshotted.

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

The WideSeek adapter (`adapter: wideseek`, `trace_format: wideseek_langgraph_v1`)
reads a collection written by `sft/wideseek` on main, snapshotted with
`--adapter wideseek`. Each attempt's `result.json` names the execution it scored.
The snapshot takes, per finished attempt, that result, the execution's
`trace.json`, and the manager's first logged model call, which holds the tools
and system prompt the manager saw. Restarted executions that no result names, and
the rest of the call log, are left out.

- **Selection.** `collector_qualifies` applies the collector's rule
  (`gyms/wideseek/metrics.qualifies` on main): a normal finish, a scored
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
  trace_format: wideseek_langgraph_v1
  selection:
    policy: collector_qualifies
    success_threshold: 0.9
  expected_native_rollouts: <finished attempts>
  expected_candidates: <finished attempts>
```

### Qwen3.8 four-gym release v3 for the unloop student

The v3 release (`v3-tau2-workplace-toolathlon-wideseek-20261008`) has the same
sources and rules as v2 below. Toolathlon and WideSeek were snapshotted again as
byte-for-byte copies at 19:41 and 19:44 UTC, and the tau2 and Workplace
snapshots are unchanged. Build it on Hertz-2 from a clean checkout:

```bash
.venv/bin/python -m sft.prepare \
  --spec sft/specs/decomposer_mixed_qwen38_qwen35_4b_unloop_nonthinking_v3.yaml \
  --output-root /mnt/share14T-2/sukhorukov/decomposer_artifacts/datasets/sft
```

| Source | Snapshot | Rollouts | Selected |
|---|---|---:|---:|
| tau2 | `170d2e3d…` | 4,843 | 2,948 |
| Workplace | `c36c909e…` | 1,255 | 988 |
| Toolathlon | `c9ea4cfe…` | 1,095 | 214 |
| WideSeek | `c160625c…` | 253 | 70 |

That gives 4,220 records (3,797 train, 423 validation) with fingerprint
`50a73343…1fdb8df4`. Every v2 record is in v3 unchanged, plus 4 Toolathlon and 6
WideSeek records. The added task groups shift the per-category validation
quotas, so 15 records changed sides (13 of them tau2). With the Qwen3.5 template
the release has 38.4M tokens, 8.86M of them supervised.

| Gym | Median tokens | Max | Supervised share | Cut at 16K | Kept at 16K | Cut at 32K | Kept at 32K |
|---|---:|---:|---:|---:|---:|---:|---:|
| tau2 | 6.8K | 65.5K | 54% | 93 | 97% | 4 | 99% |
| Workplace | 6.0K | 30.7K | 13% | 6 | 99% | 0 | 100% |
| Toolathlon | 29.3K | 64.3K | 27% | 191 | 44% | 84 | 80% |
| WideSeek | 20.5K | 54.8K | 6% | 45 | 55% | 17 | 88% |
| All | | | | 335 | 80% | 105 | 93% |

"Kept" is the share of supervised tokens left after cutting from the end; no
record is left without any. A two-step LoRA smoke on the 4 longest train records
with the training defaults (16K) trained and exported
(`/mnt/share14T-2/sukhorukov/decomposer_artifacts/training/sft/smokes/mixed-v3-20261008`).
The Toolathlon snapshot holds the proxy key, like v2's, and must not be
published.

### Qwen3.8 four-gym release v2 for the unloop student

The v2 release (`v2-tau2-workplace-toolathlon-wideseek-20261008`) adds the
Toolathlon and WideSeek coverage collections, as finished on 2026-10-08, to the
Workplace traces of v1 and the tau2 rerun in which only subagents get the domain
policy (FP8 teacher). Every source has the Qwen3.8-Flash non-thinking manager
and Qwen3.5-4B-unlooped thinking subagents, and the records keep parallel tool
calls as emitted. Build it on Hertz-2 from a clean checkout:

```bash
.venv/bin/python -m sft.prepare \
  --spec sft/specs/decomposer_mixed_qwen38_qwen35_4b_unloop_nonthinking_v2.yaml \
  --output-root /mnt/share14T-2/sukhorukov/decomposer_artifacts/datasets/sft
```

| Source | Snapshot | Rollouts | Selected | Rule |
|---|---|---:|---:|---|
| tau2 | `170d2e3d…` | 4,843 | 2,948 | reward 1 |
| Workplace | `c36c909e…` | 1,255 | 988 | reward 1 |
| Toolathlon | `35ea4233…` | 1,064 | 210 | collector: pass or check fraction > 0.9 |
| WideSeek | `50cc2286…` | 242 | 64 | collector: score 1 or > 0.9 |

That gives 4,210 records (3,789 train, 421 validation) with fingerprint
`cb774a4e…f62573`. Each gym keeps its own tool schema. With the Qwen3.5 template
the release has 38.1M tokens, 8.7M of them supervised; Toolathlon contributes 5%
of the records but 26% of the supervised tokens.

| Gym | Median tokens | Over 16K | Over 32K | Max |
|---|---:|---:|---:|---:|
| tau2 | 6.8K | 93 | 4 | 65.5K |
| Workplace | 6.0K | 6 | 0 | 30.7K |
| Toolathlon | 29.1K | 187 | 81 | 64.3K |
| WideSeek | 20.5K | 42 | 16 | 54.8K |

Cutting from the end keeps this share of the supervised tokens; no record is
left without any:

| Gym | Cut at 16K | Kept at 16K | Cut at 32K | Kept at 32K |
|---|---:|---:|---:|---:|
| tau2 | 93 | 97% | 4 | 99% |
| Workplace | 6 | 99% | 0 | 100% |
| Toolathlon | 187 | 44% | 81 | 80% |
| WideSeek | 42 | 55% | 16 | 88% |
| All | 328 | 81% | 101 | 94% |

A two-step LoRA smoke on the 4 longest train records, each cut to 32,768 tokens,
trained and exported
(`/mnt/share14T-2/sukhorukov/decomposer_artifacts/training/sft/smokes/mixed-v2-20261008`).

The Toolathlon traces contain the LLM proxy master key: the task containers
exposed it, and subagents listed their environment while looking for API
tokens. The snapshot keeps the traces as collected, so it must not be
published.

### Qwen3.8 tau2 + Workplace release for the unloop student (current core)

The first release of the current core trains the non-thinking Qwen3.5-4B unloop
checkpoint on the 2026-10-06 Qwen3.8-Flash non-thinking teacher traces: tau2
`decomposer_broad_v1` and the Workplace train split, one rollout per task, with
Qwen3.5-4B-unlooped thinking subagents. Build it on Hertz-2 from a clean checkout
of the committed specification:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m sft.prepare \
  --spec sft/specs/decomposer_mixed_qwen38_qwen35_4b_unloop_nonthinking_v1_32k.yaml \
  --output-root /mnt/share14T-2/sukhorukov/decomposer_artifacts/datasets/sft
```

Of 6,098 rollouts, 3,976 have reward `1`. Each keeps its gym's own tool schema
and the teacher prompt, together with the mistakes the core answered (for
example, 36 calls of non-Decomposer tools and 7 early answers). The 32K token
limit excludes 3 tau2 traces, producing 3,973 records (2,985 tau2, 988
Workplace): 3,578 train and 395 validation. Of 3,834 prompt groups, 135 tau2
prompts appear in both a domain and its `_dsh` implementation; each such pair
stays on one side of the split. The median record has 7.3K tokens, and 20% of
the tokens are supervised.

### Qwen Workplace + Toolathlon final all-reward release

The final mixed Qwen release intentionally keeps both full- and non-full-reward
traces. Its terminal Toolathlon run contains 404 completed episodes and 99
failed episodes with no trace/result pair. Import only that run into a
hash-namespaced immutable location:

```bash
.venv/bin/python -m sft.import_toolathlon \
  --archive /home/sukhorukov/traces_full.tar.gz \
  --archive-prefix matrosov/decomposer-qwen/artifacts/gyms/toolathlon_gym \
  --run-id 20260826T122838Z-84ae95f3 \
  --expected-sha256 493c24c4f8853230c0b7557b905ce2027138522ee6da06d5d2e71d0cade94753 \
  --output-root /home/sukhorukov/decomposer_artifacts/evaluation/data/toolathlon_gym/imports/snapshots/493c24c4
```

Build the immutable mixed release after committing the preparation code and
specification:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m sft.prepare \
  --spec sft/specs/decomposer_mixed_deepseek_qwen35_4b_nonthinking_v1_32k.yaml \
  --output-root /home/sukhorukov/decomposer_artifacts/datasets/sft \
  --source toolathlon-deepseek-v4-flash-0731-qwen35-4b-nonthinking-n1=/home/sukhorukov/decomposer_artifacts/evaluation/data/toolathlon_gym/imports/snapshots/493c24c4/20260826T122838Z-84ae95f3
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
/home/sukhorukov/.venv-mls/bin/python \
  -m sft.run_train_jobs \
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
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m sft.prepare \
  --spec sft/specs/decomposer_mixed_deepseek_qwen35_4b_nonthinking_v1_filtered_pass_quality_32k.yaml \
  --output-root /home/sukhorukov/decomposer_artifacts/datasets/sft \
  --source toolathlon-deepseek-v4-flash-0731-qwen35-4b-nonthinking-n1=/home/sukhorukov/decomposer_artifacts/evaluation/data/toolathlon_gym/imports/snapshots/493c24c4/20260826T122838Z-84ae95f3
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
/home/sukhorukov/.venv-mls/bin/python \
  -m sft.run_train_jobs \
  --filter qwen35-4b-nonthinking-mixed-v1-final-493c24c4-404-filtered-pass-qgt90-32k-full-4gpu \
  --priority high
```

This run uses the same optimization settings as the all-rewards comparison:
32K inputs, four GPUs, five epochs at most, per-epoch evaluation, and
early-stopping patience two. It has an independent output directory and does
not resume or overwrite the all-rewards run.

#### Filtered GAIA2 n=3 extension

The GAIA2 extension adds only binary reward-1 traces from logical rollouts
1–3 on the immutable `execution-110-50-v1` training partition. Universes 25,
26, and 28 remain isolated as test holdout. A checked-in assignment manifest
preserves all 1,251 Workplace/Toolathlon task memberships and pins all 110
GAIA2 task groups now (99 train, 11 validation), including tasks without a
correct n=3 trace, so later logical rollouts 4–10 cannot move a task between
splits.

Build the release from a clean committed checkout:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m sft.prepare \
  --spec sft/specs/decomposer_mixed_deepseek_qwen35_4b_nonthinking_v2_gaia2_execution_110_n3_filtered_32k.yaml \
  --output-root /home/sukhorukov/decomposer_artifacts/datasets/sft
```

The expected source grid has 1,989 candidates. Filtering retains 1,347
structurally valid correct/quality-selected traces before tokenization,
including 96 GAIA2 reward-1 traces from 330 training-partition attempts. The
32K limit excludes 13 traces (12 Toolathlon and one GAIA2), producing 1,334
records: 1,209 train and 125 validation. GAIA2 rewards are strictly validated
as binary `0` or `1`.

Submit the base Qwen3.5-4B run on four GPUs at high priority:

```bash
/home/sukhorukov/.venv-mls/bin/python \
  -m sft.run_train_jobs \
  --filter qwen35-4b-nonthinking-mixed-v2-493c24c4-gaia2-110-n3-filtered-32k-full-4gpu \
  --priority high
```

The run has a five-epoch ceiling and early-stopping patience two. It uses the
student prompt, removes teacher reasoning at preprocessing time, and has a
distinct immutable dataset and checkpoint path.

#### Filtered GAIA2 n=7 teacher-prompt extension

The n=7 release combines the immutable logical 1–3 evaluation source with a
compact snapshot of completed trace rounds 4–7. It intentionally excludes
round 8 and later rounds so future releases can extend the same pinned task
split without changing this dataset. The 770-attempt GAIA grid contains 215
binary reward-1 traces; structural validation retains 210 and the 32K limit
retains 208.

The complete mixed source grid has 2,429 candidates. Reward, quality, and
structural filtering retains 1,461 before tokenization. The teacher prompt
makes 20 records exceed 32K, producing 1,441 records: 1,314 train and 127
validation. The final environment counts are 953 Workplace, 280 Toolathlon,
and 208 GAIA2 records.

Build the release with the explicit teacher prompt from a clean checkout:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m sft.prepare \
  --spec sft/specs/decomposer_mixed_deepseek_qwen35_4b_nonthinking_v3_gaia2_execution_110_n7_teacher_prompt_filtered_32k.yaml \
  --output-root /home/sukhorukov/decomposer_artifacts/datasets/sft
```

Submit the stable SDPA, global-batch-4 run from the base Qwen3.5-4B checkpoint:

```bash
/home/sukhorukov/.venv-mls/bin/python \
  -m sft.run_train_jobs \
  --filter qwen35-4b-nonthinking-mixed-v3-493c24c4-gaia2-110-n7-teacher-prompt-filtered-32k-full-4gpu \
  --priority high
```

The training config requires the dataset's teacher profile and matching prompt
hash, removes hidden teacher reasoning, and uses early-stopping patience two.

#### Accelerated Toolathlon-only ablation

The checksum-pinned Toolathlon-only candidate view contains 252 training and
28 validation records inherited byte-for-byte from the mixed v3 release. Its
eight-epoch ceiling approximately matches the supervised-token exposure of
the five-epoch mixed run. It starts independently from the pinned base
Qwen3.5-4B checkpoint and retains early-stopping patience two.

Submit the four-H100 run with the pinned HF FlashAttention-2 and FLA/causal
runtime at high priority:

```bash
/home/sukhorukov/.venv-mls/bin/python \
  -m sft.run_train_jobs \
  --filter qwen35-4b-nonthinking-toolathlon-only-v1-493c24c4-teacher-prompt-filtered-32k-hf-fa2-fla-b8-e8-full-4gpu \
  --priority high
```

#### Accelerated GAIA2 execution-only ablation

The GAIA2 execution-only release combines ten logical rollouts for each of
the 110 training scenarios. Of 1,100 attempts, 316 have exact binary reward
one, strict structural validation retains 306, and the 32K token limit retains
303. Its pinned task-level split assigns 99 scenarios to train and 11 to
validation, yielding 273 train and 30 validation records with no task overlap.
The validation tasks are selected deterministically to land on the exact
10-percent record target while keeping every task wholly in one partition.

Build the immutable teacher-prompt release from a clean checkout:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m sft.prepare \
  --spec sft/gaia2/specs/decomposer_gaia2_execution_deepseek_qwen35_4b_nonthinking_v1_110_n10_teacher_prompt_r1_balanced_32k.yaml \
  --output-root /home/sukhorukov/decomposer_artifacts/datasets/sft
```

Submit the four-H100 accelerated run at high priority:

```bash
/home/sukhorukov/.venv-mls/bin/python \
  -m sft.run_train_jobs \
  --filter qwen35-4b-nonthinking-gaia2-execution-only-v1-110-n10-teacher-prompt-r1-balanced-32k-hf-fa2-fla-b8-e24-full-4gpu \
  --priority high
```

The 24-epoch ceiling approximately matches the supervised-token exposure of
the mixed run. Validation and checkpointing happen after every epoch; early
stopping uses patience two and the final export reloads the checkpoint with
the lowest validation loss.

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
.venv/bin/python -m sft.import_toolathlon \
  --archive /home/sukhorukov/traces.tar.gz \
  --archive-prefix matrosov/decomposer-qwen/artifacts/gyms/toolathlon_gym \
  --run-id 20260826T122838Z-84ae95f3 \
  --expected-sha256 3983f60540f1887befbc2654db8a8d7b169c397a88d62d1017625cd480386f5f \
  --output-root /home/sukhorukov/decomposer_artifacts/evaluation/data/toolathlon_gym/imports/snapshots/3983f605
```

After committing the preparation implementation, build the release:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m sft.prepare \
  --spec sft/specs/decomposer_mixed_deepseek_qwen35_4b_nonthinking_v1_partial_3983f605_327_32k.yaml \
  --output-root /home/sukhorukov/decomposer_artifacts/datasets/sft \
  --source toolathlon-deepseek-v4-flash-0731-qwen35-4b-nonthinking-n1=/home/sukhorukov/decomposer_artifacts/evaluation/data/toolathlon_gym/imports/snapshots/3983f605/20260826T122838Z-84ae95f3
```

This build has 1,532 retained traces: 1,379 train and 153 validation. The
train split contains 1,118 Workplace and 261 Toolathlon traces; validation
contains 125 and 28 respectively. Preparation excludes 36 malformed traces
and 14 Toolathlon traces over 32K without truncating any record. The longest
retained train trace has 32,394 Qwen tokens.

Run the longest-trace, one-step smoke at high priority before the full job:

```bash
/home/sukhorukov/.venv-mls/bin/python \
  -m sft.run_train_jobs \
  --sanity-check \
  --filter qwen35-4b-nonthinking-mixed-v1-partial-3983f605-327-32k-smoke-4gpu \
  --priority high
```

Submit the full five-epoch run only after that smoke succeeds:

```bash
/home/sukhorukov/.venv-mls/bin/python \
  -m sft.run_train_jobs \
  --filter qwen35-4b-nonthinking-mixed-v1-partial-3983f605-327-32k-full-4gpu \
  --priority high
```

Canonical records keep every assistant message as the teacher emitted it. A
message with several tool calls stays one message, and its results follow in
the order the harness returned them, so the student learns the same turns that
OPD and RL later sample. Calls the harness refused without executing them keep
the core's refusal text as their result, like other mistakes the core answered:
a `wait` sharing its message with any other call, a `fork` and `run` of the same
subagent, and several `run` calls of the same subagent
(`PARALLEL_WAIT_CALL_ERROR`, `PARALLEL_FORK_RUN_CALL_ERROR` and
`PARALLEL_RUN_CALL_ERROR` in `decomposer.prompts`). Releases built before
`nemo_gym` adapter 8, `toolathlon_gym` adapter 9 and `gaia2` adapter 5 split
such messages into one call per turn.

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
uv run --group train python -m sft.workplace_assistant.prepare \
  --dataset workplace-deepseek-e4b-thinking-v2-8k
uv run --group train python -m sft.workplace_assistant.prepare \
  --dataset workplace-deepseek-e4b-thinking-v2-32k
```

The default v2 run remains 8K. Its smoke and full MLSpace experiments are:

```bash
uv run --with-requirements sft/requirements-mlspace.txt \
  python -m sft.run_train_jobs --sanity-check \
  --filter gemma4-e4b-nonthinking-deepseek-e4b-v2-8k-smoke-4gpu

uv run --with-requirements sft/requirements-mlspace.txt \
  python -m sft.run_train_jobs \
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

LoRA for the Qwen3.5-4B unloop student on the v3 four-gym release, on Hertz-2:

```bash
cd ~/decomposer_sft
nohup sft/train_qwen35_h200.sh \
  sft/configs/qwen35_4b_unloop_nonthinking_mixed_v3_lora_16k_4gpu.yaml 4,5,6,7 \
  > ~/qwen35-v3-lora-16k.log 2>&1 &
```

The 32K config (`…_lora_32k_4gpu.yaml`) differs only in `max_length` and the
output directory. Both use the H200 benchmark recipe with LoRA `r: 32`,
`alpha: 64` and `dropout: 0.05`. They run 3 epochs, evaluating and saving after
each one, and keep the checkpoint with the lowest validation loss; training
stops after an epoch that does not improve it. Both log to ClearML.

Validation loss is reported for the whole split as `eval_all_loss`, which picks
the checkpoint, and per gym as `eval_<gym>_loss` when the split mixes gyms.
ClearML plots each one separately and also all of them together in
`eval/loss_by_dataset`. A config's `metric_for_best_model: eval_loss` means
`eval_all_loss`.
`sft/train_qwen35_h200.sh` sets the environment from
`docs/sft_qwen35_h200_benchmark.md` and refuses any listed GPU that holds memory.

Four-GPU E2B with fused CE and global batch eight:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run --group train \
  torchrun --standalone --nproc-per-node=4 \
  -m sft.train \
  --config sft/configs/gemma4_e2b_nonthinking_4gpu_liger_workplace_26b_v3.yaml
```

Four-GPU E4B with fused CE and global batch four:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run --group train \
  torchrun --standalone --nproc-per-node=4 \
  -m sft.train \
  --config sft/configs/gemma4_e4b_nonthinking_4gpu_liger_workplace_26b_v3.yaml
```

For the 8K DeepSeek/E4B v1 run, first launch the one-step smoke experiment into
the separate sanity artifact root:

```bash
uv run --with-requirements sft/requirements-mlspace.txt \
  python -m sft.run_train_jobs \
  --sanity-check \
  --filter gemma4-e4b-nonthinking-deepseek-e4b-v1-8k-smoke-4gpu
```

After the smoke run succeeds, launch the full experiment:

```bash
uv run --with-requirements sft/requirements-mlspace.txt \
  python -m sft.run_train_jobs \
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
sft/configs/gemma4_e2b_nonthinking_4gpu_liger_workplace_26b_v3.yaml
sft/configs/gemma4_e4b_nonthinking_4gpu_liger_workplace_26b_v3.yaml
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
  -m sft.train \
  --config sft/configs/gemma4_e2b_nonthinking_4gpu_liger_workplace_26b_v3.yaml \
  --include-reasoning \
  --max-length 65536 \
  --output-dir /home/sukhorukov/decomposer_artifacts/training/sft/checkpoints/gemma4-e2b-thinking
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

Full SFT keeps fp32 master weights (fp32 DDP weights, or FSDP's upcast), so the
export rewrites `final/` in bf16, with a bf16 `config.json`, on both paths.
Evaluation serves bf16 anyway (`--dtype bfloat16`), and resuming uses the Trainer
checkpoints, which are unchanged.

Gemma-4 KV-shared layers intentionally omit unused `k_norm` weights after SFT,
but vLLM 0.24 still requires those parameters during checkpoint validation.
Create a zero-copy compatibility view for vLLM without changing `final/`:

```bash
uv run --group train python -m sft.vllm_compat \
  --source /path/to/run/final \
  --output /path/to/run/final-vllm
```

The derived directory symlinks the original model tensor and adds identity
`k_norm` tensors only for the KV-shared layers. Transformers consumers should
continue to use `final/`; vLLM consumers should use `final-vllm/`.

### Length

`training.max_length` sets the longest sequence a run trains on; it defaults to
16,384 (`max_length: null` means no limit). A record over it is cut from the end
by default: TRL keeps its first `max_length` tokens, which hold the system
prompt, the tools and the task, and drops a record left with no supervised
token. A config can instead drop such records (`data.exclude_overlength: true`)
or refuse them (`data.error_on_truncation: true`). `training_summary.json` lists
dropped records under `overlength_exclusions` and cut ones under
`overlength_truncations`, each with the supervised tokens it keeps.

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
  dropout: 0.0
```

Dropout is 0 for single-epoch runs. The four-gym release
(`v2-tau2-workplace-toolathlon-wideseek-20261008`) has about 38M tokens, roughly
9M of them supervised. The LoRA config trains 1 epoch, so every record is seen
once. In a single pass there is little to overfit, and dropout mostly adds noise
and costs about 4% of throughput (`docs/sft_qwen35_h200_benchmark.md`). For
multi-epoch runs, raise it (for example to 0.05) if validation loss starts rising.

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
  -m sft.train \
  --config sft/configs/gemma4_e2b_smoke.yaml
```

## MLSpace jobs

The stable artifact path is a symlink into shared NFS:

```text
/home/jovyan/decomposer-artifacts
  -> /home/sukhorukov/decomposer_artifacts
```

Training outputs live under `sft/jobs/`, sanity outputs under
`sft/jobs_sanity/`, staged clean Git snapshots under `code/`, and
shared training environments under `venvs/sft/<uv-lock-hash>/`. The launcher
syncs only the base and `train` dependency group. It does not require `nvcc`;
Gemma-4 Liger experiments use the pinned Python wheel and Triton JIT.

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
  --filter gemma4-e2b-nonthinking --resume-latest
```

Job descriptions contain only the `#sukhorukov` tag. Real submission refuses a
dirty tracked worktree, stages the committed source by Git hash, skips completed
runs via `training_summary.json`, and deduplicates Pending/Running jobs by their
normalized descriptions. Combined `torchrun` output is also persisted as
`console.log` in each run directory with pipefail-safe exit propagation; save an
MLSpace API log snapshot as `mlspace.log` after the job reaches terminal state.
