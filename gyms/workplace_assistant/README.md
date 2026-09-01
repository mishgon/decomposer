# Workplace Assistant gym

This package owns Workplace Assistant dataset preparation, local execution,
MLSpace submission, experiment profiles, and conversion of successful
Decomposer rollouts into canonical SFT releases. The pinned `external/Gym`
submodule remains an upstream runtime dependency; this package does not use its
baseline job shell script.

## Prepare evaluation inputs

Preparation wraps the pinned NeMo-Gym dataset converter, writes the raw and
Decomposer-specific variants outside the repository, prepares lock-keyed Gym
and component environments, validates the selected models, and writes one
manifest per experiment and split.

```bash
# Prepare all registered train experiments.
.venv/bin/python -m gyms.workplace_assistant.prepare eval --split train

# Prepare one validation experiment, reusing an existing validated source file.
.venv/bin/python -m gyms.workplace_assistant.prepare eval \
  --split validation \
  --experiment gemma4-e2b-it-non-thinking \
  --reuse-source
```

The train and validation splits contain 1,255 and 545 tasks respectively.
`--split train` is the default.

## Run locally

The local runner does not submit an MLSpace job. It starts the selected model
services, agent service, and Gym servers on the current machine, performs one
Gym evaluation, validates the output, and stops every child process.

Every run declares its intent explicitly. `--purpose trace-generation` selects
the long teacher prompt and is available only for Decomposer experiments;
`--purpose evaluation` selects the short student prompt. Pass
`--prompt-profile teacher` or `--prompt-profile student` to override that
default for a Decomposer run. Explicit overrides use a distinct output and job
identity. Simple-agent evaluations are unaffected by prompt selection.

```bash
# Simple agent backed by one local policy vLLM.
.venv/bin/python -m gyms.workplace_assistant.run \
  --purpose evaluation \
  --experiment gemma4-e2b-it-non-thinking \
  --split train

# Decomposer backed by an external teacher and local subagent services.
OPENROUTER_API_KEY_DECOMPOSER=... \
HTTPS_PROXY=... \
.venv/bin/python -m gyms.workplace_assistant.run \
  --purpose trace-generation \
  --experiment glm-5-2-gemma4-26b-a4b-non-thinking \
  --split train

# Qwen3.6-35B-A3B teacher comparison on the complete validation split.
# The loopback proxy converts the deployment's native Qwen XML tool calls into
# Responses API function-call items; reasoning is enabled at the deployment's
# service-default effort, and the local worker remains Qwen3.5-4B.
.venv/bin/python -m gyms.workplace_assistant.run \
  --purpose trace-generation \
  --experiment qwen36-35b-a3b-teacher-qwen35-4b-non-thinking \
  --split validation \
  --num-repeats 3 \
  --concurrency 16 \
  --cuda-visible-devices 0

# One-task local smoke run. There is no automatic smoke pass.
.venv/bin/python -m gyms.workplace_assistant.run \
  --purpose evaluation \
  --experiment gemma4-e2b-it-non-thinking \
  --split validation \
  --limit 1

# Isolate a local run from shared MLSpace outputs and select physical GPU 2.
.venv/bin/python -m gyms.workplace_assistant.run \
  --purpose trace-generation \
  --experiment deepseek-v4-flash-0731-gemma4-e4b-thinking \
  --split train \
  --num-repeats 3 \
  --cuda-visible-devices 2 \
  --output-dir /path/to/local-results
```

Use `--dry` to print the complete service and Gym commands without starting
processes. `--output-dir` routes every result, log, status file, and completion
marker beneath an explicit directory. Partial outputs resume by default.
`--force` archives the previous attempt before starting fresh.

Every Workplace simple agent has a 100-step cap. Every Workplace Decomposer
manager and every spawned subagent independently has an exact 100-model-call
cap. Workplace collection uses Gym's `score_zero` rollout-failure policy.
Manager exhaustion and other agent-returned failure classes are therefore
stored in the main `rollouts.jsonl` as reward-0 attempts with structured
`_ng_rollout_error` diagnostics. Subagent exhaustion terminates that LangGraph
run with an error; the normal `wait` result delivers the error report to the
manager, which may recover by delegating again. The subagent recursion guard is
1,000 so it cannot preempt the exact model-call limiter.

An individual `/run` HTTP 500 is handled the same way: the collector writes a
valid placeholder response, copies the original request parameters, records
the bounded error detail, scores the rollout as zero, and continues with the
remaining tasks. These rows are completed attempts and are not retried on
resume. Transport failures, an unreachable service, or a dead process still
fail the job because they indicate a systemic problem rather than one bad
trace. Completion requires the exact task-by-repeat grid with no missing or
duplicate identities. Reward-0 rows are filtered before trace-shape validation
when preparing reward-1 SFT data, so placeholder responses cannot enter the
training set.

Call budgets are part of the artifact identity: simple runs use `calls100` and
Decomposer runs use `managercalls100-subagentcalls100`. This prevents an old
six-step or uncapped result from being silently reused.

## Submit MLSpace jobs

The launcher selects experiments from the same registry, skips completed and
active runs, stages a clean Git revision, and dispatches the shared local runner
inside each worker. Run it with the Python environment that provides `mls`:

```bash
MLSPY=/mnt/shared_ru.ml.SZ-5_000264/sukhorukov/.venv-mls/bin/python

$MLSPY -m gyms.workplace_assistant.run_eval \
  --purpose evaluation \
  --experiment gemma4-e2b-it-non-thinking \
  --split train \
  --author-name sukhorukov \
  --dry

# Submit the matched n=3 Qwen3.6 teacher comparison.
$MLSPY -m gyms.workplace_assistant.run_eval \
  --purpose trace-generation \
  --experiment qwen36-35b-a3b-teacher-qwen35-4b-non-thinking \
  --split validation \
  --num-repeats 3 \
  --concurrency 16 \
  --priority high \
  --author-name sukhorukov

$MLSPY -m gyms.workplace_assistant.run_eval \
  --purpose evaluation \
  --filter qwen35 \
  --split validation \
  --author-name sukhorukov
```

Run the Qwen3.5-4B simple-agent control on the complete validation split:

```bash
$MLSPY -m gyms.workplace_assistant.run_eval \
  --purpose evaluation \
  --experiment qwen35-4b-base-non-thinking \
  --split validation \
  --num-repeats 3 \
  --priority high \
  --author-name sukhorukov
```

Historical six-step results can be renamed once, without modifying rollout or
metric bytes. The command is a dry run by default and refuses to overwrite any
destination:

```bash
.venv/bin/python -m gyms.workplace_assistant.migrate_call_limit_artifacts
.venv/bin/python -m gyms.workplace_assistant.migrate_call_limit_artifacts --apply
```

The applied migration writes a checksum manifest beside the validation result
directories and normalizes the completed Qwen3.5-4B 100-step run to the
canonical `qwen35-4b-base-non-thinking-calls100-n3` identity.

Run the one-GPU E4B comparison on the complete validation split with three
rollouts per task:

```bash
$MLSPY -m gyms.workplace_assistant.run_eval \
  --purpose evaluation \
  --experiment gemma4-e4b-it-non-thinking-gemma4-e4b-thinking \
  --experiment gemma4-e4b-it-thinking \
  --split validation \
  --num-repeats 3 \
  --author-name sukhorukov
```

The Decomposer profile shares one E4B vLLM server between its non-thinking
manager requests and thinking subagent requests, so both comparison jobs use
one GPU each.

Run the untuned Qwen manager control on the complete validation split with the
same two-GPU topology and non-thinking base worker as the tuned Qwen manager:

```bash
$MLSPY -m gyms.workplace_assistant.run_eval \
  --purpose evaluation \
  --experiment qwen35-4b-base-non-thinking-qwen35-4b-non-thinking \
  --split validation \
  --num-repeats 3 \
  --priority medium \
  --author-name sukhorukov
```

Logical GPU 0 serves the untuned base manager under a dedicated model name;
logical GPU 1 serves an identical base Qwen3.5-4B checkpoint to subagents.

After the mixed Workplace plus partial-Toolathlon SFT run exports `final/`,
prepare and submit its student-prompt validation evaluation with the same base
Qwen3.5-4B non-thinking worker:

```bash
.venv/bin/python -m gyms.workplace_assistant.prepare eval \
  --split validation \
  --experiment qwen35-4b-sft-mixed-v1-partial-3983f605-327-32k-non-thinking-qwen35-4b-non-thinking \
  --reuse-source

$MLSPY -m gyms.workplace_assistant.run_eval \
  --purpose evaluation \
  --experiment qwen35-4b-sft-mixed-v1-partial-3983f605-327-32k-non-thinking-qwen35-4b-non-thinking \
  --split validation \
  --num-repeats 3 \
  --priority high \
  --author-name sukhorukov
```

The full terminal-Toolathlon snapshot checkpoint uses its own immutable
experiment identity and output directory:

```bash
.venv/bin/python -m gyms.workplace_assistant.prepare eval \
  --split validation \
  --experiment qwen35-4b-sft-mixed-v1-final-493c24c4-404-non-thinking-qwen35-4b-non-thinking \
  --reuse-source

$MLSPY -m gyms.workplace_assistant.run_eval \
  --purpose evaluation \
  --experiment qwen35-4b-sft-mixed-v1-final-493c24c4-404-non-thinking-qwen35-4b-non-thinking \
  --split validation \
  --num-repeats 3 \
  --priority high \
  --author-name sukhorukov
```

The filtered final-snapshot patience-1 best checkpoint has a separate immutable
experiment identity, so its evaluation does not overwrite the partial-snapshot
results:

```bash
.venv/bin/python -m gyms.workplace_assistant.prepare eval \
  --split validation \
  --experiment qwen35-4b-sft-mixed-v1-final-493c24c4-404-filtered-p1-s279-non-thinking-qwen35-4b-non-thinking \
  --reuse-source

$MLSPY -m gyms.workplace_assistant.run_eval \
  --purpose evaluation \
  --experiment qwen35-4b-sft-mixed-v1-final-493c24c4-404-filtered-p1-s279-non-thinking-qwen35-4b-non-thinking \
  --split validation \
  --num-repeats 3 \
  --priority high \
  --author-name sukhorukov
```

The patience-2 checkpoint trained on filtered Workplace, Toolathlon, and the
pinned 110-task GAIA2 training partition uses the same student prompt and base
non-thinking Qwen3.5-4B worker:

```bash
.venv/bin/python -m gyms.workplace_assistant.prepare eval \
  --split validation \
  --experiment qwen35-4b-sft-mixed-v2-493c24c4-gaia2-110-n3-filtered-p2-non-thinking-qwen35-4b-non-thinking \
  --reuse-source

$MLSPY -m gyms.workplace_assistant.run_eval \
  --purpose evaluation \
  --experiment qwen35-4b-sft-mixed-v2-493c24c4-gaia2-110-n3-filtered-p2-non-thinking-qwen35-4b-non-thinking \
  --split validation \
  --num-repeats 3 \
  --concurrency 8 \
  --priority high \
  --author-name sukhorukov
```

After the Workplace E4B SFT run has exported its merged `final/` checkpoint,
evaluate the tuned non-thinking manager and vanilla thinking E4B subagent on
dedicated GPUs:

```bash
uv run --group train python -m training.sft.vllm_compat \
  --source /mnt/shared_ru.ml.SZ-5_000264/sukhorukov/decomposer_artifacts/training/sft/jobs/gemma4-e4b-nonthinking-deepseek-e4b-v1-8k-full-4gpu/final \
  --output /mnt/shared_ru.ml.SZ-5_000264/sukhorukov/decomposer_artifacts/training/sft/jobs/gemma4-e4b-nonthinking-deepseek-e4b-v1-8k-full-4gpu/final-vllm

.venv/bin/python -m gyms.workplace_assistant.prepare eval \
  --split validation \
  --experiment gemma4-e4b-sft-deepseek-e4b-v1-8k-non-thinking-gemma4-e4b-thinking \
  --reuse-source

$MLSPY -m gyms.workplace_assistant.run_eval \
  --purpose evaluation \
  --experiment gemma4-e4b-sft-deepseek-e4b-v1-8k-non-thinking-gemma4-e4b-thinking \
  --split validation \
  --num-repeats 3 \
  --author-name sukhorukov
```

Logical GPU 0 serves the SFT manager and logical GPU 1 serves the vanilla E4B
thinking subagent. The full run writes 1,635 rollouts under the student-prompt
Decomposer evaluation root.

Dry runs do not stage code or submit jobs and redact credentials from printed
payloads. The live allocation's selected instance types are kept in
`experiments.py` as the single source of truth.

## Prepare SFT releases

The benchmark-neutral canonical schema, builder, and NeMo-Gym adapter remain in
`data.sft`. Workplace-specific build specifications and their public commands
live here:

```bash
uv run --group train python -m gyms.workplace_assistant.prepare sft \
  --dataset workplace-all-v3

uv run --group train python -m gyms.workplace_assistant.prepare sft \
  --dataset workplace-26b-nonthinking-v3

uv run --group train python -m gyms.workplace_assistant.prepare sft \
  --dataset workplace-deepseek-e4b-thinking-v1
```

Build the new non-thinking Gemma-4 E4B releases from the same normalized trace
pool with explicit prepared-token ceilings:

```bash
uv run --group train python -m gyms.workplace_assistant.prepare sft \
  --dataset workplace-deepseek-e4b-thinking-v2-8k

uv run --group train python -m gyms.workplace_assistant.prepare sft \
  --dataset workplace-deepseek-e4b-thinking-v2-32k
```

Both releases apply reward and trace-validity filtering, assign prompt groups to
train or validation once, remove teacher reasoning for tokenization, render the
prompt profile selected by the dataset specification with the Gemma-4 E4B
training template, and then apply the inclusive token ceiling. Consequently,
`v2-8k` is a strict subset of
`v2-32k`, and every shared record keeps the same ID and split. Raw rollout
artifacts are never removed.

Each retained row carries its prepared token and supervised-token counts. The
manifest pins the tokenizer revision and chat-template hashes and records raw
and retained length statistics plus every overlength exclusion. Releases are
immutable; preparation refuses to replace an existing version directory.

## Artifact layout

All generated data is outside the worktree under
`/mnt/shared_ru.ml.SZ-5_000264/sukhorukov/decomposer_artifacts`:

```text
evaluation/data/workplace_assistant/       source files and preparation manifests
evaluation/results/<split>/<run-name>/     teacher traces and simple-agent evaluations
evaluation/results/<split>/evaluation/     student-prompt Decomposer evaluations
datasets/sft/<dataset-id>/<version>/        immutable canonical SFT releases
venvs/gym/<lock-hash>/                      shared Gym CLI runtime
venvs/workplace-assistant/<lock-hash>/      shared Gym component runtimes
code/<git-commit>/                          immutable MLSpace code stages
```
