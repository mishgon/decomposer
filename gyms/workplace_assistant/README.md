# Workplace Assistant gym

This package owns Workplace Assistant dataset preparation, local execution,
MLSpace submission, and experiment profiles. Turning successful Decomposer
rollouts into canonical SFT releases lives in `sft/workplace_assistant`. The pinned `external/Gym`
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

Every run declares its intent explicitly with `--purpose trace-generation` or
`--purpose evaluation`. Both use `decomposer.prompts.DECOMPOSER_SYSTEM_PROMPT`.
Trace generation is available only for Decomposer experiments.

The matched text-default profiles are:

- `gemma4-26b-a4b-thinking-gemma4-e4b-thinking-text-defaults` and its simple
  control `gemma4-e4b-thinking-simple-text-defaults`, using Gemma's
  `temperature=1.0`, `top_p=0.95`, `top_k=64` thinking preset.
- `qwen36-35b-a3b-non-thinking-teacher-qwen35-4b-non-thinking-text-defaults`
  and its simple control
  `qwen35-4b-non-thinking-simple-general-text-defaults`, using Qwen's general
  instruct preset. The remote manager is reached only through
  `LLM_PROXY_URL`/`LLM_PROXY_MASTER_KEY`; both Qwen chat templates explicitly
  disable thinking.
- `qwen36-35b-a3b-thinking-teacher-qwen35-4b-non-thinking-text-defaults`
  keeps the same teacher prompt, local Qwen3.5-4B non-thinking worker, 128K
  context, and runtime limits, but explicitly enables Qwen3.6 thinking through
  the LLM proxy. Output reasoning is captured, but the upstream Responses
  deployment discards reasoning items replayed by the harness. This profile is
  capture-only and is not a clean comparison with full-replay local Gemma or
  OpenRouter DeepSeek.
- `qwen38-flash-non-thinking-teacher-qwen35-4b-unlooped-thinking` needs no GPU and
  takes both roles from the `src/decomposer/models.py` presets
  (`gyms/workplace_assistant/model_presets.py`). The manager is `lmrouter/qwen_3_8_flash_next_non_thinking`
  behind the loopback manager proxy, whose extra body is that preset with
  `reasoning.effort: "none"` (the proxy ignores `chat_template_kwargs` on the
  Responses API). The subagents are `lmrouter/qwen_3_5_4b_unlooped_thinking`
  (`subagent_backend="preset"`): the `qwen35_4b_unlooped_thinking` graph is the
  preset's own `create_model()` client, thinking, with reasoning kept across turns and
  no output cap, reaching the proxy on its own endpoint with `LLM_PROXY_MASTER_KEY`.
  No subagent proxy starts.
- `qwen38-flash-thinking-low-teacher-qwen35-4b-unlooped-non-thinking` needs no
  GPU. The manager is Qwen3.8-Flash-Next through the LLM proxy, thinking at
  `reasoning.effort` low, with the teacher prompt and no presence penalty. The
  subagents are `Qwen/Qwen3.5-4B-unlooped` non-thinking through the same proxy, via a
  loopback `gyms.remote_model_proxy` on port 8025 (`subagent_backend="llm_proxy"`).
  They use that model's recommended non-thinking sampling, 0.7/0.8/20 with no
  penalties, capped at 8192 tokens instead of its 2048, which cut legitimate
  full-record reports in the smoke (`subagent_sampling`, passed to the graph as
  `DECOMPOSER_SUBAGENT_SAMPLING_JSON`). Unlike Qwen3.6, Qwen3.8 on the proxy renders
  replayed reasoning items (probe on 2026-09-25: 83 → 158 input tokens), so its
  identity records `capture_replay_upstream_verified_v1`. Before starting, the run
  checks that the proxy lists both models.
- `deepseek-v4-flash-0731-teacher-gemma4-26b-a4b-non-thinking`: OpenRouter
  DeepSeek with high reasoning and the teacher prompt, plus one local
  non-thinking Gemma-4-26B-A4B worker.
- `gemma4-26b-a4b-thinking-teacher-gemma4-26b-a4b-non-thinking-text-defaults`:
  a local thinking Gemma-4-26B-A4B teacher and non-thinking worker using the
  teacher prompt. The actors share one checkpoint, vLLM process, endpoint, and
  GPU while retaining distinct per-request thinking modes.

All these profiles use 128K context and provider-controlled output length.
Workplace simple agents have 100 model calls; each Decomposer manager and each
spawned subagent has its own 100-model-call limit. The runner omits
`max_output_tokens` unless an experiment explicitly requests a bounded-output
ablation.

The simple-agent registry also contains matched thinking and non-thinking
Gemma-4 pairs for E2B, E4B, dense 31B, and 26B-A4B:
`gemma4-{e2b,e4b,31b,26b-a4b}-it-{non-thinking,thinking}`. They use the same
128K context, provider-controlled output length, Gemma text sampling defaults,
and 100-call budget. The dense 31B pair targets one 140 GB H200. Install its
pinned snapshot with:

```bash
HF_HOME=/home/sukhorukov/.cache/huggingface \
  .venv/bin/hf download google/gemma-4-31B-it \
  --revision 842da3794eaa0b77d5f08bae87a17459d91ff475 \
  --max-workers 8
```

Local thinking profiles capture structured reasoning, save it separately from
visible text, replay it through `ChatVLLM`, and set Gemma's
`preserve_thinking=true` template option. Their run identity records
`capture_replay_v2_template_preserved`. Every Qwen3.6 profile using
`LLM_PROXY_URL` is explicitly marked
`capture_only_upstream_no_replay_v1`: the harness sends prior reasoning, but
that upstream service discards it.

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
# service-default effort, and the local worker remains Qwen3.5-4B. evals runs the
# gym runner and then writes eval_metrics.json and the checksum-pinned comparison.json.
.venv/bin/python -m evals.workplace_assistant.run \
  --purpose trace-generation \
  --experiment qwen36-35b-a3b-teacher-qwen35-4b-non-thinking \
  --split validation \
  --num-repeats 3 \
  --concurrency 16 \
  --cuda-visible-devices 0

# Matched explicit-thinking Qwen3.6 manager comparison.
.venv/bin/python -m evals.workplace_assistant.run \
  --purpose evaluation \
  --experiment qwen36-35b-a3b-thinking-teacher-qwen35-4b-non-thinking-text-defaults \
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
Completion and resume validate the full runtime configuration, including the
output-length policy. An existing 32K-capped result cannot be reused as an
uncapped run; `--force` archives the previous attempt before starting fresh.

Use `--port-offset` to run several local evaluations on the same host. The
offset is added to every coordinated loopback endpoint: model servers, manager
proxy, LangGraph, the Gym head server, and the complete Gym component range.
Offset zero is the default and preserves the historical ports and artifact
names. A nonzero default output and cache identity ends in
`-port-offset-N`; resuming through an explicit `--output-dir` requires the same
offset.

```bash
# Qwen simple agent: vLLM 8000 and Gym 11000-11999.
.venv/bin/python -m gyms.workplace_assistant.run \
  --experiment qwen35-4b-non-thinking-simple-general-text-defaults \
  --purpose evaluation --split validation --port-offset 0

# Gemma simple agent: ports shifted by 12000.
.venv/bin/python -m gyms.workplace_assistant.run \
  --experiment gemma4-e4b-thinking-simple-text-defaults \
  --purpose evaluation --split validation --port-offset 12000

# Gemma Decomposer: ports shifted by 24000.
.venv/bin/python -m gyms.workplace_assistant.run \
  --experiment gemma4-26b-a4b-thinking-gemma4-e4b-thinking-text-defaults \
  --purpose evaluation --split validation --port-offset 24000
```

Offsets may be any non-negative value whose resulting ports do not exceed
65535. A stride of 12000 is recommended because it keeps the current
1000-port Gym component ranges disjoint. MLSpace launchers intentionally do
not expose this local-host option and continue to use offset zero.

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
Decomposer runs use `managercalls100-subagentcalls100`. The stored runtime
configuration provides the remaining identity checks, including context,
sampling, thinking mode, and output-length policy.

## Subagent server state

The runner serves the subagent graphs in `subagents/langgraph.json` through
`langgraph_server.py`, which calls `run_server(disable_persistence=True)` from
`<run>/langgraph/`. Under `langgraph dev` the store in `subagents/.langgraph_api` grew
by 0.5-2.3 GB per run and, reloaded by the next run, slowed `threads.get_history` until
runs timed out; the CLI offers no way to disable that persistence.

## Submit MLSpace jobs

The launcher selects experiments from the same registry, skips completed and
active runs, stages a clean Git revision, and dispatches the shared local runner
inside each worker. Run it with the Python environment that provides `mls`:

```bash
MLSPY=/home/sukhorukov/.venv-mls/bin/python

$MLSPY -m gyms.workplace_assistant.run_eval \
  --purpose evaluation \
  --experiment gemma4-e2b-it-non-thinking \
  --split train \
  --author-name sukhorukov \
  --dry

# Submit the matched n=3 Qwen3.6 teacher comparison; evals.workplace_assistant.submit
# takes the same flags, but each job also computes metrics and the comparison.
$MLSPY -m evals.workplace_assistant.submit \
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
.venv/bin/python -m evals.workplace_assistant.migrate_call_limit_artifacts
.venv/bin/python -m evals.workplace_assistant.migrate_call_limit_artifacts --apply
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
uv run --group train python -m sft.vllm_compat \
  --source /home/sukhorukov/decomposer_artifacts/training/sft/jobs/gemma4-e4b-nonthinking-deepseek-e4b-v1-8k-full-4gpu/final \
  --output /home/sukhorukov/decomposer_artifacts/training/sft/jobs/gemma4-e4b-nonthinking-deepseek-e4b-v1-8k-full-4gpu/final-vllm

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

## SFT releases

The named Workplace SFT releases and their specs live in
[`sft/workplace_assistant`](../../sft/workplace_assistant/README.md).

## Artifact layout

All generated data is outside the worktree under
`/home/sukhorukov/decomposer_artifacts`:

```text
evaluation/data/workplace_assistant/       source files and preparation manifests
evaluation/results/<split>/<run-name>/     teacher traces and simple-agent evaluations
evaluation/results/<split>/evaluation/     student-prompt Decomposer evaluations
datasets/sft/<dataset-id>/<version>/        immutable canonical SFT releases
venvs/gym/<lock-hash>/                      shared Gym CLI runtime
venvs/workplace-assistant/<lock-hash>/      shared Gym component runtimes
code/<git-commit>/                          immutable MLSpace code stages
```

## Known issue: scoring from reported calls

The Workplace resources server (`external/Gym/resources_servers/workplace_assistant`)
scores the tool calls the Decomposer reports, which it rebuilds from subagent
histories. A subagent that dies usually leaves no history, so its calls are not
scored although they ran. The tau2 gym scores the server's own call log instead
(`gyms/tau2_gym/README.md`, "How scoring works"); Workplace needs the same change in
the Gym fork.
