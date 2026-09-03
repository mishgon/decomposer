# Gaia2 evaluation

This package owns Gaia2 evaluation data preparation, the Decomposer external
agent adapter, local execution, and MLSpace submission. Evaluation supports the
`execution`, `search`, and `ambiguity` domains. Teacher trace generation remains
restricted to the immutable execution train partition described below.

The Gaia runtime is pinned through `external/gaia2` at commit
`993389ceb9e789a3965576c4a31a4d74f3cbba5b`. Preparation materializes a
clean, immutable checkout under Decomposer artifacts. Every preparation and
run manifest records both repository commits.

## Prepare data and runtimes

```bash
.venv/bin/python -m gyms.gaia2.prepare eval \
  --split validation \
  --domain execution
```

Use `--domain search` or `--domain ambiguity` to prepare another supported
domain. Each non-execution domain is stored in an isolated dataset-revision
subtree, so preparation cannot change the existing execution source or results.

Preparation loads revision
`78ea3bdbdeec2bdcd6afa5420915d8a22f23ed99` of
`meta-agents-research-environments/gaia2`. It reuses the shared Hugging Face
cache when present and otherwise downloads the dataset, then writes the 160
serialized scenarios unchanged to:

```text
/mnt/shared_ru.ml.SZ-5_000264/sukhorukov/decomposer_artifacts/
  evaluation/data/gaia2/<dataset-revision>/validation/execution/
```

The manifest is outside the scenario directory, so ARE never interprets it as
a task. `--reuse-source` forbids Hugging Face access and only validates the
already materialized files. `--skip-runtime` validates the lock-keyed runtime
without creating it. Select one experiment with repeated `--experiment` flags.

Preparation also mirrors revision
`132e26376f5e963bb59f64bcccdd02188cb08dee` of
`meta-agents-research-environments/gaia2_filesystem` into the artifact tree and
points `DEMO_FS_PATH` at its local `demo_filesystem/`. Evaluation therefore
does not lazily fetch execution-task files from Hugging Face.

The Gaia runtime is created with `uv sync --frozen` under
`decomposer_artifacts/venvs/gaia2/<uv-lock-hash>`. Evaluation never falls back
to a live Hugging Face dataset.

## Audit tool schemas before evaluation

GAIA2 owns the canonical `AppTool` to OpenAI JSON Schema converter. Native
simple agents and broker-backed Decomposer workers consume the same output;
registered `AppTool`s never fall back to ARE's legacy scalar-only converter.
The converter recursively supports unions, lists, string-keyed dictionaries,
`Literal`, and closed `TypedDict` objects. It omits variadic Python arguments,
preserves explicit defaults, and rejects unsupported annotations with the tool
and argument name. Arguments with ordinary defaults remain optional. Empty-string
defaults are the deliberate exception: Gaia2 uses them for payload fields such as
`content`, `subject`, and `discount_code`, so those fields remain required. Set
`ARE_TOOLS_REQUIRE_ALL=1` only for the opt-in all-required schema ablation.

When an ARE tool raises a model-correctable execution error, both the simple
agent and the Decomposer worker append the same complete canonical OpenAI
schema for that one failing tool. The shared renderer uses compact JSON rather
than reconstructing a lossy text signature or repeating the complete registry.
Native simple-agent and Decomposer worker requests explicitly use
`tool_choice="auto"` whenever schemas are present. The Decomposer manager does
not call GAIA2 tools and is unchanged; `AgentUserInterface__send_message_to_user`
remains the simple agent's normal final-response tool.

Run the complete read-only gate after any tool or schema change:

```bash
.venv/bin/python -m gyms.gaia2.audit_tools
```

The audit instantiates all 20 registered app classes and validates every tool
with JSON Schema 2020-12. It also loads one pinned execution, search, and
ambiguity scenario and checks the actual native, broker, and LangChain-facing
schemas for byte-order-preserving equality. It rejects exposed `args` or
`kwargs`, hidden `cache_options`, incorrectly required ordinary defaults,
incorrectly optional empty-string payloads, unapproved broad objects, hidden AUI
leaks, and lossy native fallback. The report also pins the complete population of
empty-string payload surfaces for conscious review when the registry changes. It
parses the schema embedded in every retry reminder and requires exact equality
with the initial native schema. Finally, it repeats the audit under
`PYTHONHASHSEED=0,1,42` and requires one schema-and-reminder checksum.

`Contacts__edit_contact` exposes a closed partial-update object containing only
mutable contact fields. Unknown or incorrectly typed fields fail before
mutation. An audit of the 480 pinned execution, search, and ambiguity scenarios
found oracle updates only for `age`, `city_living`, `country`, `job`, `address`,
and `status`; none updates `contact_id` or `is_user`. The same audit found no
oracle use of filesystem `cache_options`, `block_size`, or variadic controls.
Filesystem tool paths accept logical `/foo`, relative `foo`, `~/foo`,
and paths returned by another filesystem tool, while rejecting traversal,
sibling-prefix, and symlink escapes. Tool errors shown to either agent surface
only logical paths and never the temporary physical sandbox root.

## Prepare the immutable training partition

`split_manifests/execution-110-50-v1.json` pins the dataset revision, source
checksum, seed 42, every scenario checksum, and complete-universe assignment.
Universes 25, 26, and 28 form the 50-scenario test holdout; the other seven
universes form the 110-scenario training partition. Preparation copies and
checksum-validates isolated train/test views without modifying the original
validation files:

```bash
.venv/bin/python -m gyms.gaia2.prepare eval \
  --experiment deepseek-v4-flash-0731-teacher-qwen35-4b-non-thinking \
  --purpose trace-generation \
  --partition train \
  --reuse-source
```

## Run locally

The Gemma profiles use pinned Hugging Face snapshots. The dense 31B checkpoint
is intended for one 140 GB H200 and can be installed into the registry's shared
cache with:

```bash
HF_HOME=/mnt/shared_ru.ml.SZ-5_000264/.cache/huggingface \
  .venv/bin/hf download google/gemma-4-31B-it \
  --revision 842da3794eaa0b77d5f08bae87a17459d91ff475 \
  --max-workers 8
```

The registered experiments are:

- `gemma4-e4b-sft-deepseek-e4b-v1-8k-non-thinking-gemma4-e4b-thinking`:
  tuned non-thinking E4B manager on the first GPU and vanilla thinking E4B
  worker on the second GPU, using the student prompt.
- `deepseek-v4-flash-0731-teacher-gemma4-e4b-thinking`: OpenRouter DeepSeek
  manager with high reasoning and the teacher prompt, plus a local thinking
  E4B worker on one GPU.
- `deepseek-v4-flash-0731-teacher-qwen35-4b-non-thinking`: the same remote
  teacher manager with a local non-thinking Qwen3.5-4B worker on one GPU.
- `deepseek-v4-flash-0731-teacher-gaia2-ambiguity-policy-qwen35-4b-non-thinking`:
  a separately identified prompt-alignment ablation using the same manager and
  worker. It appends the short GAIA2 ambiguity policy to the manager prompt;
  canonical DeepSeek results remain unchanged.
- `qwen36-35b-a3b-teacher-qwen35-4b-non-thinking`: the internal
  `Qwen/Qwen3.6-35B-A3B-FP8` teacher with the same local worker and teacher
  prompt. A credential-isolating loopback proxy normalizes the deployment's
  Qwen XML tool calls into Responses API function calls. The deployment emits
  reasoning at its service-default effort; unlike DeepSeek, no explicit
  `reasoning_effort=high` control is sent.
- `qwen36-35b-a3b-non-thinking-teacher-qwen35-4b-non-thinking-text-defaults`:
  an explicitly non-thinking Qwen3.6 manager through `LLM_PROXY_URL`, using
  Qwen's general instruct sampling preset and the teacher prompt, with a local
  text-only non-thinking Qwen3.5-4B worker. Manager and worker use 128K context,
  provider-controlled completion length, and independent 80-model-call caps.
- `qwen36-35b-a3b-thinking-teacher-qwen35-4b-non-thinking-text-defaults`:
  the matched explicit-thinking Qwen3.6 manager, also reached through the
  credential-isolating LLM proxy. It uses Qwen's thinking preset and captures
  manager reasoning, but the upstream Responses deployment discards reasoning
  items replayed by the harness. It is therefore a capture-only diagnostic,
  not a clean comparison with full-replay DeepSeek or local Gemma.
- `gemma4-26b-a4b-thinking-gemma4-e4b-thinking-text-defaults`: local thinking
  Gemma-4-26B-A4B manager and thinking Gemma-4-E4B worker using Gemma's
  `temperature=1.0`, `top_p=0.95`, `top_k=64` preset and the student prompt.
  Both models use 128K context, provider-controlled completion length, and
  independent 80-model-call caps.
- `deepseek-v4-flash-0731-teacher-gemma4-26b-a4b-non-thinking`: OpenRouter
  DeepSeek with high reasoning and the teacher prompt, plus one local
  non-thinking Gemma-4-26B-A4B worker on one GPU.
- `gemma4-26b-a4b-thinking-teacher-gemma4-26b-a4b-non-thinking-text-defaults`:
  a thinking Gemma-4-26B-A4B teacher and non-thinking worker using the teacher
  prompt. Both actors share one checkpoint, vLLM process, GPU, and endpoint;
  their per-request chat-template settings select thinking independently.
- `qwen35-4b-sft-workplace-v1-3765-32k-non-thinking-qwen35-4b-non-thinking`:
  the Workplace-trained non-thinking Qwen3.5-4B manager on the first GPU and
  a base non-thinking Qwen3.5-4B worker on the second GPU, using the student
  prompt and 128K context.
- `qwen35-4b-sft-mixed-v1-partial-3983f605-327-32k-non-thinking-qwen35-4b-non-thinking`:
  the Workplace plus partial-Toolathlon SFT manager with the same student
  prompt, base non-thinking Qwen3.5-4B worker, and two-GPU topology.
- `qwen35-4b-sft-mixed-v1-final-493c24c4-404-non-thinking-qwen35-4b-non-thinking`:
  the Workplace plus full terminal-Toolathlon snapshot SFT manager with the
  same student prompt, base worker, and two-GPU topology.
- `qwen35-4b-sft-mixed-v1-final-493c24c4-404-filtered-p1-s279-non-thinking-qwen35-4b-non-thinking`:
  the filtered final-snapshot patience-1 best checkpoint at step 279, with the
  same student prompt, base worker, and two-GPU topology.
- `qwen35-4b-sft-mixed-v2-493c24c4-gaia2-110-n3-filtered-p2-non-thinking-qwen35-4b-non-thinking`:
  the patience-2 mixed Workplace, Toolathlon, and reward-1 Gaia2 checkpoint,
  with the same student prompt, base worker, and two-GPU topology.
- `qwen35-4b-sft-toolathlon-only-v1-493c24c4-teacher-prompt-filtered-32k-non-thinking-qwen35-4b-non-thinking`:
  the filtered Toolathlon-only checkpoint with the same non-thinking base
  worker and two-GPU topology.
- `qwen35-4b-sft-gaia2-execution-only-v1-110-n10-teacher-prompt-r1-balanced-32k-non-thinking-qwen35-4b-non-thinking`:
  the reward-1 balanced GAIA2 execution-only checkpoint with the same base
  worker and two-GPU topology.
- `qwen35-4b-base-non-thinking-qwen35-4b-non-thinking`: untuned non-thinking
  Qwen3.5-4B manager and worker using the compact student prompt.
- `qwen35-4b-base-non-thinking-teacher-qwen35-4b-non-thinking`: the same
  untuned non-thinking Qwen3.5-4B manager and worker using the full teacher
  prompt. This is a separate experiment so its manifest and results cannot
  collide with the student-prompt baseline.
- `gemma4-e2b-it-{non-thinking,thinking}`,
  `gemma4-e4b-it-{non-thinking,thinking}`,
  `gemma4-31b-it-{non-thinking,thinking}`, and
  `gemma4-26b-a4b-it-{non-thinking,thinking}`: matched vanilla Gemma-4 simple
  agents on one GPU. Every profile uses the official text sampling preset,
  128K context, provider-controlled output length, and an 80-call budget.
- `qwen35-4b-non-thinking`: vanilla non-thinking Qwen3.5-4B simple agent on
  one GPU, using the recommended general-task sampling parameters and 128K
  context.
- `qwen35-4b-non-thinking-simple-general-text-defaults` and
  `gemma4-e4b-thinking-simple-text-defaults`: matched text-only controls for
  the two new Decomposer profiles. Each uses 128K context, provider-controlled
  output length, and the shared 80-call per-actor budget.
- `qwen35-2b-base-non-thinking` and `qwen35-9b-base-non-thinking`: matching
  non-thinking simple-agent baselines using the pinned local checkpoints.
- `deepseek-v4-flash-0731`: remote OpenRouter DeepSeek simple agent with high
  reasoning. The runner starts only a credential-isolating loopback proxy and
  does not allocate a GPU.

All GAIA2 experiments use the same per-actor budget: a simple agent may make
80 actual policy-model invocations, while a Decomposer manager and each of its
subagents may independently make 80. Locally queued calls from one native
multi-tool response do not consume additional model calls. This is not an
aggregate-compute limit: spawning multiple subagents can make Decomposer's
episode total larger. Budget exhaustion is recorded as a failed rollout and
does not stop the remaining evaluation.

Simple agents, managers, and subagents do not set an explicit per-response
token limit; output length is controlled by the model server or provider and
remains bounded by its finite context. Input-context overflow and provider
output truncation are terminal for only the affected actor (`fail_actor_v1`):
the harness does not retry an impossible request, compact history, or execute a
truncated tool call. A simple-agent or manager overflow makes that rollout a
recorded failure; a subagent overflow is returned to the manager as an error
report. Subsequent benchmark rollouts still run normally.

```bash
# Decomposer, three attempts per scenario.
.venv/bin/python -m gyms.gaia2.run \
  --experiment gemma4-e4b-sft-deepseek-e4b-v1-8k-non-thinking-gemma4-e4b-thinking \
  --num-repeats 3 \
  --cuda-visible-devices 6,7

# Qwen SFT Decomposer, three attempts per scenario.
.venv/bin/python -m gyms.gaia2.run \
  --experiment qwen35-4b-sft-workplace-v1-3765-32k-non-thinking-qwen35-4b-non-thinking \
  --num-repeats 3 \
  --cuda-visible-devices 6,7

# Mixed SFT Decomposer, three attempts per scenario.
.venv/bin/python -m gyms.gaia2.run \
  --experiment qwen35-4b-sft-mixed-v1-partial-3983f605-327-32k-non-thinking-qwen35-4b-non-thinking \
  --num-repeats 3 \
  --cuda-visible-devices 6,7

# Filtered final-snapshot SFT Decomposer, three attempts per scenario.
.venv/bin/python -m gyms.gaia2.run \
  --experiment qwen35-4b-sft-mixed-v1-final-493c24c4-404-filtered-p1-s279-non-thinking-qwen35-4b-non-thinking \
  --num-repeats 3 \
  --cuda-visible-devices 6,7

# Full final-snapshot SFT Decomposer, three attempts per scenario.
.venv/bin/python -m gyms.gaia2.run \
  --experiment qwen35-4b-sft-mixed-v1-final-493c24c4-404-non-thinking-qwen35-4b-non-thinking \
  --num-repeats 3 \
  --cuda-visible-devices 6,7

# Untuned Qwen Decomposer with the teacher prompt, three attempts per scenario.
.venv/bin/python -m gyms.gaia2.run \
  --experiment qwen35-4b-base-non-thinking-teacher-qwen35-4b-non-thinking \
  --num-repeats 3 \
  --cuda-visible-devices 6,7

# Explicit-thinking Qwen3.6 proxy manager with the teacher prompt and one
# local non-thinking Qwen3.5-4B worker.
.venv/bin/python -m gyms.gaia2.run \
  --experiment qwen36-35b-a3b-thinking-teacher-qwen35-4b-non-thinking-text-defaults \
  --domain execution \
  --partition test \
  --num-repeats 3 \
  --concurrency 16 \
  --cuda-visible-devices 0

# Simple-agent one-scenario smoke.
.venv/bin/python -m gyms.gaia2.run \
  --experiment gemma4-e4b-it-thinking \
  --num-repeats 1 \
  --limit 1 \
  --cuda-visible-devices 7

# Simple Qwen, three attempts per scenario.
.venv/bin/python -m gyms.gaia2.run \
  --experiment qwen35-4b-non-thinking \
  --num-repeats 3 \
  --cuda-visible-devices 7

# Remote DeepSeek simple agent, three attempts per scenario. No CUDA flag.
.venv/bin/python -m gyms.gaia2.run \
  --experiment deepseek-v4-flash-0731 \
  --num-repeats 3

# OpenRouter teacher one-scenario smoke. HTTPS_PROXY (or https_proxy) and
# OPENROUTER_API_KEY_DECOMPOSER must also be exported.
.venv/bin/python -m gyms.gaia2.run \
  --experiment deepseek-v4-flash-0731-teacher-gemma4-e4b-thinking \
  --num-repeats 1 \
  --limit 1 \
  --cuda-visible-devices 6 \
  --output-dir "$HOME/decomposer_artifacts/evaluation/gaia2/corrected-smoke-gemma"
```

`LLM_PROXY_URL` and `LLM_PROXY_MASTER_KEY` must point to the canonical proxy
judge. Use `--dry` to print every service and ARE command without starting
processes. `--output-dir` isolates an ad-hoc run. A rerun skips a completed
marker; `--force` archives the previous attempt before starting again.

For parallel local runs, pass a distinct non-negative `--port-offset` to each
runner. The offset is added consistently to the simple-agent server or proxy,
manager server or proxy, worker vLLM, LangGraph server, and Decomposer service.
Remote manager and judge URLs are not changed. Offset zero preserves the
existing ports and artifact names; a nonzero default output gains a
`-port-offset-N` suffix so its logs, caches, runtime configuration, and resume
markers stay isolated. Explicit `--output-dir` paths are not renamed, but their
stored run identity still prevents reuse with another offset. Every effective
port must remain at or below 65535.

The following three runs use four GPUs and disjoint port layouts. A stride of
12000 matches the Workplace convention and leaves ample separation:

```bash
# GPU 0, base ports.
.venv/bin/python -m gyms.gaia2.run \
  --domain search \
  --partition test \
  --experiment qwen35-4b-non-thinking \
  --num-repeats 3 \
  --cuda-visible-devices 0 \
  --port-offset 0 &

# GPUs 1 and 2, every local endpoint shifted by 12000.
.venv/bin/python -m gyms.gaia2.run \
  --domain search \
  --partition test \
  --experiment qwen35-4b-sft-workplace-v1-3765-32k-non-thinking-qwen35-4b-non-thinking \
  --num-repeats 3 \
  --cuda-visible-devices 1,2 \
  --port-offset 12000 &

# GPU 3 plus the remote Qwen3.6 manager, shifted by 24000.
.venv/bin/python -m gyms.gaia2.run \
  --domain search \
  --partition test \
  --experiment qwen36-35b-a3b-teacher-qwen35-4b-non-thinking \
  --num-repeats 3 \
  --cuda-visible-devices 3 \
  --port-offset 24000 &

wait
```

The option applies to execution, search, and ambiguity evaluations and to
execution trace generation. MLSpace jobs continue to use offset zero because
each job has an isolated network namespace.

Each Decomposer experiment declares its student or teacher prompt profile.
`--prompt-profile teacher|student` can override it for local and MLSpace runs;
explicit overrides use a distinct output/job identity and record the resolved
prompt hash.
Workers see strict JSON schemas generated from the original scenario-bound ARE
tools through an authenticated loopback broker; defaulted parameters are
optional and variadic Python parameters are not exposed. The final
user-interface tools remain manager-only. Multiple tool calls are valid and
each is executed once under the scenario lock. Invalid argument types are
returned to the worker as correctable tool feedback and are never silently
coerced.

Before scheduling full reruns, run one execution and one search task for each
of the matched simple and Decomposer profiles below. Give every local process
an isolated output directory and port offset, and run the commands sequentially
when sharing GPUs:

```text
qwen35-4b-non-thinking-simple-general-text-defaults
gemma4-e4b-thinking-simple-text-defaults
qwen36-35b-a3b-non-thinking-teacher-qwen35-4b-non-thinking-text-defaults
qwen36-35b-a3b-thinking-teacher-qwen35-4b-non-thinking-text-defaults
gemma4-26b-a4b-thinking-gemma4-e4b-thinking-text-defaults
```

For each profile use `--num-repeats 1 --limit 1`, then inspect `output.jsonl`,
`hf/`, `lite/`, and Decomposer sidecars for schema/type errors, duplicated
sandbox roots, or physical `are_simulation_fs_sandbox_...` paths. For native
simple agents, also inspect the wire dump and service log to confirm that vLLM
accepted `tool_choice="auto"`, recognized emitted tool calls, rejected invalid
extra properties, and delivered the final answer through
`AgentUserInterface__send_message_to_user`. A smoke is a workflow integrity
check; its task reward is not an acceptance criterion.

OpenRouter Responses API reasoning blocks remain available in the sidecar
manager trace, but only visible text blocks are sent to the Gaia2 user
interface. Local Chat Completions reasoning is captured and replayed on every
later call; thinking profiles enable Gemma's `preserve_thinking` chat-template
option so a newer user turn cannot suppress earlier reasoning. Simple-agent
traces store it as a separate `reasoning_content` field; local Decomposer
manager sidecars store it in the manager message metadata. Local Decomposer
subagents receive their prior reasoning during the active run, but their
private message histories are not copied into the parent sidecar. The runtime
identity records
`structured_reasoning_policy=capture_replay_v2_template_preserved`, so these
runs cannot resume from or mix with older turn-local-reasoning artifacts.
Qwen3.6 managers reached
through the current LLM-proxy Responses transport are the explicit exception:
the upstream service captures output reasoning but discards replayed input
reasoning, so their identity records
`capture_only_upstream_no_replay_v1`. Migrating that path to Chat Completions
and a strict schema-aware XML parser is deferred.

ARE emits one `output.jsonl` row per attempted rollout. Missing/empty manager
answers, uncollected subagents, recursion limits, and scenario timeouts remain
failed rows with score zero and do not stop the other attempts. A service death,
invalid dataset/runtime, nonzero ARE exit, or incomplete rollout cardinality is
a systemic failure and does not create `.eval_done.json`.

## Submit MLSpace jobs

```bash
/mnt/shared_ru.ml.SZ-5_000264/sukhorukov/.venv-mls/bin/python \
  -m gyms.gaia2.run_eval \
  --experiment gemma4-e4b-it-thinking \
  --num-repeats 3 \
  --author-name sukhorukov \
  --dry

# Prepare and submit the mixed-SFT checkpoint after its final export exists.
.venv/bin/python -m gyms.gaia2.prepare eval \
  --experiment qwen35-4b-sft-mixed-v1-partial-3983f605-327-32k-non-thinking-qwen35-4b-non-thinking \
  --reuse-source

/mnt/shared_ru.ml.SZ-5_000264/sukhorukov/.venv-mls/bin/python \
  -m gyms.gaia2.run_eval \
  --experiment qwen35-4b-sft-mixed-v1-partial-3983f605-327-32k-non-thinking-qwen35-4b-non-thinking \
  --num-repeats 3 \
  --priority high \
  --author-name sukhorukov

# Prepare and submit the filtered final-snapshot step-279 checkpoint.
.venv/bin/python -m gyms.gaia2.prepare eval \
  --experiment qwen35-4b-sft-mixed-v1-final-493c24c4-404-filtered-p1-s279-non-thinking-qwen35-4b-non-thinking \
  --reuse-source

/mnt/shared_ru.ml.SZ-5_000264/sukhorukov/.venv-mls/bin/python \
  -m gyms.gaia2.run_eval \
  --experiment qwen35-4b-sft-mixed-v1-final-493c24c4-404-filtered-p1-s279-non-thinking-qwen35-4b-non-thinking \
  --num-repeats 3 \
  --priority high \
  --author-name sukhorukov

# Prepare and submit the full final-snapshot checkpoint.
.venv/bin/python -m gyms.gaia2.prepare eval \
  --experiment qwen35-4b-sft-mixed-v1-final-493c24c4-404-non-thinking-qwen35-4b-non-thinking \
  --reuse-source

/mnt/shared_ru.ml.SZ-5_000264/sukhorukov/.venv-mls/bin/python \
  -m gyms.gaia2.run_eval \
  --experiment qwen35-4b-sft-mixed-v1-final-493c24c4-404-non-thinking-qwen35-4b-non-thinking \
  --num-repeats 3 \
  --priority high \
  --author-name sukhorukov

# Prepare and submit the untuned Qwen teacher-prompt baseline.
.venv/bin/python -m gyms.gaia2.prepare eval \
  --experiment qwen35-4b-base-non-thinking-teacher-qwen35-4b-non-thinking \
  --reuse-source

/mnt/shared_ru.ml.SZ-5_000264/sukhorukov/.venv-mls/bin/python \
  -m gyms.gaia2.run_eval \
  --experiment qwen35-4b-base-non-thinking-teacher-qwen35-4b-non-thinking \
  --num-repeats 3 \
  --priority high \
  --author-name sukhorukov
```

The launcher uses one `a100plus.1gpu.80vG.12C.182G` instance for simple agents
and OpenRouter-managed Decomposers, and one
`a100plus.2gpu.80vG.24C.364G` instance when both manager and worker are local.
It skips completed results and matching Pending/Running jobs, refuses a dirty
worktree for real submissions, stages Decomposer by Git SHA, and reuses the
independently staged Gaia SHA. Dry-run payloads redact credentials and submit
nothing.

Held-out evaluation uses `--purpose evaluation --partition test`. It runs only
the 50 scenarios from universes 25, 26, and 28 and writes under the pinned
split namespace without changing the existing full-validation result paths.
For an n=3 complete test run, `comparison.json` reuses and checksum-pins the
held-out rows from the completed full-validation baselines; it does not rerun
or rejudge them.

## Evaluate the Search domain

Search contains 160 single-turn, static scenarios. The immutable
`search-118-42-v1` split reserves the same complete universes 25, 26, and 28 as
execution, which yields 42 Search test tasks and 118 train tasks. The holdout is
not padded with tasks from exposed universes. `--partition full` remains
available for complete-domain evaluation.

Prepare a native simple agent and a Decomposer experiment together:

```bash
.venv/bin/python -m gyms.gaia2.prepare eval \
  --domain search \
  --partition test \
  --experiment qwen35-4b-non-thinking \
  --experiment qwen35-4b-sft-mixed-v2-493c24c4-gaia2-110-n3-filtered-p2-non-thinking-qwen35-4b-non-thinking
```

Run either locally by passing the same domain and partition. The simple agent
uses ARE `native_tools`; the Decomposer uses the broker-backed worker harness:

```bash
.venv/bin/python -m gyms.gaia2.run \
  --domain search \
  --partition test \
  --experiment qwen35-4b-non-thinking \
  --num-repeats 3 \
  --cuda-visible-devices 0

.venv/bin/python -m gyms.gaia2.run \
  --domain search \
  --partition test \
  --experiment qwen35-4b-sft-mixed-v2-493c24c4-gaia2-110-n3-filtered-p2-non-thinking-qwen35-4b-non-thinking \
  --num-repeats 3 \
  --cuda-visible-devices 0,1
```

The MLSpace launcher accepts the same `--domain search` flag. Search does not
support `--purpose trace-generation`; the CLI rejects that combination before
preparation or service startup.

The matched Qwen3.6 teacher comparison uses three attempts and concurrency 16:

```bash
.venv/bin/python -m gyms.gaia2.prepare eval \
  --experiment qwen36-35b-a3b-teacher-qwen35-4b-non-thinking \
  --purpose evaluation \
  --partition test \
  --reuse-source

/mnt/shared_ru.ml.SZ-5_000264/sukhorukov/.venv-mls/bin/python \
  -m gyms.gaia2.run_eval \
  --experiment qwen36-35b-a3b-teacher-qwen35-4b-non-thinking \
  --purpose evaluation \
  --partition test \
  --num-repeats 3 \
  --concurrency 16 \
  --priority high \
  --author-name sukhorukov
```

```bash
.venv/bin/python -m gyms.gaia2.prepare eval \
  --experiment qwen35-4b-sft-mixed-v2-493c24c4-gaia2-110-n3-filtered-p2-non-thinking-qwen35-4b-non-thinking \
  --purpose evaluation \
  --partition test \
  --reuse-source

/mnt/shared_ru.ml.SZ-5_000264/sukhorukov/.venv-mls/bin/python \
  -m gyms.gaia2.run_eval \
  --experiment qwen35-4b-sft-mixed-v2-493c24c4-gaia2-110-n3-filtered-p2-non-thinking-qwen35-4b-non-thinking \
  --purpose evaluation \
  --partition test \
  --num-repeats 3 \
  --concurrency 4 \
  --priority high \
  --author-name sukhorukov
```

## Evaluate the Ambiguity domain

Ambiguity contains 160 static scenarios. The immutable
`ambiguity-128-32-v1` split reserves complete universes 25, 26, and 28 for a
future 32-task holdout and leaves 128 training tasks; the baseline below uses
`--partition full` and evaluates all 160 tasks.

The native simple agent receives ARE's canonical benchmark-wide ambiguity rule.
The canonical Decomposer keeps its existing teacher prompt. A third, separately
named ablation appends a short equivalent policy to the Decomposer manager only.
The longer internal `eval/conf/ambiguity_prompt.txt` override is not used, and
the launcher does not set `ARE_EXTRA_SYSTEM_PROMPT` or
`ARE_EXTRA_SYSTEM_PROMPT_FILE`.

Prepare all three experiments:

```bash
.venv/bin/python -m gyms.gaia2.prepare eval \
  --domain ambiguity \
  --partition full \
  --experiment qwen35-4b-non-thinking \
  --experiment deepseek-v4-flash-0731-teacher-qwen35-4b-non-thinking \
  --experiment deepseek-v4-flash-0731-teacher-gaia2-ambiguity-policy-qwen35-4b-non-thinking
```

Dry-run or submit the matched full-validation jobs together:

```bash
/mnt/shared_ru.ml.SZ-5_000264/sukhorukov/.venv-mls/bin/python \
  -m gyms.gaia2.run_eval \
  --domain ambiguity \
  --partition full \
  --experiment qwen35-4b-non-thinking \
  --experiment deepseek-v4-flash-0731-teacher-qwen35-4b-non-thinking \
  --experiment deepseek-v4-flash-0731-teacher-gaia2-ambiguity-policy-qwen35-4b-non-thinking \
  --num-repeats 3 \
  --concurrency 4 \
  --priority high \
  --author-name sukhorukov
```

Each job uses one A100, makes exactly 480 attempts, and writes to an isolated
`validation/ambiguity/<experiment>-n3` result directory. The first two jobs are
the primary canonical comparison; the policy-aligned Decomposer is an ablation.

### Generate seven additional teacher traces per training scenario

The trace workflow starts the Qwen worker, LangGraph server, and Decomposer
service once. It then makes seven sequential ARE passes with one native run per
scenario, so dispatch covers all 110 scenarios before repeating any scenario.
Logical rollout numbers 4 through 10 line up with the existing n=3 teacher run:

```bash
/mnt/shared_ru.ml.SZ-5_000264/sukhorukov/.venv-mls/bin/python \
  -m gyms.gaia2.run_eval \
  --experiment deepseek-v4-flash-0731-teacher-qwen35-4b-non-thinking \
  --purpose trace-generation \
  --partition train \
  --num-repeats 7 \
  --rollout-offset 3 \
  --concurrency 10 \
  --priority high \
  --author-name sukhorukov \
  --dry
```

Remove `--dry` to submit. Raw outputs live in `round_04` through `round_10`
under `evaluation/gaia2/traces/execution-110-50-v1/train/`. A successful round
gets `.round_done.json`; resubmitting the same command skips those rounds but
reruns a partial systemic-failure round. If ARE finished the full grid before a
validator failure, the runner validates and seals the unmarked round instead of
recomputing it. ARE omits `run_number` for these single-run passes, so the trace
manifest records native run `0`, matching the `__run0.json` sidecars.
Per-scenario HTTP, recursion, timeout, and judge failures remain score-zero
attempted records and are not retried.
After validating 770 unique `(scenario_id, logical_rollout_number)` keys and
seven keys per scenario, the runner writes `trace_manifest.jsonl`, metrics, and
finally `.trace_done.json`.

Canonical results are stored under:

```text
decomposer_artifacts/evaluation/gaia2/results/
  validation/execution/<experiment>-n3/
```

Each completed directory contains native `lite/` and `hf/` traces,
`output.jsonl`, service logs, redacted Decomposer sidecars when applicable,
`metrics.json`, `run_status.json`, and `.eval_done.json`.
