# Gaia2 execution evaluation

This package owns Gaia2 evaluation data preparation, the Decomposer external
agent adapter, local execution, and MLSpace submission. Evaluation uses the
complete `validation` split of the `execution` domain, while teacher trace
generation uses the immutable train/test partition described below.

The Gaia runtime is pinned through `external/gaia2` at commit
`3bee736488864e028231755ce2ee32a7065e8648`. Preparation materializes a
clean, immutable checkout under Decomposer artifacts. Every preparation and
run manifest records both repository commits.

## Prepare data and runtimes

```bash
.venv/bin/python -m gyms.gaia2.prepare eval \
  --split validation \
  --domain execution
```

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

The registered experiments are:

- `gemma4-e4b-sft-deepseek-e4b-v1-8k-non-thinking-gemma4-e4b-thinking`:
  tuned non-thinking E4B manager on the first GPU and vanilla thinking E4B
  worker on the second GPU, using the student prompt.
- `deepseek-v4-flash-0731-teacher-gemma4-e4b-thinking`: OpenRouter DeepSeek
  manager with high reasoning and the teacher prompt, plus a local thinking
  E4B worker on one GPU.
- `deepseek-v4-flash-0731-teacher-qwen35-4b-non-thinking`: the same remote
  teacher manager with a local non-thinking Qwen3.5-4B worker on one GPU.
- `qwen36-35b-a3b-teacher-qwen35-4b-non-thinking`: the internal
  `Qwen/Qwen3.6-35B-A3B-FP8` teacher with the same local worker and teacher
  prompt. A credential-isolating loopback proxy normalizes the deployment's
  Qwen XML tool calls into Responses API function calls. The deployment emits
  reasoning at its service-default effort; unlike DeepSeek, no explicit
  `reasoning_effort=high` control is sent.
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
- `qwen35-4b-base-non-thinking-qwen35-4b-non-thinking`: untuned non-thinking
  Qwen3.5-4B manager and worker using the compact student prompt.
- `qwen35-4b-base-non-thinking-teacher-qwen35-4b-non-thinking`: the same
  untuned non-thinking Qwen3.5-4B manager and worker using the full teacher
  prompt. This is a separate experiment so its manifest and results cannot
  collide with the student-prompt baseline.
- `gemma4-e4b-it-thinking`: vanilla thinking E4B simple agent on one GPU.
- `qwen35-4b-non-thinking`: vanilla non-thinking Qwen3.5-4B simple agent on
  one GPU, using the recommended general-task sampling parameters and 128K
  context.
- `qwen35-2b-base-non-thinking` and `qwen35-9b-base-non-thinking`: matching
  non-thinking simple-agent baselines using the pinned local checkpoints.
- `deepseek-v4-flash-0731`: remote OpenRouter DeepSeek simple agent with high
  reasoning. The runner starts only a credential-isolating loopback proxy and
  does not allocate a GPU.

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

OpenRouter Responses API reasoning blocks remain available in the sidecar
manager trace, but only visible text blocks are sent to the Gaia2 user
interface.

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
