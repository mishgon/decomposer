# WideSeek Gym

Reusable task loading, fixed Wiki-2018 search/access tools, ReAct or Decomposer
execution, raw trajectory capture and native judging. Workflows live in
[`evals/wideseek`](../../evals/wideseek/README.md) and
[`sft/wideseek`](../../sft/wideseek/README.md).

## Setup

Configure the shared [lmrouter access](../../README.md) once. Models use the named
registry in `src/decomposer/models.py`, including its optional Unix-socket tunnel,
provider settings and retries. No local model GPU or OpenRouter is needed.

On Linux, allow about 160 GB for the corpus/index and enough RAM to load the 27 GB
page file. Retrieval uses CPU FP32 E5; upstream uses GPU FP16, so retrieval is not
bit-for-bit identical. Existing downloaded assets are reused.

```bash
UV_BIN="$HOME/.local/bin/uv" bash gyms/wideseek/setup.sh
export WS_ASSETS=/large/disk/wideseek/assets
.venv/bin/python -m gyms.wideseek.assets --root "$WS_ASSETS"
```

Start each service in its own terminal or tmux session, saving stdout/stderr under
`artifacts/gyms/wideseek/setup/`:

```bash
bash gyms/wideseek/serve.sh qdrant
bash gyms/wideseek/serve.sh retrieval
bash gyms/wideseek/serve.sh workers
```

Local ports are 16333/16334 (Qdrant), 18080 (retrieval) and 18081 (workers).
`env.sh` loads credentials from `LMROUTER_ENV` or the home environment file.
Export `LLM_PROXY_UNIX_SOCKET` when using the relay. Model parameters are not read
from Gym environment variables. `WS_SEARCH_URL`, `WS_ASSETS` and
`WS_ARTIFACT_ROOT` configure services/storage only.

## Tasks and Raw Execution

```bash
.venv/bin/python -m gyms.wideseek.prepare --source width
source gyms/wideseek/env.sh
.venv/bin/python -m gyms.wideseek.run --agent react \
  --output artifacts/gyms/wideseek/runs/raw-smoke --limit 2 -n 1 --concurrency 2
```

Preparation supports `width`, `depth` and `hybrid`, each with 20,000 rows. Hybrid
mixes the other sources; it is not an independent held-out set. Preparation pins
the dataset revision, preserves reference answers for judging, records hashes
and refuses to overwrite an existing directory. Agents receive only the question
and output-format instructions. Split future held-out sets by question hash.

`--agent react|decomposer` selects a named configuration from `agents.py`.
ReAct uses non-thinking Qwen3.5-4B unlooped. Decomposer uses non-thinking
Qwen3.8 Flash Next with thinking Qwen3.5-4B unlooped researchers. The judge is
Flash Next. Model profiles are constants in `agents.py`; sampling settings come
from the shared model registry. There are no model-override CLI flags.

Controllers run in the runner process. Delegated researchers share one Agent
Server, addressed by `--worker-url`. Its stable assistant ID is `researcher`,
independent of model profile names. Each episode has a unique controller thread,
checkpoint store and artifact directory. Cleanup only touches researchers
recorded in that episode's state, never other episodes on the shared server.
`serve.sh workers` provides four shared run slots, not four slots per episode;
concurrent episodes can queue behind each other. Wikipedia retrieval stays shared.

Each agent has recursion limit 410; task execution timeout is 45 minutes.
Optional `--model-calls` and `--output-tokens` provide shared smoke budgets.
The Gym adds no default completion cap; hosted context/output limits still apply.
Use `--resume` with identical settings/data/source to skip saved results. An
interrupted attempt gets a fresh execution directory, preserving its old logs.
A file lock prevents concurrent writers to one run.

`--allow-source-change` explicitly records a Gym-code migration during resume.
Data, models, limits and concurrency must still match. Different core Decomposer
harness versions are rejected. Workflows can reuse `prepare_run()` and
`run_jobs()` to schedule explicit task/attempt pairs without adding policy here.

`manifest.json` records data/source hashes, registry settings, retrieval revisions
and package versions. Each `simple|decomposer/<task>/attempt-NNN/result.json`
references an execution directory with model/tool/judge logs, final graph state,
worker states and provider usage. Failed attempts remain available for analysis.
No aggregate evaluation or collection policy lives in this directory.

At episode end, cleanup discovers runs on every known subagent thread, including
runs whose launch response was lost. It cancels active runs and waits for them,
then saves one `subagents/<thread_id>.json` archive containing final state and run
metadata before deleting the remote thread/checkpoints. Model/tool logs remain
in the execution directory. Failed cancellation or archiving retains the remote
thread and records a cleanup error, excluding that attempt from SFT successes.

The current hosted Qwen4B deployment has a 128k context limit. Overflow stops that
worker and `wait()` reports its error to the Decomposer. `run()` and `fork()` reject
failed workers; `new()` creates a replacement with empty history. Other workers
and tasks continue. Controller overflow is saved as `context_exceeded`, with its
partial trace and native evaluation, without aborting the collection. The Gym
does not expand the provider limit or silently truncate the conversation.

## Native Scoring and Tests

Width tasks use upstream's strict Markdown-table extraction and item-F1. Depth
tasks use boxed-answer extraction and equivalence judging. Judge/API failures
remain unscored. Flash Next judging is a development diagnostic; this training
dataset run is not a published WideSearch benchmark reproduction.

```bash
PYTHONPATH=src .venv/bin/pytest tests/test_wideseek*.py -q
```

The vendored scorer retains upstream functions and Apache license. Tests cover
scoring, invalid-tool feedback, shared budgets, interrupted executions, model
profiles and workflow separation. Live retrieval and model/tool smokes must also
pass before bulk collection.

Sources: [training data](https://huggingface.co/datasets/RLinf/WideSeek-R1-train-data),
[RLinf implementation](https://github.com/RLinf/RLinf), and
[public evaluation](https://github.com/RLinf/WideSeek-R1-Eval).
