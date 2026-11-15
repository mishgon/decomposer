# WideSeek Gym

Reusable task loading, the colleagues' hosted Agentic RAG tools, ReAct or Decomposer
execution, raw trajectory capture and native judging. Workflows live in
[`evals/wideseek`](../../evals/wideseek/README.md) and
[`sft/wideseek`](../../sft/wideseek/README.md).

## Setup

Configure shared [lmrouter access](../../README.md), install dependencies with
`bash gyms/wideseek/setup.sh`, and export `AGENTIC_RAG_API_KEY`.
`AGENTIC_RAG_URL` defaults to `https://lmrouter.2a2i.org/rag`.
Start researchers with `bash gyms/wideseek/serve.sh workers` (port 18081).
Local Qdrant and retrieval services are no longer required by these agents.
The existing asset and retrieval utilities remain for inspecting historical runs.

Both `react` and Decomposer researchers use `occ-4b` non-thinking and the same
`search`, `open`, `find`, `finish` protocol over the API's `wikipedia` source.
Only a valid `finish` submits an answer. Invalid turns and repeated identical
calls terminate after three attempts, as in the colleagues' harness.
Researchers have 180 seconds of work plus 180 seconds for `finish`, and at most
32 model calls per run. The 32nd call permits only `finish`, as in the colleagues'
harness. Context and cumulative output limits are 65,536 tokens;
individual responses allow 4,096 tokens.

## Tasks and Raw Execution

```bash
gyms/wideseek/.venv/bin/python -m gyms.wideseek.prepare --source width
source gyms/wideseek/env.sh
gyms/wideseek/.venv/bin/python -m gyms.wideseek.run --agent react \
  --output artifacts/gyms/wideseek/runs/raw-smoke --limit 2 -n 1 --concurrency 2
```

Preparation supports `width`, `depth` and `hybrid`, each with 20,000 rows. Hybrid
mixes the other sources; it is not an independent held-out set. Preparation pins
the dataset revision, preserves reference answers for judging, records hashes
and refuses to overwrite an existing directory. Agents receive only the question
and output-format instructions. Split future held-out sets by question hash.

`--agent react|decomposer` selects a named configuration from `agents.py`.
ReAct and delegated researchers use thinking Qwen3.5-4B and the same
system prompt. Decomposer uses non-thinking
Qwen3.8 Flash Next with thinking Qwen3.5-4B researchers. The judge is
Flash Next. Model profiles are constants in `agents.py`; sampling settings come
from the shared model registry. There are no model-override CLI flags.

Controllers run in the runner process. Delegated researchers share one Agent
Server, addressed by `--worker-url`. Its stable assistant ID is `researcher`,
independent of model profile names. Each episode has a unique controller thread,
checkpoint store and artifact directory. Cleanup only touches researchers
recorded in that episode's state, never other episodes on the shared server.
`serve.sh workers` provides four shared run slots, not four slots per episode;
concurrent episodes can queue behind each other. Wikipedia retrieval stays shared.

Each agent has recursion limit 410. `--episode-timeout` defaults to 2,700 seconds
and covers agent execution, worker cleanup and evaluation. On expiry, emergency
cleanup gets up to 60 additional seconds; available traces and results are then saved.
Optional `--model-calls` and `--output-tokens` provide shared smoke budgets.
The Gym adds no default completion cap; hosted context/output limits still apply.
Raw and evaluation runs require a new output directory. Only the SFT collector
supports `--resume`. A file lock prevents concurrent writers to one run.
Workflows reuse `describe_run()` and `run_jobs()` without adding scheduling policy
to the Gym.

`manifest.json` records the task-data hash, registry settings, retrieval revisions
and Git revision. Each `simple|decomposer/<task>/attempt-NNN/result.json`
references an execution directory with model/tool/judge logs, final graph state,
worker states and provider usage. Failed attempts remain available for analysis.
No aggregate evaluation or collection policy lives in this directory.

The manifest also indexes completed episodes with their status, native evaluation
and relative `trace_path`. Each execution saves `trace.json` with common episode,
model, message and agent-run fields, plus `usage.json` using the shared trace-usage
summary. `result.json` also counts provider calls from durable logs, including
failed or unfinished calls that did not reach the final graph state.
Decomposer traces include `trace.html` through the shared visualizer; ReAct has no
orchestration timeline. `model_calls.jsonl` appends request deltas and responses,
including reasoning. A call ID joins its start and outcome records; thread IDs
separate concurrent conversations. Calls without an outcome count as unfinished.
Judge calls remain separate so judge tokens do not inflate agent usage.

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
