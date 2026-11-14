# WideSeek Gym

Reusable task loading, fixed Wiki-2018 search/access tools, ReAct or Decomposer
execution, raw trajectory capture and native judging. Workflows live in
[`evals/wideseek`](../../evals/wideseek/README.md) and
[`sft/wideseek`](../../sft/wideseek/README.md).

## Setup

Configure the shared [lmrouter access](../../README.md) once. Models use the named
registry in `src/decomposer/models.py`, including its optional Unix-socket tunnel,
provider settings and retries. No local model GPU or OpenRouter is needed.

On Linux, allow about 160 GB for the corpus/index plus space for the SQLite URL
index and temporary Qdrant optimization files (budget 250 GB). Pages are read
from disk using a SQLite offset index; the 27 GB page file stays on disk. Retrieval uses CPU FP32 E5; upstream uses GPU FP16, so retrieval is not
bit-for-bit identical. Existing downloaded assets are reused.

```bash
UV_BIN="$HOME/.local/bin/uv" bash gyms/wideseek/setup.sh
export WS_ASSETS=/large/disk/wideseek/assets
gyms/wideseek/.venv/bin/python -m gyms.wideseek.assets --root "$WS_ASSETS"
gyms/wideseek/.venv/bin/python -m gyms.wideseek.pages "$WS_ASSETS/Wiki-2018-Corpus/wiki_webpages.jsonl"
```

Run these commands from the repository root. Setup creates an isolated
`gyms/wideseek/.venv` for the Gym and CPU retrieval dependencies; it does not
install into the root `.venv` used for training or other gyms.
It also initializes `external/RLinf` at the commit pinned by Git. The extracted
scorer and its Apache license live in `external/wideseek_reward/`.

Start each service in its own terminal or tmux session, saving stdout/stderr under
`artifacts/gyms/wideseek/setup/`:

```bash
bash gyms/wideseek/serve.sh qdrant
bash gyms/wideseek/serve.sh retrieval
bash gyms/wideseek/serve.sh workers
```

Local ports are 16333/16334 (Qdrant), 18080 (retrieval) and 18081 (workers).
`env.sh` requires `LLM_PROXY_MASTER_KEY` in the calling shell's environment.
Configure credentials using the shared lmrouter instructions before sourcing it;
the Gym does not read a personal credential file.
Export `LLM_PROXY_UNIX_SOCKET` when using the relay. `WS_MODEL_BACKEND=local` selects the local Pozon profiles described below.
Sampling parameters come from the model registry. `WS_SEARCH_URL` and `WS_ASSETS` configure
services/storage only. `--output` selects the run directory on any writable disk.

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
ReAct and delegated researchers use thinking Qwen3.5-4B unlooped and the same
system prompt. Decomposer uses non-thinking
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

## Local Pozon Setup

The `pozon/local-wideseek-4b` branch provides `pozon.sh` for the measured host:
WSL2, RTX 5080 16 GiB, 30 GiB guest RAM. The project is under
`~/projects/vkr-wideseek/decomposer`, assets under `~/projects/vkr-wideseek/assets`.
Keep assets in the Linux filesystem for random-read performance. The WSL virtual
filesystem reports 757 GiB free, but its Windows C: backing drive had only
320 GiB free at setup; monitor both. The pinned corpus download is 155.9 GB.

Run from the repository root. Install `uv` into `~/.local/bin` first, then:

```bash
UV_BIN="$HOME/.local/bin/uv" bash gyms/wideseek/setup.sh
export WS_ASSETS="$HOME/projects/vkr-wideseek/assets"
gyms/wideseek/.venv/bin/python -m gyms.wideseek.assets --root "$WS_ASSETS"
bash gyms/wideseek/pozon.sh pages
bash gyms/wideseek/pozon.sh setup-vllm
```

The page-index build streams JSONL into a SQLite URL/byte-offset index. Duplicate
URLs preserve the old loader's last-row-wins behavior. Rebuild after replacing
the page file; startup rejects an index with a mismatched size or modification
time. Treat the corpus as immutable while retrieval runs. Do not run two index
builds simultaneously. Failed builds leave the previous complete index intact.

Start these services in separate tmux sessions, saving logs outside the checkout:

```bash
bash gyms/wideseek/pozon.sh qdrant
bash gyms/wideseek/pozon.sh retrieval
bash gyms/wideseek/pozon.sh vllm
# Only needed for --agent decomposer:
bash gyms/wideseek/pozon.sh workers
```

Qdrant uses Docker v1.16.2, binds localhost, and has an 8 GiB RAM cap with one
optimization thread. Stop the earlier empty smoke container first if it occupies
the ports: `docker stop vkr-wideseek-qdrant-smoke`. The command imports the
upstream store and requests disk-backed vectors and HNSW through the API.
Segment migration is asynchronous; inspect the collection's `status`,
`optimizer_status`, and `config` before evaluating. The first import/migration
has not yet been measured on this host. If startup exceeds the memory cap,
inspect the logs before changing it. After creation use
`docker start vkr-wideseek-qdrant` to restart; do not run a second process on the
same store. Do not overwrite a migrated store with a fresh corpus download.

vLLM has a separate `.venv-vllm` (CUDA 12.9 build 0.24.0), keeping CPU retrieval's
Torch isolated. It serves BF16 Qwen3.5-4B on localhost:8024, with 16k context,
one active sequence, 2k prefill batches, eager execution and 85% GPU allocation.
The vision encoder is disabled. Sampling uses the PyTorch fallback to avoid
requiring a system CUDA toolkit for FlashInfer JIT compilation. Validate the actual VRAM requirement on startup.
Longer episodes can exceed 16k; those failures remain in the run artifacts.
The standard checkpoint differs from the hosted unlooped deployment.

```bash
gyms/wideseek/.venv/bin/python -m gyms.wideseek.prepare --source width
export WS_MODEL_BACKEND=local
source gyms/wideseek/env.sh
gyms/wideseek/.venv/bin/python -m evals.wideseek.run --agent react \
  --output artifacts/evals/wideseek/runs/pozon-react-smoke \
  --limit 1 -n 1 --concurrency 1 --model-calls 8 --output-tokens 8192
```

`WS_MODEL_BACKEND=local` selects local non-thinking 4B for ReAct, researchers,
manager and judge. Both client requests and the server default set
`enable_thinking=false`. It needs no hosted key. Without it,
existing hosted profiles remain the default. A single endpoint serializes all
roles; begin with ReAct. Scores from this 4B judge are development diagnostics
and are not comparable to scores from the hosted Flash Next judge. These are
WideSeek training tasks, not the public held-out WideSearch benchmark.

References: [Qwen3.5 vLLM recipe](https://github.com/vllm-project/recipes/blob/main/Qwen/Qwen3.5.md),
[Qdrant collection configuration](https://qdrant.tech/documentation/manage-data/collections/).
