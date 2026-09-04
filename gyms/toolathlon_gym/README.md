# Toolathlon-Gym

This gym integration keeps Decomposer and its model credentials on the host while a
task container owns the Toolathlon workspace, MCP servers, and evaluator.
The host runner owns one supervised vLLM process for a batch. Task containers
reuse it and do not pay a per-example model cold start.
The configured Gemma4-26B-A4B subagent runs in non-thinking mode.

## Image layout

The adapter image extends `toolathlon-pack:latest`, preserving Toolathlon's
existing `/opt/venv` and prebuilt MCP servers. It adds a separate Python 3.12
environment at `/opt/subagents` for the LangGraph subagent server.
The two environments are intentionally isolated. It also patches the bundled
Canvas MCP client to use its existing PostgreSQL router when `PG_HOST` is set,
while preserving its normal HTTP client outside Toolathlon.

Build both images from the repository root:

```bash
gyms/toolathlon_gym/build.sh
```

Override either image name with `TOOLATHLON_BASE_IMAGE` or
`TOOLATHLON_DECOMPOSER_IMAGE`.

## Episode lifecycle

The adapter image requires `TOOLATHLON_TASK` and an empty `/artifacts/data`
mount. Its `task.py serve` command prepares the task with Toolathlon's native
Python environment, writes `/artifacts/data/runtime.json`, and then starts the
LangGraph server on port 2024. The server uses the separate `/opt/subagents`
environment.

The container resolves the host vLLM server through this variable:

- `GEMMA_4_26B_A4B_BASE_URL`

It defaults to port 8023 on `host.docker.internal`.

Set the OpenRouter key and local subagent model once:

```bash
export OPENROUTER_API_KEY=...
GEMMA=/home/matrosov/models/gemma-4-26B-A4B-it
```

Run the full dataset once (an omitted `-n` means one repetition):

```bash
uv run python gyms/toolathlon_gym/run.py --all \
  --purpose trace-generation \
  --model deepseek/deepseek-v4-flash-0731 \
  --subagent-model "$GEMMA" \
  --subagent-gpu 7 \
  --concurrency 16 \
  --container-slots 2 \
  --n-jobs-per-worker 1000
```

Run a subset, with every selected task repeated three times:

```bash
uv run python gyms/toolathlon_gym/run.py \
  --tasks howtocook-event-menu-ppt canvas-announcement-summary \
  -n 3 \
  --purpose trace-generation \
  --subagent-model "$GEMMA" \
  --subagent-gpu 1
```

The command prints a run ID and manifest path. Resume an interrupted run with
that ID; completed episodes are skipped, while failed or interrupted episodes
get a new attempt directory:

```bash
uv run python gyms/toolathlon_gym/run.py \
  --resume 20260821T180000Z-0123abcd \
  --purpose trace-generation
```

Resume loads the model, image, GPU, port, timeout, and vLLM settings from the
manifest. Benchmark score `false` still counts as a completed harness episode
and is skipped. It is an agent-quality result, not a reason to repeat the run.

The original one-task smoke interface remains available:

```bash
uv run python gyms/toolathlon_gym/run.py howtocook-event-menu-ppt \
  --purpose trace-generation \
  --subagent-model "$GEMMA" \
  --subagent-gpu 0
```

Use `--reuse-vllm` only when the configured port already serves the expected
model and the runner must not own that external process.

For several externally managed replicas, pass their host-local ports as a pool.
The batch verifies every endpoint and assigns episodes round-robin:

```bash
uv run python gyms/toolathlon_gym/run.py --all \
  --purpose trace-generation \
  --subagent-ports 18100 18101 18102 18103 \
  --concurrency 16 \
  --n-jobs-per-worker 1000
```

## MLSpace inference pool

Toolathlon's Docker containers remain on Hertz-2. MLSpace jobs only run Qwen
vLLM replicas and expose them through dedicated SSH reverse tunnels bound to
Hertz-2 loopback. This avoids requiring Docker inside MLSpace.

The reproducible service definitions are in `mlspace_experiments.py`. The live
SR008 mapping uses one H100 for the connectivity pilot and sixteen high-priority
single-H100 jobs for the full pool. Independent jobs avoid waiting for scarce
eight-GPU shapes and keep one eviction from taking out half the pool. Artifacts,
staged code, service logs, and dedicated tunnel
credentials live outside the checkout at:

```text
/mnt/shared_ru.ml.SZ-5_000264/matrosov/decomposer-toolathlon-artifacts/
  code/<git-commit>/
  inference/<experiment>/
  secrets/
```

Run the zero-GPU payload validation from the configured OCC environment:

```bash
conda run -n decomposer_jobs python \
  gyms/toolathlon_gym/run_mlspace_inference_jobs.py \
  --full --author-name matrosov --telegram-nick js0n_statham --dry
```

Submit `--pilot` first and verify port 18099 from Hertz-2. Submit `--full` only
after that succeeds; the sixteen full jobs forward ports 18100 through 18115.

## Artifacts and resume behavior

All new output is under `artifacts/gyms/toolathlon_gym/`:

```text
runs/<run-id>/
  manifest.json                 # task/repetition status, score, paths, timings, errors
  events.jsonl                  # append-only raw lifecycle events
  vllm.log                      # shared server log, appended on resume
  attempts/<task>/rep-NNN/attempt-NNN/
    attempt.json
    runner.stdout.log
    runner.stderr.log
traces/<task>/<episode-id>/
  runtime.json
  trace.json
  answer.txt
  task.log
  task.inspect.json
  postgres.log
  postgres.inspect.json
  cleanup.json
  workspace/...
evals/<task>/<episode-id>/result.json
```

The manifest is atomically replaced after every state transition. Each
task/repetition has its own entry and each retry has a distinct attempt log
directory. Existing trace/evaluation directories are never overwritten. A
batch runs up to `--concurrency` episodes at once, creates a fresh Docker
network, PostgreSQL container, and task container for each one, and removes all
three in the worker's `finally` block. Container startup and cleanup are briefly
bounded by `--container-slots` independent locks (default 1; two slots are the
validated high-throughput setting), and each task receives its PostgreSQL container's network address
directly to avoid rootless Podman DNS races; Decomposer execution remains
concurrent. Each task container starts LangGraph with `--n-jobs-per-worker`
slots (default 1000). The supervised vLLM is started once before the worker pool
and stopped once after it. `--agent-timeout` and `--episode-timeout` bound stuck
work. Keep one supervised vLLM process on one GPU for each model.

## Runtime boundary

- Host: Decomposer, OpenRouter credentials, and vLLM servers.
- Task container: task preprocessing, LangGraph subagent server, MCP server
  processes, workspace, and native evaluation.
- Episode network: an isolated Toolathlon PostgreSQL container.

Evaluation records use the same episode identifier as their traces.
