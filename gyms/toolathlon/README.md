# Toolathlon benchmark

This integration runs Decomposer on the hkust-nlp [Toolathlon benchmark]
(`external/toolathlon`, pinned at the Toolathlon-Verified release). Decomposer
and its model credentials stay on the host; the task container owns the
Toolathlon environment, the MCP tool gateway, and the native evaluator,
exactly as the benchmark's decoupled agent loop prescribes.

This integration is the evaluation (test) path for Toolathlon: it measures
the agent, harness, and model. Trace collection and distillation happen in
the training gyms; the traces written here exist because the benchmark's
native evaluator requires `traj_log.json`, and they double as failure
analysis material.

[Toolathlon benchmark]: https://github.com/hkust-nlp/Toolathlon

## How an episode works

1. The runner starts the benchmark task image with `--network host`, mounting
   the episode directory as `/workspace/dumps` (and `/workspace/logs`).
   Everything the container listens on is therefore reachable on the host
   loopback, and the container reaches host services the same way.
2. The runner copies the trusted task directory and any user-provided
   credentials into the container, then runs the benchmark's
   `scripts.decoupled.container_preprocess` to prepare the workspace and
   produce a trusted task bundle.
3. The benchmark's `task_artifact_guard` stashes the evaluator and
   ground-truth artifacts so the agent cannot see them.
4. The container starts its MCP gateway (SSE) on a fresh host port. In
   `simple` mode the host invokes the Qwen LangGraph directly. In
   `decomposer` mode the runner also starts the container LangGraph server
   used by Decomposer subagents.
5. Qwen requests are served by host vLLM. The selected agent mode changes
   only the host harness; task preparation and native grading stay identical.
   Every subagent uses a 256,000-token context and recursion limit 410.
   DeepSeek models always run with high reasoning effort.
6. After the agent loop the runner writes the benchmark-format
   `traj_log.json` into the shared episode directory, restores the evaluator
   artifacts, and runs `scripts.decoupled.container_eval` in the container.
   The evaluator replaces the trajectory config with the trusted resolved
   config and grades the task; `eval_res.json` is read back into the
   evaluation record.

The episode runner exits zero only when the Decomposer loop completed with a
final answer. The benchmark evaluator forces the trajectory status to
`success` for exit code zero and downgrades a forged `success` after a
non-zero exit, so the exit code and the written trajectory always agree.

## Image layout

The adapter image extends the benchmark's official task image
(`docker.io/lockon0927/toolathlon-task-image:1016beta`), which already
provides the Python environment, Node.js, Kind/kubectl, Playwright, and the
local MCP servers. It bakes the pinned benchmark sources into `/workspace`
(keeping the image's own `pyproject.toml`, `uv.lock`, and `.venv` so `uv run`
does not re-sync) and adds a separate Python 3.12 environment at
`/opt/subagents` for the LangGraph subagent server.

Build the image from the repository root:

```bash
gyms/toolathlon/build.sh
```

Override the base or resulting image name with `TOOLATHLON_BASE_IMAGE` or
`TOOLATHLON_BENCH_IMAGE`. The build warns when `external/toolathlon` is
checked out at a different commit than the repository pins. It also applies
a fail-fast compatibility patch to the pinned 12306 MCP package for the
railway site's current bootstrap page and network transport; the MCP version,
tools, and schemas remain unchanged.

User-provided, gitignored benchmark credentials (GCP OAuth keys,
`token_key_session.py`, `configs/.mcp-auth`, Notion state, ...) are read from
the `external/toolathlon` checkout at run time: regular files are copied into
the container, and `configs/.mcp-auth` is bind-mounted for OAuth refresh
persistence.

## First-time setup

The runner self-heals the common first-run pitfalls:

- an uninitialized `external/toolathlon` submodule fails fast with the exact
  `git submodule update --init external/toolathlon` command to run;
- a missing `configs/global_configs.py` (gitignored in the benchmark) is
  created from `configs/global_configs_example.py`, the benchmark's own
  first-run step, and the runner warns if the submodule is checked out at a
  different commit than the repository pins;
- `configs/.mcp-auth/` is created so OAuth token refresh persists between
  runs;
- a missing task image fails with a pointer to `gyms/toolathlon/build.sh`,
  and a missing Docker/podman socket fails with the candidates it tried
  (rootless podman needs `systemctl --user enable --now podman.socket`
  first).

What it cannot provide: real app credentials. The example `global_configs.py`
ships placeholder API keys, and the per-app keys (GCP OAuth, Notion, email,
...) are gitignored user files — see the benchmark's
`global_preparation/how2register_accounts.md`. Tasks for unconfigured apps
will fail until their credentials are in place.

The Verified final pool also needs its local app stack. Run the benchmark's
`global_preparation/deploy_containers.sh` before a formal evaluation. On a
rootless Podman host, set `podman_or_docker="podman"` in the gitignored
`external/toolathlon/configs/global_configs.py`; the runner exposes the
selected rootless socket at both the Docker and Podman paths expected by
nested task tooling.

## Running

Choose the local Qwen snapshot:

```bash
QWEN=/home/matrosov/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/<snapshot>
```

Run the full dataset once (an omitted `-n` means one repetition):

```bash
uv run python gyms/toolathlon/run.py --all \
  --agent-mode simple \
  --purpose evaluation \
  --subagent-model "$QWEN" \
  --subagent-gpu 0 \
  --vllm-data-parallel-size 1 \
  --concurrency 16 \
  --container-slots 4
```

For the credential-feasible evaluation suite, run the 55 validated tasks three
times with `--all-valid -n 3`. Metrics still use the complete 108-task
benchmark as their denominator: every one of the 53 omitted tasks contributes
three failures to pass@1 and one failed task to pass@3 and pass^3. The manifest
records the omitted task names explicitly. Plain `--all` continues to run all
108 tasks.

```bash
uv run python gyms/toolathlon/run.py --all-valid -n 3 \
  --purpose evaluation \
  --subagent-model "$QWEN" \
  --subagent-gpu 0 \
  --vllm-data-parallel-size 1 \
  --concurrency 8 \
  --container-slots 4
```

`simple` does not require OpenRouter. For the Decomposer harness, use
`--agent-mode decomposer`, select its model with `--model`, and export
`OPENROUTER_API_KEY`. This keeps the model choice and harness choice
orthogonal for the planned Qwen, DeepSeek, and trained-Decomposer matrix.

Select the tool-agent provider independently with `--subagent-provider`:

```bash
# DeepSeek direct tool agent
uv run python gyms/toolathlon/run.py --all \
  --purpose evaluation --agent-mode simple \
  --subagent-provider openrouter \
  --subagent-model deepseek/deepseek-v4-flash-0731

# DeepSeek decomposer delegating to DeepSeek tool agents
uv run python gyms/toolathlon/run.py --all \
  --purpose evaluation --agent-mode decomposer \
  --model deepseek/deepseek-v4-flash-0731 \
  --subagent-provider openrouter \
  --subagent-model deepseek/deepseek-v4-flash-0731
```

OpenRouter tool-agent servers run on the host, so the API key is never exposed
inside task containers. Agent loops default to a 45-minute limit and complete
episodes to a 55-minute total limit; override these with `--agent-timeout` and
`--episode-timeout` for intentionally longer experiments.

If OpenRouter blocks the remote host's public IP, keep the API egress local.
Start the payload-silent, localhost-only relay on the local machine and expose
it to the remote host with an SSH reverse tunnel:

```bash
python gyms/toolathlon/openrouter_relay.py --port 18041
ssh -N -R 127.0.0.1:18041:127.0.0.1:18041 Hertz-2
```

Then export
`TOOLATHLON_OPENROUTER_BASE_URL=http://127.0.0.1:18041/api/v1` for the remote
evaluation command. The relay has a fixed `openrouter.ai` upstream, binds only
to loopback, and does not log authenticated request paths, headers, or bodies.

Use the hosted Qwen3.6 teacher through lmrouter with thinking explicitly
enabled:

```bash
uv run python gyms/toolathlon/run.py --tasks finalpool/find-alita-paper \
  --purpose evaluation --agent-mode decomposer \
  --decomposer-provider lmrouter --decomposer-prompt teacher \
  --model Qwen/Qwen3.6-35B-A3B-FP8 \
  --subagent-provider vllm --subagent-gpu 0 \
  --vllm-data-parallel-size 1
```

This requires `LLM_PROXY_URL` and `LLM_PROXY_MASTER_KEY`. Toolathlon enforces
one GPU per locally served model; data-parallel sizes above one are rejected.

Run a subset, with every selected task repeated three times:

```bash
uv run python gyms/toolathlon/run.py \
  --tasks finalpool/find-alita-paper finalpool/academic-pdf-report \
  -n 3 \
  --purpose evaluation \
  --subagent-model "$QWEN" \
  --subagent-gpu 0
```

The command prints a run ID and manifest path. Resume an interrupted run with
that ID; completed episodes are skipped, while failed or interrupted episodes
get a new attempt directory:

```bash
uv run python gyms/toolathlon/run.py \
  --resume 20260821T180000Z-0123abcd \
  --purpose evaluation
```

Resume loads the model, image, GPU, port, timeout, vLLM, step, and eval
settings from the manifest. Benchmark score `false` (and `null`) still counts
as a completed harness episode and is skipped. It is an agent-quality result,
not a reason to repeat the run.

The original one-task smoke interface remains available:

```bash
uv run python gyms/toolathlon/run.py finalpool/find-alita-paper \
  --purpose evaluation \
  --subagent-model "$QWEN" \
  --subagent-gpu 0
```

Use `--reuse-vllm` only when the configured port already serves the expected
model and the runner must not own that external process.

When the OpenAI-compatible server runs on the macOS host and task containers
run through Colima, also pass
`--subagent-base-url http://host.docker.internal:8030/v1`,
`--docker-socket /var/run/docker.sock`, and `--publish-service-ports`.
The socket bind source is resolved inside Colima's Linux VM, while publishing
the gateway and LangGraph ports makes them reachable from macOS loopback.

For several externally managed replicas (for example the MLSpace pool below),
pass their host-local ports as a pool. The batch verifies every endpoint and
assigns episodes round-robin:

```bash
uv run python gyms/toolathlon/run.py --all \
  --purpose evaluation \
  --subagent-ports 18200 18201 18202 18203 \
  --concurrency 16 \
  --n-jobs-per-worker 16
```

Per-episode gateway and subagent-server ports are allocated automatically;
vLLM ports are the only fixed ones.

## MLSpace inference pool

Toolathlon's Docker containers remain on Hertz-2. MLSpace jobs only run Qwen
vLLM replicas and expose them through dedicated SSH reverse tunnels bound to
Hertz-2 loopback. This avoids requiring Docker inside MLSpace.

The reproducible service definitions are in `mlspace_experiments.py`. The
live mapping uses one H100 for the connectivity pilot (port 18199) and
sixteen high-priority single-H100 jobs for the full pool (ports 18200 through
18215). The range is disjoint from the Toolathlon-Gym pool (18099-18115), so
both can run at once. Independent jobs avoid waiting for scarce eight-GPU
shapes and keep one eviction from taking out half the pool. Artifacts, staged
code, service logs, and dedicated tunnel credentials live outside the
checkout at:

```text
/mnt/shared_ru.ml.SZ-5_000264/matrosov/decomposer-toolathlon-bench-artifacts/
  code/<git-commit>/
  inference/<experiment>/
  secrets/
```

Run the zero-GPU payload validation from the configured OCC environment:

```bash
conda run -n decomposer_jobs python \
  gyms/toolathlon/run_mlspace_inference_jobs.py \
  --full --author-name matrosov --telegram-nick js0n_statham --dry
```

Submit `--pilot` first and verify port 18199 from Hertz-2. Submit `--full`
only after that succeeds.

## Artifacts and resume behavior

All output is under `artifacts/gyms/toolathlon/`:

```text
runs/<run-id>/
  manifest.json                 # task/repetition status, score, paths, timings, errors
  metrics.json                  # pass@1, pass@3, pass^3, and score coverage
  events.jsonl                  # append-only raw lifecycle events
  vllm-<port>.log               # shared vLLM logs, appended on resume
  container.lock
  attempts/<domain>/<task>/rep-NNN/attempt-NNN/
    attempt.json
    runner.stdout.log
    runner.stderr.log
traces/<domain>/<task>/<episode-id>/
  traj_log.json                 # benchmark-format trajectory (shared with container)
  trace.json                    # full Decomposer + subagent trace
  answer.txt
  preprocess.log
  gateway.log
  subagent_server.log
  subagent_model_calls.jsonl    # fsync'd request/response records with usage
  usage.json                    # teacher, per-subagent, and total token usage
  eval.log
  container.log
  task.inspect.json
  task_bundle.json              # trusted bundle, published after evaluation
  cleanup.json
  workspace/...                 # agent workspace, shared with the container
  eval_res.json                 # evaluator output, read back from the container
stashes/<domain>/<task>/<episode-id>/
  (host-private stash of evaluator artifacts; removed after restore)
evals/<domain>/<task>/<episode-id>/result.json
logs/<domain>/<task>/<episode-id>/vllm.log
```

The per-attempt `runner.stdout.log` and `runner.stderr.log` files contain the
complete runner output. `trace.json` contains every Decomposer message and
the full message history for each subagent run, plus the effective context,
recursion, provider, endpoint, and reasoning settings.

The manifest is atomically replaced after every state transition. Each
task/repetition has its own entry and each retry has a distinct attempt log
directory. Existing trace/evaluation directories are never overwritten. A
batch runs up to `--concurrency` episodes at once, creates a fresh task
container for each one (all on `--network host`, with unique per-episode
gateway and subagent ports), and removes it in the worker's `finally` block.
Container startup and cleanup are briefly serialized through
one lock per `--container-slots` slot (default 1); two slots are the validated
fast rootless-Podman setting on Hertz-2. vLLM startup is shared. The runner
also supplies canonical Podman short-name aliases so cached Kubernetes images
load into Kind without registry pulls. Each subagent server starts
LangGraph with `--n-jobs-per-worker` slots (default 16). The supervised
vLLM is started once before the worker pool and stopped once after it.
Kubernetes episodes are serialized because each creates a Kind cluster and
the default host inotify budget cannot safely support several concurrently;
their task-provided stop scripts delete the cluster after native evaluation.

## Runtime boundary

- Host: selected simple/Decomposer harness, its credentials, vLLM servers,
  artifact stash, trajectory and evaluation records.
- Task container: task preprocessing, MCP gateway, LangGraph subagent
  server, MCP server processes, workspace, and native evaluation.

Evaluation records use the same episode identifier as their traces.
