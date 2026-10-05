# Toolathlon Gym

Reusable environment execution only: task setup, container lifecycle, model/tool
loops, native result checking, cleanup, and raw artifacts.

## Run

From the repository root:

Ensure `docker` resolves to your configured runtime. On Hertz with the user-local
Podman wrapper, first run `export PATH="$HOME/.local/bin:$PATH"`.

```bash
PYTHONPATH=src:. python -m gyms.toolathlon_gym.run \
  --tasks task-a task-b --agent decomposer --concurrency 8 -n 1
```

Use `--agent qwen_3_5_4b_thinking` for the same tool-equipped worker acting directly on the
task, without a Decomposer. Its model is configured in `agents.py`.
Use a positional task for one task, `--tasks` for a subset, or `--all`.
The fixed executor has no coverage policy, culling, adaptive retries or resume.

Both roles use `create_model(model_id)` from [models.py](../../src/decomposer/models.py).
Each entry declares its provider, URL, sampling parameters and reasoning behavior
directly. Set `LLM_PROXY_MASTER_KEY` for lmrouter or `OPENROUTER_API_KEY` for
OpenRouter. The runner passes keys into the task container by environment variable
name, without storing them in artifacts. Local vLLM requires neither key.
Sampling does not depend on environment variables.

Choose models in [agents.py](agents.py); their sampling settings are defined in
[models.py](../../src/decomposer/models.py). The container records its model IDs
and settings in `runtime.json`, which the runner copies into `trace.json`.

Defaults: 45-minute agent timeout, 55-minute total episode timeout, recursion
limit 410. Decomposer uses upstream's `new / fork / run / wait` interface.

## Raw Outputs

`--output-dir` defaults to `artifacts/gyms/toolathlon_gym/raw`. Each execution
gets a new run ID with:

- `manifest.json`: selected tasks, repetitions, completion and process outcomes.
- `traces/<task>/<episode>/`: raw messages, agent histories, usage, answer,
  task workspace and cleanup records.
- `evals/<task>/<episode>/result.json`: native evaluator output.
- `logs/<task>/<repetition>/`: process stdout and stderr.

After saving the native evaluation, Decomposer episodes generate `trace.html`
next to `trace.json` using `decomposer.visualization.write_trace_html`.
A visualization error produces a warning and preserves the episode's result.

An agent timeout/model failure retains partial messages, cancels active remote
runs and waits for them to stop before native evaluation. This also covers ReAct
runs and runs launched just before a checkpoint was interrupted. If shutdown
cannot be confirmed within 60 seconds, the evaluator is skipped and the attempt
fails with a saved shutdown error. Its diagnostic score does not turn the attempt into a success.
A hard process kill or infrastructure failure can still prevent evaluation.
Missing native scores remain missing; the Gym does not invent an aggregate score.
Container inspection artifacts contain only allowlisted lifecycle fields, never
the environment or command arguments. Older artifacts may contain credentials;
sanitize them before sharing.

## Build

`gyms/toolathlon_gym/build.sh` builds the task adapter on top of
`toolathlon-pack:latest`. Native tools run in `/opt/venv`; LangGraph workers
use `/opt/agents`. Rebuild the adapter after changing packaged code.

## Workflows

- [SFT collection](../../sft/toolathlon_gym/README.md): resumable coverage-first scheduling.
- [Evaluation](../../evals/toolathlon_gym/README.md): fixed-sample runs and metrics.

Workflows depend on this Gym, never the reverse.

Python MCP servers launch directly from their preinstalled per-project virtual
environments. Task startup never resolves or rebuilds those dependencies. A missing
executable is an image-build problem and fails explicitly. Numerical thread pools
are capped to one thread in the runtime image.

## Agent Server

This container-local LangGraph server exposes two assistants:

- `qwen_3_5_4b_thinking`: a tool-equipped agent.
- `decomposer`: orchestrates agents on the same server.

Model IDs and agent factories are defined in [agents.py](agents.py).
Change its constants to select models, then rebuild the adapter image.
`--agent` selects an assistant ID from `langgraph.json`.
The tool-equipped agent is used for standalone runs and by Decomposer.

When selecting a `vllm/` model, start its server on the host before running tasks.
It must listen on a container-accessible interface at port 8024. With
`vllm_server`, pass `host="0.0.0.0"`; the default binds only to loopback.
The runner sets `VLLM_HOST=host.docker.internal` inside the task container;
it does not start or stop vLLM.

The server reads the prepared task configuration from
`$TOOLATHLON_DATA_DIR/runtime.json`. When it starts, it opens one persistent
stdio session for each required MCP server and shares the loaded tools between
the graphs. It closes every session when it stops.
