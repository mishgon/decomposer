# Toolathlon-Gym

This gym integration keeps Decomposer and its model credentials on the host while a
task container owns the Toolathlon workspace, MCP servers, and evaluator.
Host vLLM servers remain long-lived services outside the task container.

## Image layout

The adapter image extends `toolathlon-pack:latest`, preserving Toolathlon's
existing `/opt/venv` and prebuilt MCP servers. It adds a separate Python 3.12
environment at `/opt/subagents` for the LangGraph subagent server.
The two environments are intentionally isolated.

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

The container resolves host vLLM servers through these variables:

- `GEMMA_4_E2B_BASE_URL`
- `GEMMA_4_E4B_BASE_URL`
- `GEMMA_4_12B_BASE_URL`
- `GEMMA_4_26B_A4B_BASE_URL`

They default to the corresponding ports on `host.docker.internal`.

Start the four vLLM servers first, then run an episode from the host:

```bash
export OPENROUTER_API_KEY=...
uv run python gyms/toolathlon_gym/run.py howtocook-event-menu-ppt
```

The runner checks the vLLM endpoints, creates an isolated Docker network and
PostgreSQL container, starts the task container, runs Decomposer, evaluates the
result inside the live task container, and removes all temporary Docker
resources. It writes `runtime.json`, `trace.json`, `answer.txt`, and
`container.log` under `artifacts/data/toolathlon_gym/<task>/<episode-id>/`, and
the native evaluation result under
`artifacts/evals/toolathlon_gym/<task>/<episode-id>/result.json`.

## Runtime boundary

- Host: Decomposer, OpenRouter credentials, and vLLM servers.
- Task container: task preprocessing, LangGraph subagent server, MCP server
  processes, workspace, and native evaluation.
- Episode network: an isolated Toolathlon PostgreSQL container.

Evaluation records use the same episode identifier as their traces.
