# Toolathlon Benchmark

Raw episodes on the hkust-nlp [Toolathlon benchmark](https://github.com/hkust-nlp/Toolathlon)
(`external/toolathlon`, pinned at the Toolathlon-Verified release): task setup,
container lifecycle, model/tool loops, native result checking, cleanup, and raw
artifacts. This is the test benchmark; it is unrelated to Toolathlon Gym.

## Run

From the repository root, with `LLM_PROXY_MASTER_KEY` set:

```bash
PYTHONPATH=src:. python -m gyms.toolathlon_bench.run \
  --tasks find-alita-paper academic-pdf-report --agent decomposer --concurrency 8 -n 1
```

Use `--agent qwen_3_5_4b_thinking` for the same tool-equipped worker acting directly
on the task, without a Decomposer. Use a positional task for one task, `--tasks` for
a subset, or `--all`. Task names are directories under `tasks/finalpool`.

Both roles use `create_model(model_id)` from [models.py](../../src/decomposer/models.py).
Choose models in [agents.py](agents.py). The runner passes the key into the task
container by its environment variable name, without storing it in artifacts.

Defaults: 45-minute agent timeout, 100-minute total episode timeout (Toolathlon
preprocess and native evaluation can each take tens of minutes), recursion limit 410.
Decomposer uses upstream's `new / fork / run / wait` interface.

## Episode

1. The task container starts on the host network, because Toolathlon's local
   services (Canvas, WooCommerce, email) listen on host loopback. The episode
   directory is mounted at `/workspace/dumps`. User credentials from
   `external/toolathlon/configs` are copied in, including the Gmail and Calendar
   MCP OAuth files; missing `global_configs.py` and `token_key_session.py` are
   first created from Toolathlon's examples. The host Docker or Podman socket
   (`DOCKER_HOST`, else `/var/run/docker.sock`) is mounted for Kubernetes tasks.
2. Toolathlon's `container_preprocess` prepares the workspace and a trusted task
   bundle, which the runner keeps in a private host directory. Kubernetes tasks
   get three attempts, each from a deleted cluster.
3. Toolathlon's `task_artifact_guard` moves the evaluator and ground truth out of
   the container.
4. [task.py](task.py) starts the MCP [tool gateway](tool_gateway.py) and the Agent
   Server, then deletes the bundle copy that the gateway read.
5. After the agent loop, the runner stops the Agent Server, writes Toolathlon's
   `traj_log.json`, restores the evaluator and runs Toolathlon's `container_eval`.
   The gateway and the processes agents started through it keep running until
   the container is removed, because some evaluators check them.
6. Cleanup gives the episode directory back to the host user and removes the
   container.

Tasks whose services are not configured fail; see Toolathlon's
`global_preparation/how2register_accounts.md`. Run its
`global_preparation/deploy_containers.sh` for the local service stack.

## Raw Outputs

`--output-dir` defaults to `artifacts/gyms/toolathlon_bench/raw`. Each execution
gets a new run ID with:

- `manifest.json`: selected tasks, repetitions, completion and process outcomes.
- `traces/<task>/<episode>/`: raw messages, agent histories, usage, answer,
  `traj_log.json`, `eval_res.json`, preprocess, gateway and Agent Server logs,
  the task workspace and cleanup records.
- `evals/<task>/<episode>/result.json`: native evaluator output.
- `logs/<task>/<repetition>/`: process stdout and stderr.

`pass` requires a native pass and a normal agent finish. `native_pass` grades the
artifacts even after an agent failure, for diagnostics.

## Build

```bash
gyms/toolathlon_bench/build.sh
```

The image extends Toolathlon's official task image with the pinned benchmark
sources and installs the Agent Server in `/opt/agents`. Rebuild it after changing
packaged code.

## Workflows

- [Evaluation](../../evals/toolathlon_bench/README.md): fixed-sample runs and metrics.

Workflows depend on this benchmark executor, never the reverse.

## Agent Server

This container-local LangGraph server exposes two assistants:

- `qwen_3_5_4b_thinking`: a tool-equipped agent with the task's Toolathlon system
  prompt, which names the workspace directory.
- `decomposer`: orchestrates agents on the same server.

Tools come from the gateway, which exposes the task's MCP servers, `claim_done`,
and the requested Toolathlon local tools. Outputs over 100K characters are
truncated and saved for Toolathlon's overlong-output tools.
