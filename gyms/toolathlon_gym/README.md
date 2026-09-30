# Toolathlon Gym

Reusable environment execution only: task setup, container lifecycle, model/tool
loops, native result checking, cleanup, and raw artifacts.

## Run

From the repository root:

Ensure `docker` resolves to your configured runtime. On Hertz with the user-local
Podman wrapper, first run `export PATH="$HOME/.local/bin:$PATH"`.

```bash
PYTHONPATH=src:. python -m gyms.toolathlon_gym.run \
  --tasks task-a task-b --harness decomposer --concurrency 8 -n 1
```

Use `--harness react` for the same tool-equipped worker acting directly on the
task, without a Decomposer. Its model is selected by `--subagent-api-model`.
Use a positional task for one task, `--tasks` for a subset, or `--all`.
The fixed executor has no coverage policy, culling, adaptive retries or resume.

Both roles use `create_model(model)` from [models.py](../../src/decomposer/models.py).
Each entry declares its provider, URL, sampling parameters and reasoning behavior
directly. Set `LLM_PROXY_MASTER_KEY` before creating models; the runner passes the
key into the task container by its environment variable name, without storing it
in artifacts. Sampling does not depend on environment variables.

| Role | Deployment | Generation Settings |
| --- | --- | --- |
| Teacher | `Qwen/Qwen3.8-Flash-Next-NVFP4` | Non-thinking; temperature 0.7, top-p 0.8, top-k 20, min-p 0, presence penalty 1.5, repetition penalty 1 |
| Subagent / ReAct | `Qwen/Qwen3.5-4B-unlooped` | Thinking; temperature 0.6, top-p 0.95, top-k 20; other sampling parameters use defaults |

The subagent profile follows the checkpoint's `SAMPLING.md` and `eval_sampling.yaml`.
Worker reasoning is saved and replayed between tool calls. Teacher settings follow
the [official non-thinking profile](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#api-usage).
For a private inference network, follow the [shared lmrouter setup](../../README.md#hosted-models-and-private-lmrouter-access).
The same factory runs on the host and inside Docker through the mounted socket.

Defaults: 45-minute agent timeout, 55-minute total episode timeout, recursion
limit 410. Decomposer uses upstream's `new / fork / run / wait` interface.

## Raw Outputs

`--output-dir` defaults to `artifacts/gyms/toolathlon_gym/raw`. Each execution
gets a new run ID with:

- `manifest.json`: selected tasks, repetitions, completion and process outcomes.
- `traces/<task>/<episode>/`: raw messages, subagent histories, usage, answer,
  task workspace and cleanup records.
- `evals/<task>/<episode>/result.json`: native evaluator output.
- `logs/<task>/<repetition>/`: process stdout and stderr.

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
use `/opt/subagents`. Rebuild the adapter after changing packaged code.

## Workflows

- [SFT collection](../../sft/toolathlon_gym/README.md): resumable coverage-first scheduling.
- [Evaluation](../../evals/toolathlon_gym/README.md): fixed-sample runs and metrics.

Workflows depend on this Gym, never the reverse.

Python MCP servers launch directly from their preinstalled per-project virtual
environments. Task startup never resolves or rebuilds those dependencies. A missing
executable is an image-build problem and fails explicitly. Numerical thread pools
are capped to one thread in the runtime image.
