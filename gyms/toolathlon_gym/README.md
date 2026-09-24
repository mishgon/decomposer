# Toolathlon Gym

Reusable environment execution only: task setup, container lifecycle, model/tool
loops, native result checking, cleanup, and raw artifacts.

## Run

From the repository root:

```bash
PYTHONPATH=src:. python -m gyms.toolathlon_gym.run \
  --tasks task-a task-b --harness decomposer --concurrency 8 -n 1 \
  --model "$TEACHER_MODEL" \
  --subagent-model "$SUBAGENT_MODEL" --subagent-api-model "$SUBAGENT_MODEL" \
  --subagent-base-url "$LLM_PROXY_URL"
```

Use `--harness react` for the same tool-equipped worker acting directly on the
task, without a Decomposer. Its model is selected by `--subagent-api-model`.
Use a positional task for one task, `--tasks` for a subset, or `--all`.
The fixed executor has no coverage policy, culling, adaptive retries or resume.

The teacher uses `LLM_PROXY_URL` and `LLM_PROXY_MASTER_KEY` when configured.
Hosted workers use `--subagent-base-url` and `VLLM_API_KEY` (falling back to
`LLM_PROXY_MASTER_KEY`). Optional `--subagent-host hostname:IP` supplies a
container DNS mapping without disabling TLS verification. Without a hosted
worker URL, the runner starts one shared local vLLM server.

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

An interrupted/model-failed episode retains partial messages and logs. Missing
native scores remain missing; the Gym does not invent an aggregate score.

## Build

`gyms/toolathlon_gym/build.sh` builds the task adapter on top of
`toolathlon-pack:latest`. Native tools run in `/opt/venv`; LangGraph workers
use `/opt/subagents`. Rebuild the adapter after changing packaged code.

## Workflows

- [SFT collection](../../sft/toolathlon_gym/README.md): resumable coverage-first scheduling.
- [Evaluation](../../evals/toolathlon_gym/README.md): fixed-sample runs and metrics.

Workflows depend on this Gym, never the reverse.
