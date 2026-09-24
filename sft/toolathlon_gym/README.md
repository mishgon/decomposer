# Toolathlon Gym SFT Collection

Off-policy collection for the upstream four-tool Decomposer harness. This layer
owns manifests, resume, coverage scheduling, culling and provider backoff.
It invokes the shared Gym episode runner and process supervisor.

## Launch

Load router credentials into the environment without placing keys in commands.
Confirm exact deployment IDs and perform a one-task smoke before a full run.

```bash
PYTHONPATH=src:. python -m sft.toolathlon_gym.run \
  --all --adaptive -n 1 --concurrency 8 \
  --model "$TEACHER_MODEL" \
  --subagent-model "$SUBAGENT_MODEL" --subagent-api-model "$SUBAGENT_MODEL" \
  --subagent-base-url "$LLM_PROXY_URL"
```

Teacher: `LLM_PROXY_URL` + `LLM_PROXY_MASTER_KEY`.
Workers: hosted URL above, with `VLLM_API_KEY` or the same router key.
No local GPU server is started for hosted workers. Add
`--subagent-host hostname:IP` if containers need an explicit DNS mapping.
Infrastructure-specific inference launchers are isolated under `inference/`.

## Coverage Policy

Each wave launches one attempt per task with no qualifying trace. A native pass
or partial score strictly above 0.90 qualifies. After six unsuccessful launches,
a task is culled; tasks with unparseable evaluations are protected from culling.
Once coverage is exhausted, the scheduler balances retained tasks toward four
qualifying traces. This is bounded for scored zero-success tasks, not a fixed
total attempt count. Inspect unscored failures rather than letting them retry
indefinitely.

Use `--adaptive-target-successes 1` for coverage only. Existing CLI threshold and
target overrides remain available. Failed traces and all native outputs are kept.

## Artifacts and Resume

The default root is `artifacts/sft/toolathlon_gym`:

- `runs/<run-id>/manifest.json`, event log and per-attempt stdout/stderr.
- `traces/<task>/<episode>/`: raw traces, model logs and task workspaces.
- `evals/<task>/<episode>/result.json`: native scores used by scheduling.

```bash
PYTHONPATH=src:. python -m sft.toolathlon_gym.run --resume RUN_ID --adaptive
```

For an existing artifact root, also pass `--gym-artifacts-dir PATH`.
Completed attempts are not repeated on resume. Do not mix pre-migration
`spawn_subagent` traces and new-harness traces under a new collection identity.
