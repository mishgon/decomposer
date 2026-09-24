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

### Private Router Tunnel

When inference is reachable only through an SSH tunnel, expose its loopback
listener as a private Unix socket:

```bash
python -m sft.toolathlon_gym.inference.socket_relay \
  --socket "$HOME/.local/share/lmrouter-relay/router.sock" --port 18443
```

Set `LLM_PROXY_UNIX_SOCKET` to that path. The runner mounts its directory
read-only into task containers. HTTP clients still use the original HTTPS URL
and verify its certificate; the relay never decrypts traffic or stores keys.
The directory should contain only the relay socket. Run the relay separately
from collection and keep it alive when resuming.

`collect_hosted.sh` selects Flash Next thinking and Qwen4B-unlooped non-thinking
with native tool calling. It loads credentials from `LMROUTER_ENV` and requires
a validated `COLLECTION_IMAGE` plus `LLM_PROXY_UNIX_SOCKET`. Example:

```bash
bash sft/toolathlon_gym/collect_hosted.sh --all --adaptive -n 1 --concurrency 8
```

This profile disables the legacy XML workaround and OpenRouter fallback.
`DECOMPOSER_PARSE_QWEN_XML=0` selects native tool calling in the generic runner;
the compatibility default remains unchanged for older deployments.

`inference/Dockerfile.refresh` can refresh application code on a pinned,
previously validated Python 3.12 runtime image without downloading dependencies.
It preserves that image's system packages and task-service patches. Record both
the base image ID and the resulting image ID when using it.

## Coverage Policy

Each wave launches one attempt per task with no qualifying trace. A native pass
or partial score strictly above 0.90 qualifies, provided the agent finished.
Scores from agent errors/timeouts are diagnostic only. Check-marker logs without
a successful evaluator exit or explicit totals remain unscored.
After six unsuccessful launches,
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

A run lock rejects a second collector for the same run ID. After a crash, resume
recovers saved `attempt.json` records before scheduling work. Do not delete the
lock file: the OS releases its lock when the collector exits.
