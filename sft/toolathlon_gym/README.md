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
  --subagent-base-url https://lmrouter.2a2i.org/v1
```

Teacher and workers: `LLM_PROXY_MASTER_KEY`, using the explicit entries in
[models.py](../../src/decomposer/models.py).
No local GPU server is started for hosted workers. Add
`--subagent-host hostname:IP` if containers need an explicit DNS mapping.
Shared tunnel tools live under `scripts/lmrouter/`. The Gym image refresh remains
under `inference/`.

### Private Router Tunnel

Follow [the shared setup instructions](../../README.md#hosted-models-and-private-lmrouter-access)
to create the restricted SSH key and start the tunnel and relay once per host:

```bash
python scripts/lmrouter/socket_relay.py \
  --socket "$HOME/.local/share/lmrouter-relay/router.sock" --port 18443
```

Set `LLM_PROXY_UNIX_SOCKET` to that path. The runner mounts its directory
read-only into task containers. HTTP clients still use the original HTTPS URL
and verify its certificate; the relay never decrypts traffic or stores keys.
The directory should contain only the relay socket. Run the relay separately
from collection and keep it alive when resuming.

`collect_hosted.sh` selects Flash Next non-thinking and Qwen4B-unlooped thinking
with native tool calling. It loads credentials from `LMROUTER_ENV` and requires
a validated `COLLECTION_IMAGE` plus `LLM_PROXY_UNIX_SOCKET`. Example:

```bash
bash sft/toolathlon_gym/collect_hosted.sh --all --adaptive -n 1 --concurrency 8
```

The two registry entries use native tool calling. The runner has no automatic
provider fallback. Old traces keep their recorded settings; start a new run when
changing generation profiles to keep collections comparable.

`inference/Dockerfile.refresh` can refresh application code on a pinned,
previously validated Python 3.12 runtime image without downloading dependencies.
It preserves that image's system packages and task-service patches. Record both
the base image ID and the resulting image ID when using it.

## Coverage Policy

Each wave launches one attempt per task with no qualifying trace. A native pass
or partial score strictly above 0.90 qualifies, provided the agent finished.
Scores from agent errors/timeouts are diagnostic only. Check-marker logs without
a successful evaluator exit or explicit totals remain unscored.
After six unsuccessful launches, a task is culled, including tasks with
unparseable evaluations. Missing scores remain unknown, not fabricated zeros.
Once coverage is exhausted, the scheduler balances retained tasks toward four
qualifying traces, with at most 24 additional attempts per task in that phase.
The target is not guaranteed: the run stops when the budgets are exhausted.

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
