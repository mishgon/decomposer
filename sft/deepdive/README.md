# DeepDive Trace Collection

Prepare the [DeepDive gym](../../gyms/deepdive/README.md) and start or reuse your own
WideSeek retrieval service. No GPU or second corpus/index is needed for collection.
The shared retrieval service must remain alive for the duration of the job.

From the repository root, with `LLM_PROXY_MASTER_KEY` supplied securely:

```bash
source gyms/wideseek/env.sh
python -m sft.deepdive.launch --agent decomposer \
  --output artifacts/sft/deepdive/runs/coverage-001
```

The launcher starts dedicated researcher workers on port 18085 with 512 worker
slots. Episode concurrency defaults to 128, independently of worker slots.
It collects one attempt for every task in `qa_rl` and stops after that pass.
Use `--limit 8 --concurrency 2` for a small smoke run, `--worker-port` to select
another free port, or `--worker-python /path/to/python` when workers use a separate
environment. It shuts down its workers when collection exits and never manages
the shared retrieval service.

To use already running workers instead:

```bash
python -m sft.deepdive.run --agent decomposer \
  --worker-url http://127.0.0.1:18085 \
  --output artifacts/sft/deepdive/runs/coverage-002
```

All attempts retain raw traces and scores. At completion, `successful-traces.jsonl`
indexes only normally finished, correctly answered episodes with captured traces
and successful worker cleanup. The inherited threshold is 0.9; because QA scores
are binary, only score 1 qualifies. `collection.json` records the coverage count.
Failures and low scores remain in the run directory for later solvability analysis.
No task filtering or 32K trace truncation happens during preparation or collection.

The shared collector also supports `--resume` with identical settings/revision,
and `--adaptive --target-successes 1` for its existing bounded coverage-retry policy.
Adaptive retries are opt-in; the default is one pass. Never run two writers against
the same output directory. Commit reviewed changes before launching experiments.
