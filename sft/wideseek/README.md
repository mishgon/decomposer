# WideSeek SFT Trace Collection

The collector reuses `gyms/wideseek/run.py` and saves every trajectory, including
failed and unscored attempts. It adds a relative-path index of successful traces
and a task-coverage summary. It does not train a model or start an indefinite queue.

Defaults use main's named registry profiles:

- Teacher and judge: `qwen_3_8_flash_next_non_thinking`.
- Subagents: `qwen_3_5_4b_unlooped_thinking`, with reasoning preserved.

Sampling and provider settings come directly from `src/decomposer/models.py`.
Model outputs, reasoning, tool calls, usage, subagent states and judge responses
stay under the run's execution directories. Authentication headers are not logged.

Start the services described in [the Gym setup](../../gyms/wideseek/README.md),
then run a bounded smoke:

```bash
source gyms/wideseek/env.sh
.venv/bin/python -m sft.wideseek.run --harness decomposer \
  --output artifacts/sft/wideseek/runs/smoke --limit 4 -n 1 --concurrency 2
```

Repeat the same command with `--resume` to skip saved results and retry interrupted
executions in fresh directories. Completed results are never rerun. Source/data
and model-setting changes require a new run directory.

The fixed schedule runs each selected task `-n` times. Increase `--limit` to select
more tasks; the prepared width dataset has 20,000 tasks. The collector imposes no
extra sampling or generation limits. Agent timeout is 45 minutes and recursion
limits are 410; the hosted deployment enforces its own context/output limits.

`successful-traces.jsonl` contains normally finished, scored traces with native
score >= `--success-threshold` (default 0.9) and no worker-cleanup errors.
`collection.json` records successful-trace count and distinct-task coverage.
The native table metric is item-F1, not a binary benchmark pass rate. These
development-set scores are not directly comparable with public WideSearch results.

Use `bash evals/wideseek/watch-wideseek.sh` to inspect the latest collection or
evaluation. Its ETA covers the selected fixed run.
