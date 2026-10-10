# WideSeek SFT Trace Collection

The collector reuses `gyms/wideseek/run.py` and saves every trajectory, including
failed and unscored attempts. It adds a relative-path index of successful traces
and a task-coverage summary. It supports fixed attempts or bounded coverage-first
scheduling; it does not train a model or start an indefinite queue.

Defaults use main's named registry profiles:

- Teacher and judge: `lmrouter/qwen_3_8_flash_next_non_thinking`.
- Subagents: `lmrouter/qwen_3_5_4b_unlooped_thinking`, with reasoning preserved.

Sampling and provider settings come directly from `src/decomposer/models.py`.
Model outputs, reasoning, tool calls, usage, subagent states and judge responses
stay under the run's execution directories. Authentication headers are not logged.

Start the services described in [the Gym setup](../../gyms/wideseek/README.md),
then run a bounded smoke:

```bash
source gyms/wideseek/env.sh
gyms/wideseek/.venv/bin/python -m sft.wideseek.run --agent decomposer \
  --student-tokenizer /path/to/student-tokenizer \
  --output artifacts/sft/wideseek/runs/smoke --limit 4 -n 1 --concurrency 2
```

Repeat the same command with `--resume` to skip saved results and retry interrupted
executions in fresh directories. Completed results are never rerun. Task data
and model settings must match the saved manifest. Each attempt records its Git
revision; use a new run directory when comparing harness versions.

## Coverage-First Collection

The policy matches Toolathlon SFT collection. Each wave launches one attempt for
every task without a qualifying trace. Qualification requires a normal finish,
no cleanup error, a saved trajectory and a native score **strictly above 0.90**
(or full score 1.0). Exactly 0.90 does not qualify. Failed/unscored attempts count
toward the six-launch budget; their scores are never invented. Tasks with zero
qualifying traces after six launches are culled.

Once coverage is exhausted, retained tasks are balanced toward four good traces
each. The lowest-success-count tasks run first. Each retained task receives at
most 24 additional attempts; a once-solved task is never culled. Collection stops
when targets or retry budgets are exhausted. Use `--target-successes 1` for
coverage only.

For the full 20,000-task width split, the prepared command is:

```bash
source gyms/wideseek/env.sh
gyms/wideseek/.venv/bin/python -m sft.wideseek.run --agent decomposer --adaptive \
  --student-tokenizer /path/to/student-tokenizer \
  --output artifacts/sft/wideseek/runs/width-coverage \
  --limit 20000 -n 1 --concurrency 2
```

This command is not scheduled automatically. Models, reasoning preservation and
context limits are unchanged from the smoke. Choose concurrency before launch.

The required student tokenizer enables shared `sft/sequence_limit.py` middleware
on every collection run. Above 32,768 student tokens, the teacher stops after its
current response. The attempt is `skipped`, not an error or success; all raw
messages and diagnostic scores remain saved. Batch filtering uses the same
`sft/filtering.py` implementation as Toolathlon collection.

`scheduler.json` saves policy, per-task counts, culling and waves before launches.
Resume the same command with `--resume`: saved results count toward both budgets
and coverage; only unfinished attempts in the saved wave launch again.
`--max-waves 1` runs one bounded validation wave per invocation, then stops.

## Fixed Attempts and Artifacts

The fixed schedule runs each selected task `-n` times. Increase `--limit` to select
more tasks; the prepared width dataset has 20,000 tasks. The collector imposes no
extra sampling or generation limits. Agent timeout is 45 minutes and recursion
limits are 410; the hosted deployment enforces its own context/output limits.

`successful-traces.jsonl` contains normally finished, scored traces with native
score > `--success-threshold` (default 0.9), or full score 1.0, and no worker-cleanup errors.
`collection.json` records successful-trace count and distinct-task coverage.
The native table metric is item-F1, not a binary benchmark pass rate. These
development-set scores are not directly comparable with public WideSearch results.

Use `bash evals/wideseek/watch-wideseek.sh` to inspect the latest collection or
evaluation. Fixed runs show whole-run ETA; adaptive runs show current-wave ETA
and live coverage. Whole-collection ETA is unknown because later waves depend
on scores. Index and coverage summaries refresh after each wave.
