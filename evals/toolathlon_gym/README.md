# Toolathlon Gym Evaluation

Fixed-sample evaluation reuses the Gym executor. It does not import the SFT
scheduler. This is Toolathlon Gym, not the separate Toolathlon benchmark.

```bash
PYTHONPATH=src:. python -m evals.toolathlon_gym.run \
  --tasks task-a task-b --harness react -n 3 --concurrency 8 \
  --subagent-api-model "$SUBAGENT_MODEL" --subagent-base-url "$LLM_PROXY_URL"
```

The other model and environment options match `gyms.toolathlon_gym.run`.
Default output: `artifacts/evals/toolathlon_gym/<run-id>`, containing raw outputs
plus `summary.json`.

Aggregate an existing fixed run without rerunning it:

```bash
PYTHONPATH=src:. python -m evals.toolathlon_gym.summary PATH_TO_RUN
```

Metrics include pass@k and pass^k for k=1,3,5 when enough repetitions exist.
Missing attempts, timeouts and infrastructure failures count as failures.
Only normal finishes with a native pass count as strict successes. Native partial
scores are reported separately. `--denominator N` treats unselected tasks as
failures and must be at least the selected-task count.

These fixed-sample estimates must not be applied to adaptively stopped SFT
attempts. Use `python -m evals.toolathlon_gym.trace_stats PATH` for descriptive
trace/usage totals on arbitrary collections.
