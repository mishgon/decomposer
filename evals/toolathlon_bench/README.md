# Toolathlon Benchmark Evaluation

Fixed-sample evaluation reuses the benchmark executor in `gyms/toolathlon_bench`.
This is the Toolathlon benchmark, not Toolathlon Gym.

```bash
PYTHONPATH=src:. python -m evals.toolathlon_bench.run \
  --all --agent qwen_3_5_4b_thinking -n 3 --concurrency 8
```

Models are configured in `gyms/toolathlon_bench/agents.py`.
The environment options match `gyms.toolathlon_bench.run`.
Default output: `artifacts/evals/toolathlon_bench/<run-id>`, containing raw outputs
plus `summary.json`.

Aggregate an existing fixed run without rerunning it:

```bash
PYTHONPATH=src:. python -m evals.toolathlon_bench.summary PATH_TO_RUN
```

Metrics include pass@k and pass^k for k=1,3,5 when enough repetitions exist.
Missing attempts, timeouts and infrastructure failures count as failures.
Only normal finishes with a native pass count as successes. `--denominator N`
treats unselected tasks as failures and must be at least the selected-task count.
Use `--denominator 108` to report a subset against the full benchmark.

Many tasks need app credentials or Toolathlon's local service stack (see
`gyms/toolathlon_bench/README.md`). These 22 tasks need no credentials:

```bash
PYTHONPATH=src:. python -m evals.toolathlon_bench.run --denominator 108 -n 3 --tasks \
  arrange-workspace cooking-guidance courses-ta-hws detect-revised-terms dietary-health \
  excel-data-transformation excel-market-research find-alita-paper \
  imagenet interview-report invoice-org latex-prompt-box paper-checker ppt-analysis \
  privacy-desensitization reimbursement-form-filler sales-accounting shopping-helper \
  stock-build-position travel-exchange university-course-selection yahoo-analysis
```

Use `python -m evals.toolathlon_bench.trace_stats PATH` for descriptive trace and
usage totals.
