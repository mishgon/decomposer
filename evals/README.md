# Evaluations

One command per gym runs an evaluation on top of `gyms/<gym>` and computes its
metrics. The gym code runs the rollouts and saves the raw results. `evals/<gym>`
reuses it unchanged: `run.py` accepts every flag of the gym runner (experiment,
all or some tasks, repeats, checkpoint, ...), runs it, and writes
`<run_dir>/eval_metrics.json`. A run that is already complete is not repeated,
and `--metrics-only <run_dir>` recomputes metrics without a GPU.

| gym | one command | also here |
|---|---|---|
| tau2 | `python -m evals.tau2_gym.run --experiment ...` | `analyze_traces.py` (decomposition and parallelism statistics) |
| Workplace Assistant | `python -m evals.workplace_assistant.run --experiment ...`; MLSpace: `evals.workplace_assistant.submit` | `comparison.py` (Qwen3.6 vs DeepSeek teacher), `migrate_call_limit_artifacts.py` |
| GAIA2 | `python -m evals.gaia2.run --experiment ...`; MLSpace: `evals.gaia2.submit` | `comparison.py` (held-out baselines), `trace_stats.py` (cost, tokens, latency, parallelism), `chat_trace.py` (one rollout as a timeline) |

`evals/common.py` holds the shared pass metrics: for tasks that each ran `k`
times, `pass_at_1` is the share of passing rollouts, `pass_at_k` the share of
tasks passed at least once, and `pass_pow_k` the share passed on every repeat.

Evaluations only add files next to a run (`eval_metrics.json`, `comparison.json`).
They never rewrite the gym's own markers (`.eval_done.json`, `run_status.json`),
which OPD, SFT adapters and skip logic read.
