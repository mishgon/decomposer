# WideSeek Evaluation

Run one setup with `python -m evals.wideseek.run --harness react` (or
`decomposer`), plus `--output`, `--limit`, `-n`, and `--concurrency`.
The runner reuses `gyms.wideseek.run` and adds aggregate scores. Native
per-episode judging remains in the Gym; scoring rules are unchanged.

Use `python -m evals.wideseek.sequence --help` to schedule separate setups,
or `python -m evals.wideseek.rescore --help` to judge saved answers without
rerunning agents. The watcher still reads individual run manifests and results.

For raw execution without aggregate summaries, use
`python -m gyms.wideseek.run --harness react --output PATH`.
The old `--mode simple|decomposer` spelling is accepted; artifact mode names
stay unchanged. Existing source-hash resume guards still reject changed code.
