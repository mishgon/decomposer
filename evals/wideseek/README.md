# WideSeek Evaluation

`run.py` reuses the raw Gym runner and adds aggregate native scores, episode
timings, agent tokens and completion/error counts. Judge errors remain unscored;
the explicitly named `mean_native_score_infra_zero` also counts them as zero.
The aggregate is saved to `summary.json` in the run directory.

```bash
source gyms/wideseek/env.sh
gyms/wideseek/.venv/bin/python -m evals.wideseek.run --agent react \
  --output artifacts/evals/wideseek/runs/qwen4b-simple \
  --limit 100 -n 3 --concurrency 2
```

Use `--agent decomposer` with a separate output directory for Flash Next plus
thinking Qwen4B researchers. This is a different-model comparison, not a
same-model harness ablation. Configurations live in `gyms/wideseek/agents.py`;
provider and sampling settings stay in `src/decomposer/models.py`.
Native per-episode judging belongs to the Gym.

Run the two configurations separately, using distinct output directories.
`rescore.py` scores saved answers with a selected registry judge into a new output
directory, preserving every original result. Use each module's `--help`.

```bash
bash evals/wideseek/watch-wideseek.sh
```

The watcher selects the latest individual raw, evaluation or SFT run. Pass its
run name to select one explicitly. It shows completed attempts, normal finishes,
early stops, unscored results, mean native score, worker counts and whole-run ETA.

For adaptive SFT collection, the watcher shows live task coverage, exhausted
tasks, current phase and wave progress. ETA covers the current wave; it does not
pretend to know how many future attempts will succeed.

See [Gym setup and scoring](../../gyms/wideseek/README.md) and
[SFT collection](../../sft/wideseek/README.md). Table item-F1 and development-set
results are not binary benchmark pass rates or public WideSearch scores.
