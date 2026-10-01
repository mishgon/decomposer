# WideSeek Evaluation

`run.py` reuses the raw Gym runner and adds aggregate native scores, episode
timings, agent tokens and completion/error counts. Judge errors remain unscored;
the explicitly named `mean_native_score_infra_zero` also counts them as zero.

```bash
source gyms/wideseek/env.sh
.venv/bin/python -m evals.wideseek.run --harness react \
  --output artifacts/evals/wideseek/runs/qwen4b-simple \
  --limit 100 -n 3 --concurrency 2
```

Repeat with `--harness decomposer` and a separate output directory for comparison.
Both default to the same non-thinking Qwen4B policy and subagents. Registry profile
overrides are explicit CLI arguments; provider and sampling settings stay in
`src/decomposer/models.py`. Native per-episode judging belongs to the Gym.

`sequence.py` schedules the two setups sequentially and stops on process failure.
`rescore.py` scores saved answers with a selected registry judge into a new output
directory, preserving every original result. Use each module's `--help`.

```bash
bash evals/wideseek/watch-wideseek.sh
```

The watcher selects the latest individual raw, evaluation or SFT run. Pass its
run name to select one explicitly. It shows completed attempts, normal finishes,
early stops, unscored results, mean native score, worker counts and whole-run ETA.
Historical combined-run artifacts remain readable.

For adaptive SFT collection, the watcher shows live task coverage, exhausted
tasks, current phase and wave progress. ETA covers the current wave; it does not
pretend to know how many future attempts will succeed.

See [Gym setup and scoring](../../gyms/wideseek/README.md) and
[SFT collection](../../sft/wideseek/README.md). Table item-F1 and development-set
results are not binary benchmark pass rates or public WideSearch scores.
