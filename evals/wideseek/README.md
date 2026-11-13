# WideSeek Evaluation

`python -m evals.wideseek.paired --panel /path/to/frozen-opd-data
--output /path/to/new-evaluation --worker-python /path/to/worker/python`
compares hosted non-thinking Qwen3.5-4B unlooped and Flash Next coordinators on
the same frozen 16 training and 8 held-out tasks, once each. Both use the same
thinking Qwen4B researchers, native judge and sampling registry. No GPU is required
for inference; the existing Wikipedia retrieval service must be available.

The paper's normal 10-turn lead / 20-turn researcher paths map to recursion limits
32 / 81 with our middleware. Boundary tests include the final graph termination
check. These are graph-step budgets, not exact model-call caps: coordinator retry
edges can skip tool nodes. Each researcher run retains its 60-second work budget
and 60-second grace. The episode safety timeout remains 2,700 seconds, including
judging. This does not reproduce the paper's parallel-tool or parallel-agent caps.
Raw results remain under `student/` and `teacher/`; `summary.json` updates after
each episode with separate train and held-out scores, times and token counts.
Judge failures remain unscored. This evaluation does not change OPD settings.

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

## Teacher-Solved Depth Evaluation

`teacher_solved_local` freezes every depth task with a saved Flash Next QA score
of 1 in the supplied collection runs. It excludes the retired `spawn_subagent`
interface, deduplicates task IDs, and shuffles with seed 42. The checkpoint must
already be merged; retrieval and hosted researcher/judge credentials must be
configured as for WideSeek OPD.

```bash
python -m evals.wideseek.teacher_solved_local \
  --source /shared/data/depth/tasks.jsonl \
  --teacher-runs /shared/collections/depth-pass1 /shared/collections/depth-pass2 \
  --model /shared/models/merged-sft --output /shared/evals/depth-n3 \
  --worker-python /shared/runtime/workers/bin/python \
  --gpu 0 --concurrency 128 -n 3 --episode-timeout 2700
```

The launcher owns one loopback-only vLLM server (a free port, 32K context) and a
dedicated researcher server. It uses the existing Qwen non-thinking sampling
profile and collector outage/cleanup recovery. `--resume` keeps the frozen task
set and completed attempts. `data/selection.json` records teacher evidence;
`eval/` contains raw traces and scores. `task-success-counts.json` records each
task's 0–3 success count when the run exits. Unscored infrastructure failures
remain unresolved rather than entering the zero-success bucket.
