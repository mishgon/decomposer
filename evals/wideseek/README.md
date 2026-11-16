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

## RL Band of a Manager Checkpoint

`--agent student` runs an SFT manager checkpoint as the Decomposer on WideSeek or
DeepDive. Serve the checkpoint with vLLM on port 8024 under the name `Qwen/Qwen3.5-4B`
(`--tool-call-parser qwen3_xml`). The profile samples it non-thinking as RL rollouts
do: temperature 1.0 with no top-p, top-k or presence penalty, and at most 8,192 tokens
per turn. All setups use the same `researcher` graph: non-thinking `occ-4b` with the
colleagues' `search`, `open`, `find`, and `finish` tools. `react` runs that researcher directly.
Researchers have 180 seconds of work and 180 seconds to submit through `finish`.
There is no turn limit. The original protocol guards malformed/repeated calls and
retains the 65,536-token context/output limits and 4,096-token per-call limit.
The Search API uses the `wikipedia` source, `AGENTIC_RAG_API_KEY`, and
`AGENTIC_RAG_URL` (default `https://lmrouter.2a2i.org/rag`).
The retired `student_180` and `student_rag` variants are replaced by `student`.

Collect attempts with the SFT collector in two stages. Run 2 attempts per task (`-n 2`),
then write the band so far and the tasks whose two attempts agree:

```bash
gyms/wideseek/.venv/bin/python -m evals.wideseek.band <first_run> \
  --undecided undecided.jsonl --data <pool.jsonl>
```

Run 2 more attempts on `undecided.jsonl`, then write the band of both runs:

```bash
gyms/wideseek/.venv/bin/python -m evals.wideseek.band <first_run> <second_run>
```

`band.json`, in the last run, lists the tasks solved on some attempts but not all, plus
the passes of every task over all given runs. A pass is an attempt the collector counts
as a success. Timeouts, context overflows and runaway turns are failed attempts; outage
failures are rerun, not counted.

Judge errors, infrastructure failures, failed cleanup/export and unknown errors
are excluded from both passes and attempts. `band.json` records these results in
`unresolved`. Before classifying a task, the selector requires at least the largest
`-n` requested for that task across the supplied runs. Tasks below that minimum
appear in `rerun`, with their shortfalls in `missing_attempts`; they also remain
in `--undecided`, even if their valid attempts already include a pass and a fail.

Write just those tasks to a replacement collection file:

```bash
gyms/wideseek/.venv/bin/python -m evals.wideseek.band <run> \
  --rerun rerun.jsonl --data <pool.jsonl>
```

Collect replacement attempts in a new output directory, then pass both the
original and replacement run directories to `band`. Repeat until `rerun` is empty.
For example, one pass plus one judge error in a two-attempt run is unclassified;
one valid replacement failure makes it in-band with one pass in two valid attempts.
Additional valid attempts from later stages still contribute to the pass rate.
Known episode time, token, context and recursion limits remain measured failures.

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
