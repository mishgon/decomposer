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
per turn. The student keeps the agent type and the 60-second run budget of its training
records. Its researchers run on the `researcher_non_thinking` worker graph with the
hosted non-thinking 4B model that `STUDENT_RESEARCHER_MODEL` names. `--agent student_180`
is the same student with the teachers' 180-second run budget and shutdown grace.
`--agent student_rag` is `student_180` over `researcher_agentic_rag` workers: the monorepo's
cited-search harness (search, open, find and finish, at most 32 model calls with the last for
finish only) over the hosted Search API's `wikipedia` source instead of the offline Wiki-2018
corpus. It reads the API key from `AGENTIC_RAG_API_KEY` and the front from `AGENTIC_RAG_URL`,
by default `https://lmrouter.2a2i.org/rag`.

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
