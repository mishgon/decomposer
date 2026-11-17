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

## Seen-Pool Recipe on OCCtopus

`occtopus/seen.json` pins the 4,783-task pool, four attempts per task,
concurrency 256 and a 900-second episode budget. Allocate one GPU and run:

```bash
bash evals/wideseek/occtopus/run.sh <new-run-directory> <shared-runtime> \
  <frozen-tasks.jsonl> <merged-models-directory> <vllm-python>
```

Bind `HF_TOKEN`, `LLM_PROXY_MASTER_KEY`, `AGENTIC_RAG_API_KEY` and
`CLEARML_CONFIG_FILE`. The recipe defaults to SFT v1.8.0; append
`--sft-version 1.7.0` to verify the same full-pool launch with the previous release.
Missing checkpoints are merged with the release's `merge.py` and checked against
its published SHA256. The resolved Hugging Face revision is saved with the model.
The pool and checkpoint are verified before inference starts. Use a fresh run
directory when changing the checkpoint or budget; prior attempts are not imported.
Clean episode budget exhaustion counts as a failed attempt without retry.
Infrastructure and cleanup failures retain their existing recovery rules.
ClearML reporting runs inside the collector.

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
