# Minimal example

This example runs Decomposer directly and exposes one general-purpose subagent
through a local LangGraph server. Decomposer uses GLM-5.2 through OpenRouter;
the subagent uses LiteResearcher-4B through a local vLLM server.

Set `OPENROUTER_API_KEY` in the environment before running the example.

From the repository root, start vLLM for the subagent:

```bash
scripts/vllm_serve_literesearcher_4b.sh
```

In another terminal, start the subagent server:

```bash
cd examples/literesearcher
uv run langgraph dev --no-browser
```

Then run Decomposer from the repository root:

```bash
uv run python examples/literesearcher/run.py
```

The run loads the first three BrowseComp-Plus prompts, runs them concurrently,
prints their final answers, and saves each Decomposer message history as
human-readable Markdown under `examples/literesearcher/messages/`.

## Full BrowseComp-Plus evaluation

Run the complete 830-query test split with:

```bash
uv run python examples/literesearcher/eval.py
```

The evaluator runs up to 10 samples concurrently and uses GLM-5.2 for both
Decomposer and answer-equivalence judging. Each completed query is appended to
`artifacts/literesearcher/browsecomp_plus_eval.jsonl`, so an interrupted run
resumes without repeating completed work. Aggregate accuracy is written
separately to `artifacts/literesearcher/browsecomp_plus_eval_score.json`. Use
`--concurrency N` to change parallelism or `--no-resume` to start over.
Checkpoint rows contain only the query ID, correctness, match method, and
sanitized exception type when a query fails; prompts, gold answers, generated
answers, judge output, URLs, and exception messages are never persisted.

For the direct LiteResearcher baseline, which sends each complete question
straight to the subagent without Decomposer, run:

```bash
uv run python examples/literesearcher/eval-direct.py
```

It uses the identical dataset, concurrency, matching, GLM-5.2 judge, error
handling, and checkpointing pipeline. Its result checkpoint is
`artifacts/literesearcher/browsecomp_plus_eval_direct.jsonl`, with aggregate
accuracy in `artifacts/literesearcher/browsecomp_plus_eval_direct_score.json`.
Existing partial `.json` artifacts from the previous format are imported
automatically when the corresponding evaluation is resumed.
