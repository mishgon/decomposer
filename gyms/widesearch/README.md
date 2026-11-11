# WideSearch With BrowseComp-Plus Retrieval

This experiment uses WideSearch questions and gold tables with the same
agentic-rag harness as BrowseComp: the research prompt, `search` and `fetch`,
per-agent URL access, and model profiles are shared directly. There is no
per-agent search-count budget. Decomposer workers have 180 seconds of work and
180 seconds to finish and respond; the recursion limit remains 410.

This is a local-corpus experiment, not the upstream live-web benchmark.
The corpus is BrowseComp-Plus, served by agentic-rag with BGE-M3 and Qdrant.
Both single agents and Decomposer workers use the same corpus and tools.
`fetch` accepts only URLs returned by that agent's searches.

Start agentic-rag locally and pass its URL with `--retrieval-url` (default
`http://127.0.0.1:18082`). Preflight checks both search and page fetch before
scheduling episodes. Keenable credentials are not needed for this experiment.
The previous web adapter remains in `tools.py` but is not used by these agents.

```bash
PYTHONPATH=src:. python -m gyms.widesearch.run \
  --agent decomposer_sft_1_4_0 --tasks ws_en_001 \
  --retrieval-url http://127.0.0.1:18082 --concurrency 1 -n 1
PYTHONPATH=src:. python -m evals.widesearch.run PATH_TO_RUN
```

Answers are saved verbatim in `answer.md`. The task requests a table in a fenced
`markdown` block, using the requested column names. Offline evaluation retains
the pinned upstream strict success, row F1 and item F1 metrics and records judge
failures explicitly. Model credentials are loaded by the launcher, not the CLI.
Use a fresh output directory when changing the harness or retrieval backend.
