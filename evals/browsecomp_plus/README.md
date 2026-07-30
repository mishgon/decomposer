# BrowseComp plus bench

## Prereq

Run from `environments/agentic-search`

```bash
uv run langgraph dev --n-jobs-per-worker 1000
```

## Run bench

With decomposer

```bash
uv run -m evals.browsecomp_plus decomposer
```

With just subagent

```bash
uv run -m evals.browsecomp_plus direct
```
