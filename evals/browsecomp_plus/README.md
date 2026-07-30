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

## Long runs

launch environment with nohup:

```bash
nohup uv run langgraph dev --n-jobs-per-worker 1000 &
```

lauch eval with log file and nohup (you want PYTHONUNBUFFERED to stream logs):

```bash
nohup env PYTHONUNBUFFERED=1 uv run -m evals.browsecomp_plus direct \
  > evals/browsecomp_plus/out.log 2>&1 &
```
