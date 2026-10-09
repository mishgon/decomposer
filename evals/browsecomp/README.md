# BrowseComp Evaluation

First complete [gym setup](../../gyms/browsecomp/README.md).

```bash
PYTHONPATH=src:. python -m evals.browsecomp.run --agent researcher --all -n 3 --concurrency 2
PYTHONPATH=src:. python -m evals.browsecomp.run --agent decomposer --all -n 3 --concurrency 2
```

Use `--tasks browsecomp-0001 browsecomp-0002` for a small comparison and
`--retrieval-url` for the forwarded agentic-rag endpoint.
Results live in `artifacts/evals/browsecomp/<run-id>/`. `summary.json` includes
pass@1, pass@3 and pass^3 when `-n 3` is used. The denominator is the selected
task set, with infrastructure failures counted as failures. Only a normally
finished, correctly judged episode counts as a pass. Native scores from stopped
episodes remain available for analysis. No historical artifacts are overwritten.
