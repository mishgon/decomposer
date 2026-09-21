# WideSeek gym

Minimal setup on `wideseek_gym`, based on `dev`. No Toolathlon/RL code is merged
into this branch. Current implementation prepares data only; collection and native
evaluation are not wired yet. No paid APIs are called by preparation.

```bash
python3 -m gyms.wideseek.prepare --source width
python3 -m unittest discover -s tests -p 'test_wideseek_data.py'
```

Uses only the Python standard library. Source choices: `width`, `depth`, `hybrid`.
Each has 20,000 examples; hybrid mixes the other sources, so do not concatenate
them as independent data. Preparation pins the HF revision, preserves raw rows,
checks the schema, and records file hashes. It refuses to overwrite a directory;
only a completed dataset has `manifest.json`.

Artifacts: `artifacts/gyms/wideseek/data/<source>/`. References stay in dataset
files for scoring; `agent_input` supplies only the question. Split future held-out
panels by question hash (and inspect paraphrase overlap), not by file membership.
Evaluation on this training dataset is a development measurement, not a published
WideSearch benchmark result.

## Upstream findings and remaining work

- Data: https://huggingface.co/datasets/RLinf/WideSeek-R1-train-data
- Tools: https://rlinf.readthedocs.io/en/latest/rst_source/examples/agentic/wideseek_r1/tools.html
- Scoring: `rlinf/agents/wideseek_r1/utils/reward.py` in https://github.com/RLinf/RLinf
- Benchmark: https://github.com/RLinf/WideSeek-R1-Eval

Width tasks produce Markdown tables. Upstream uses unique-column row matching
and an LLM judge for item-level F1, not plain exact match. Depth has a different
answer format. Pin/adapt upstream scoring and test perfect, incomplete, wrong,
and malformed answers before collecting model results. Judge/API failures must
remain unscored infrastructure failures, not silently become reward zero.

Upstream training usually uses an offline Wiki-2018/Qdrant search service;
WideSearch evaluation uses online Serper search and Jina page access. The search
backend, corpus, judge model and generator models still need to be chosen. Do not
silently substitute the LiteResearcher example's BrowseComp corpus.

Next steps, deliberately small:

1. One `search`/`access` adapter shared by simple and decomposer workers.
2. One episode function: question -> agent -> complete trace -> native scorer.
   Keep core Decomposer unchanged and use its existing subagent service contract.
3. One runner with bounded episode/tool concurrency, repetitions, and resume.
   Store per-attempt messages (including reasoning), tool I/O, usage, timing,
   outcome, evaluator inputs/outputs and model/backend settings under one run ID.
   Keep raw reasoning in artifacts even if model-specific replay excludes it.
4. Small same-model simple-vs-decomposer comparison, with matched tools and
   explicit total compute/token budgets, before teacher collection or RL.

No GPUs, search indexes, model services or paid trace jobs are started by this setup.
