# WideSearch With the WideSeek-R1 Wiki-2018 Harness

The gym uses WideSearch's 100 English questions and upstream scorer with the same
researcher implementation as WideSeek-R1. Researchers reuse its system prompt,
`search` and `access` tools, fixed Wikipedia-2018 corpus, truncation rules and
raw tool logging. There is no web API, Flash Next page reader or search-count cap.
Keenable credentials are not required for these runs.

Set up the index using [WideSeek setup](../wideseek/README.md#setup). The default
retrieval endpoint is `http://127.0.0.1:18080`; `--retrieval-url` selects its address.
Preflight verifies the pinned corpus revision and both retrieval operations.
Decomposer workers retain 180 seconds of work plus 180 seconds to finish; recursion
remains 410. Optional WideSeek model-call/token budgets are initialized without
limits. Single agents and workers share the same prompt and tools.

Initialize `external/WideSearch` with `git submodule update --init external/WideSearch`.
Apply the existing empty-row scoring fix with
`git -C external/WideSearch apply ../../gyms/widesearch/upstream.patch`.
Install project/Agent Server dependencies and `gyms/widesearch/requirements.txt`
in an isolated environment (the upstream evaluator needs pandas 2.2.3).

```bash
python -m gyms.widesearch.dataset
python -m gyms.widesearch.run --agent researcher_27b_thinking \
  --tasks ws_en_001 -n 1 --concurrency 1
python -m evals.widesearch.run artifacts/evals/widesearch/<run-id>
```

Other assistant IDs: `researcher_4b_thinking`, `decomposer` (Flash Next manager),
`decomposer_4b_non_thinking`, `decomposer_sft_1_2_0`, `decomposer_sft_1_4_0`.
All Decomposers use 4B thinking workers. All model sampling parameters come from
`src/decomposer/models.py`; 27B uses `reasoning_effort="medium"`.

Each episode saves `answer.md`, `trace.json`, `trace.html`, `model_calls.jsonl`,
`runtime/model_calls.jsonl`, `runtime/tool_calls/*.json`, `usage.json` and `result.json`. Evaluation is pending until the
separate scoring command runs. Only questions enter the agent input, never gold
tables. Dataset revision: `6531a7e5b497d44c8912407e0cb3dc95bd98cc09`.

Answers use one table in a fenced block labeled `markdown`, matching WideSeek
table tasks. The upstream evaluator extracts the table and ignores surrounding
explanations; its F1 and strict metrics are unchanged.

Manifests record the Wiki-2018 corpus revision and retrieval address. Historical
Keenable/Serper/BrowseComp-index runs cannot resume under this harness; use a new
run directory. Their legacy web tools remain in `tools.py` for reference, but the
registered researchers do not call them. This restricted-corpus experiment does
not reproduce the original WideSearch web-search setting.

Queue recovery: infrastructure failures in model, worker and web logs are monitored
while episodes run. Authentication/quota failures pause scheduling immediately;
three consecutive infrastructure-failed episodes also pause it. Runs with remaining
infrastructure or judging errors have `needs_attention` status. Resume preserves
successful and model-limited attempts, archives infrastructure-failed attempts and
retries their task/repetition pairs. Resume requires the same commit, input hash,
backend, model and worker budgets. Judge failures can be rescored without generating
another answer (BrowseComp via resume; WideSearch via its scoring command).
