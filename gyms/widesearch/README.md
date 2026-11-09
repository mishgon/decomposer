# WideSearch With Serper Web Search

The gym uses the 100 English WideSearch questions with live Serper Google search
and page scraping. Set `SERPER_API_KEY` and `LLM_PROXY_MASTER_KEY` in the launch
environment (the CLI does not load `.env` itself).

Researchers use the English system prompt and `search_global` / `text_browser_view`
JSON schemas from the pinned `external/WideSearch` submodule. Both single agents
and Decomposer workers use these tools. Upstream's private search/reader backend
is unavailable; its README asks users to implement their own. This adapter is
not an exact reproduction of that backend or of the upstream Runner.

- `search_global`: Serper organic results, including titles, URLs and snippets.
  The English-only gym uses `hl=en, gl=us` for both values of `use_english`.
  Counts above 100 use additional pages, up to the upstream limit of 200.
  `short` returns search snippets; `long` additionally reads each result page.
- `text_browser_view`: Serper Scrape extracts Markdown/text, then the registry's
  `lmrouter/qwen_3_8_flash_next_non_thinking` summarizes it for the requested
  `description`. The reader sees at most 60,000 characters of a page and is told
  about this limit. This fixed reader is shared across all evaluated models.
  Website blocks, missing text and tool failures remain visible in the trace.
- Serper raw replies, reported credits, page-reader inputs and model outputs are
  saved in `web_calls.jsonl`, without authentication headers. Reader model usage
  is recorded there, separately from the agent's `usage.json`.

Search and page reading both spend Serper credits. In the initial live probe,
one search reported 1 credit and one scrape reported 10; `long` search can be
expensive because it scrapes each result. Check usage before scheduling a full run.

The LangGraph runtime, Decomposer prompt, worker run budgets, recursion limit of
410, model logging and trace export remain in use. The upstream Runner's default
50-step loop is not used. Single-agent system instructions are the upstream
English prompt; Decomposer workers additionally receive the existing run-budget
instructions. The BrowseComp ten-search budget and local-index URL restrictions
are removed. There is no fallback to the local corpus.

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
`decomposer_4b_non_thinking`, `decomposer_sft_1_2_0`.
All Decomposers use 4B thinking workers. All model sampling parameters come from
`src/decomposer/models.py`; 27B uses `reasoning_effort="medium"`.

Each episode saves `answer.md`, `trace.json`, `trace.html`, `model_calls.jsonl`,
`web_calls.jsonl`, `usage.json` and `result.json`. Evaluation is pending until the
separate scoring command runs. Only questions enter the agent input, never gold
tables. Dataset revision: `6531a7e5b497d44c8912407e0cb3dc95bd98cc09`.

The shared scheduler records `retrieval_url="serper-web-v1"` as a backend marker;
this is not an HTTP endpoint. Older local-index runs cannot resume under the web
backend. Their saved results remain a separate experiment; reproduce that setting
from commit `7b822bd`. Interrupted web runs can resume with `--resume <run-dir>`.
