# WideSearch With Free Parallel Web Search

The gym uses the 100 English WideSearch questions with Parallel's public keyless
MCP endpoint (`https://search.parallel.ai/mcp`), also used by Hermes Agent.
Only `LLM_PROXY_MASTER_KEY` is needed in the launch environment; the CLI does not
load `.env` itself. Search and extraction need no API key or paid fallback.

Researchers retain the upstream English system prompt and `search_global` /
`text_browser_view` JSON schemas. Single agents and Decomposer workers share the
same tools. Upstream's private backend is unavailable, so this is an adapter,
not an exact reproduction of its search results or Runner.

- `search_global` calls Parallel `web_search` and returns titles, URLs and excerpts.
  `count` is an upper bound: the free endpoint controls the actual result count
  (10 in our probe), with no pagination parameter. `use_english` does not change
  provider settings; this gym uses English tasks and sends queries as written.
  `long` additionally reads pages with a shared 900-second reader budget;
  failed or skipped reads retain search excerpts.
- `text_browser_view` calls Parallel `web_fetch`. Available content/excerpts go
  through the unchanged `lmrouter/qwen_3_8_flash_next_non_thinking` reader with
  the requested description and a 60,000-character input cap. Provider excerpts
  may omit parts of a page. Reader inference still costs model tokens.
- Requests are serialized per episode process, with a five-second pause before
  each request. HTTP 429/502/503/504 and MCP tool rate-limit errors retry at most
  three times, waiting 60/120/240 seconds, respecting a longer HTTP Retry-After.
  The total tool timeout is 1,800 seconds, including queueing and retries.
  Start with `--concurrency 1`; separate episode processes have separate limiters.
- Raw replies, retry waits and reader inputs/outputs are saved in `web_calls.jsonl`.
  Reader usage remains separate from the agent's `usage.json`. Exhausted retries
  and page extraction errors are visible to the agent. There is no provider or
  local-index fallback.

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

The shared scheduler records `retrieval_url="parallel-free-web-v1"` as a backend marker;
this is not an HTTP endpoint. Older local-index and Serper runs cannot resume under this
backend. Their saved results remain a separate experiment; reproduce that setting
from commit `7b822bd`. Interrupted web runs can resume with `--resume <run-dir>`.
