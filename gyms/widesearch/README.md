# WideSearch With Keenable Web Search

The gym uses the 100 English WideSearch questions with the authenticated Keenable
Search and Fetch APIs. Export `KEENABLE_API_KEY_1`, `KEENABLE_API_KEY_2`, and any
additional numbered keys, plus `LLM_PROXY_MASTER_KEY`, in the launch environment.
The single-key `KEENABLE_API_KEY` remains supported; duplicate values are used once.
The CLI does not load `.env` itself.

Researchers retain the upstream English system prompt and `search_global` /
`text_browser_view` JSON schemas. Single agents and Decomposer workers share the
same tools. Upstream's private backend is unavailable, so this is an adapter,
not an exact reproduction of its search results or Runner.

- `search_global` calls Keenable `POST /v1/search` in `pro` mode and returns
  titles, URLs and snippets. `count` is an upper bound, capped at the provider's
  maximum of 50 results. `use_english` does not change provider settings; queries
  are sent as written. `long` additionally reads pages with a shared 900-second
  reader budget; failed or skipped reads retain search snippets.
- `text_browser_view` calls `GET /v1/fetch`. Page content goes through the existing
  `lmrouter/qwen_3_8_flash_next_non_thinking` reader with the requested description
  and a 60,000-character input cap. Reader inference still costs model tokens.
- Requests rotate across available keys within each episode process, with a
  0.2-second pause before each attempt. HTTP 401/402/403 disables that key for
  the process and tries another. HTTP 429 puts the key on cooldown, respecting
  `Retry-After`, while other keys remain available. If all keys are cooling down,
  the request waits for the earliest one. HTTP 502/503/504 use exponential backoff
  capped at 8 seconds, or a longer `Retry-After`. Each request has at most four
  attempts per configured unique key in total. If all keys fail, the error is
  reported to the queue. The total tool timeout remains 1,800 seconds.
  Start with `--concurrency 1`; the provider's account limit is shared across
  processes. Queueing and retries count toward the existing worker run budget.
- Raw requests, replies (including provider credit usage), retry waits and reader
  inputs/outputs are saved in `web_calls.jsonl`; authentication headers are not
  logged. Each attempt records only the environment variable name in `key_name`.
  Reader usage remains separate from the agent's `usage.json`. Failed
  calls are visible to the agent. There is no provider or local-index fallback.

API references: [Search](https://docs.keenable.ai/api-reference/search),
[Fetch](https://docs.keenable.ai/api-reference/fetch),
[limits](https://docs.keenable.ai/rate-limits), and
[credits](https://docs.keenable.ai/credits).

Decomposer workers have 180 seconds of work plus 180 seconds to finish the current
step and respond, shared with BrowseComp through its Decomposer factories.
The LangGraph runtime, Decomposer prompt, recursion limit of 410, model logging
and trace export remain in use. The upstream Runner's default
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
`decomposer_4b_non_thinking`, `decomposer_sft_1_2_0`, `decomposer_sft_1_4_0`.
All Decomposers use 4B thinking workers. All model sampling parameters come from
`src/decomposer/models.py`; 27B uses `reasoning_effort="medium"`.

Each episode saves `answer.md`, `trace.json`, `trace.html`, `model_calls.jsonl`,
`web_calls.jsonl`, `usage.json` and `result.json`. Evaluation is pending until the
separate scoring command runs. Only questions enter the agent input, never gold
tables. Dataset revision: `6531a7e5b497d44c8912407e0cb3dc95bd98cc09`.

Answers use one table in a fenced block labeled `markdown`, matching WideSeek
table tasks. The upstream evaluator extracts the table and ignores surrounding
explanations; its F1 and strict metrics are unchanged.

The shared scheduler records `retrieval_url="keenable-pro-web-v1"` as a backend
marker, not an HTTP endpoint. Runs from Parallel, Serper or the local index cannot
resume under this backend. Preserve them as separate experiments. Interrupted
Keenable runs can resume with `--resume <run-dir>` when their recorded code revision
also matches. Wiki-2018 attempts remain separate. BrowseComp continues to use the
WideSeek-R1 Wiki-2018 harness; this change applies only to WideSearch.

For parallel episodes, `KEENABLE_API_URL=http://127.0.0.1:18784/v1` can route
search and fetch requests through a trusted local reverse proxy with a shared
rate limiter. The proxy must forward requests to Keenable, preserve the API-key
header, and verify upstream TLS. The existing in-process lock does not limit
requests across episode processes. Without this variable, requests go directly
to `https://api.keenable.ai/v1`.

Queue recovery: infrastructure failures in model, worker and web logs are monitored
while episodes run. Authentication/quota failures pause scheduling immediately;
three consecutive infrastructure-failed episodes also pause it. Runs with remaining
infrastructure or judging errors have `needs_attention` status. Resume preserves
successful and model-limited attempts, archives infrastructure-failed attempts and
retries their task/repetition pairs. Resume requires the same commit, input hash,
backend, model and worker budgets. Judge failures can be rescored without generating
another answer (BrowseComp via resume; WideSearch via its scoring command).
