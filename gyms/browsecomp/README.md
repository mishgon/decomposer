# BrowseComp-Plus

This gym uses BrowseComp-Plus questions and answers with the shared WideSeek
agentic-rag harness and hosted Search API (`source=wikipedia`). It does not use
the old BrowseComp-Plus retrieval corpus or claim official live-web reproduction.

## Setup

From the repository root, in an environment with the project and LangGraph CLI installed:

```bash
uv pip install -r gyms/browsecomp/requirements.txt
PYTHONPATH=src:. python -m gyms.browsecomp.dataset
```

Preparation decodes `Tevatron/browsecomp-plus` into the ignored
`artifacts/gyms/browsecomp/tasks.jsonl`. Keep these benchmark questions and answers
private. The optional datasets dependency is needed only for preparation.

Load `LLM_PROXY_MASTER_KEY` for models and `AGENTIC_RAG_API_KEY` for retrieval.
The default Flash Next non-thinking manager delegates to OCC-4B non-thinking.
Sampling comes from `decomposer.models`. Both workers and standalone researchers
use `gyms.wideseek.agentic_rag` unchanged: search, open, find, finish, and 32 model
calls per run (the last reserved for finish). There is no per-worker time budget.
The outer episode timeout remains 45 minutes.

## Retrieval Connection

The default endpoint is `https://lmrouter.2a2i.org/rag`. Override it with
`AGENTIC_RAG_URL` or `--retrieval-url` for a compatible Search API.
Preflight checks the authenticated health endpoint with a 30-second deadline.

The shared research protocol counts context tokens with the public
`Qwen/Qwen3.5-4B` tokenizer (no model weights are loaded). Install this gym's
requirements and cache the tokenizer before running offline:

```bash
uv pip install -r gyms/browsecomp/requirements.txt
python -c 'from transformers import AutoTokenizer; AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")'
```

## Run

```bash
PYTHONPATH=src:. python -m gyms.browsecomp.run \
  --agent researcher_occ_non_thinking --tasks browsecomp-0001

PYTHONPATH=src:. python -m gyms.browsecomp.run \
  --agent decomposer --tasks browsecomp-0001 browsecomp-0002 --concurrency 2
```

Each episode has its own Agent Server and checkpointed researcher state.
Workers receive the episode directory through runtime context for tool logs.
Only URLs returned by search can be opened or cited; forks inherit this state.

Runs save `manifest.json`, then `traces/<task>-rNNN/` with `task.json`, `trace.json`,
optional `trace.html` (when the shared renderer has captured agent runs),
`usage.json`, `result.json`, `runner.log` and append-only
`model_calls.jsonl`. Judge requests and responses go into `judge.json` when needed.
The shared capture helper cancels active agents before scoring timeout/error traces.
A fresh invocation creates a new run directory. To continue after stopping a runner,
repeat its original arguments and add `--resume PATH_TO_RUN`. You may change
`--concurrency`; keep the task selection, repetitions, agent, data path, retrieval
URL and timeout unchanged. Completed episodes (including scored failures) are
preserved. Partial episode directories move to `interrupted/` before retrying,
so model logs from different attempts cannot mix. Resume accepts only a manifest
marked `interrupted`; stop the old runner first.

## Scoring Changes

Agents are asked to wrap their final answer in `\boxed{}`, matching the WideSeek
QA instruction. The evaluator uses WideSeek's balanced-brace parser to extract
the last boxed answer, ignoring surrounding explanations. Without a box, it
scores the full final text after any `</think>` marker and records `format_ok=false`.
An empty answer, including an empty box, is marked `missing_final_answer`.
The selected `answer` and `format_ok` are saved in the evaluation result; the raw
response remains in the trace. JSON and source fields are not required.

The old evaluator accepted bidirectional token inclusion, which could mark
an underspecified answer correct. Only exact case-insensitive answers now bypass
the equivalence judge. Invalid judge responses and provider errors are unscored,
not ordinary wrong answers. This is our migrated answer-equivalence evaluator,
not a claim of reproducing the official BrowseComp scorer. Historical results
in `evals/browsecomp/history.md` also used different models and prompts.

## Evaluation Variants

Choose a Python factory with `--agent`:

| Agent | Model | Harness |
| --- | --- | --- |
| `researcher_occ_non_thinking` | OCC-4B non-thinking | Shared research protocol |
| `researcher_4b_thinking` | Qwen3.5-4B thinking | ReAct |
| `researcher_27b_thinking` | Qwen3.8-27B-FP8 thinking (medium) | ReAct |
| `decomposer_4b_non_thinking` | Qwen3.5-4B non-thinking | Decomposer |
| `decomposer_sft_1_0_0` | Decomposer manager SFT 1.0.0 | Decomposer |
| `decomposer_sft_1_2_0` | Decomposer manager SFT 1.2.0 | Decomposer |

Decomposer variants use the OCC `researcher` graph, except the explicitly named
historical `_timur` variant. The `researcher` alias is also the OCC single agent;
`decomposer` uses the Flash Next manager. Legacy model factories remain available,
but now use the shared research protocol. Start new runs after this migration;
do not combine their metrics with results from the old retrieval harness.

SFT 1.0.0 is available for historical comparisons through `decomposer_sft_1_0_0`.
Its server must serve `decomposer-manager-qwen35-4b-sft-1.0.0` on
`127.0.0.1:18025`. It uses the same sampling and workers as SFT 1.2.0.

Sampling comes from `src/decomposer/models.py`. SFT 1.2.0 shares the base
Qwen3.5-4B non-thinking sampling settings. Its OpenAI-compatible server must
serve `decomposer-manager-qwen35-4b-sft-1.2.0` on `127.0.0.1:18024`.
Forward that port from the inference machine before running locally.
The other evaluation models use `https://lmrouter.2a2i.org/v1` and
`LLM_PROXY_MASTER_KEY`.

For example, after loading environment variables:

```bash
python -m evals.browsecomp.run --agent decomposer_sft_1_2_0 \
  --tasks TASK_ID_1 TASK_ID_2
```

Queue recovery: infrastructure failures in model, worker and web logs are monitored
while episodes run. Authentication/quota failures pause scheduling immediately;
three consecutive infrastructure-failed episodes also pause it. Runs with remaining
infrastructure or judging errors have `needs_attention` status. Resume preserves
successful and model-limited attempts, archives infrastructure-failed attempts and
retries their task/repetition pairs. Resume requires the same commit, input hash,
backend, model and worker budgets. Judge failures can be rescored without generating
another answer (BrowseComp via resume; WideSearch via its scoring command).
