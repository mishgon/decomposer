# BrowseComp-Plus

This gym migrates the `browse-bench-2` evaluation. It uses the **BrowseComp-Plus**
fixed corpus through agentic-rag's `/search` and `/fetch`, not live-web BrowseComp
or WideSeek's Wiki-2018 service. No corpus is downloaded or indexed by the runner.

## Setup

From the repository root, in an environment with the project and LangGraph CLI installed:

```bash
uv pip install -r gyms/browsecomp/requirements.txt
PYTHONPATH=src:. python -m gyms.browsecomp.dataset
```

Preparation decodes `Tevatron/browsecomp-plus` into the ignored
`artifacts/gyms/browsecomp/tasks.jsonl`. Keep these benchmark questions and answers
private. The optional datasets dependency is needed only for preparation.

Load model credentials as documented in the repository README. Model profiles
are constants in `agents.py`: Qwen3.8 Flash Next non-thinking orchestrates
Qwen3.5-4B thinking researchers. The standalone agent uses the same
researcher. Generation parameters come from `decomposer.models`; recursion is
410 for both roles, and the default episode agent timeout is 45 minutes.

## Retrieval Connection

Supply an HTTP endpoint for the existing agentic-rag deployment. It must have
the `browsecomp_plus` source loaded. An SSH-forwarded loopback address also works:

```bash
# RAG_HOST must already be a working SSH alias for the retrieval machine.
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -L 127.0.0.1:18000:127.0.0.1:8000 RAG_HOST
```

Run this on the machine executing the gym and pass
`--retrieval-url http://127.0.0.1:18000`. Keep SSH host-key checking enabled.
The deployment host and Hertz-2 route still need live verification. The runner
tests both search and fetch before starting any episodes or paid model calls.
It does not start or restart the remote retrieval service.

## Run

```bash
PYTHONPATH=src:. python -m gyms.browsecomp.run \
  --agent researcher --tasks browsecomp-0001 --retrieval-url http://127.0.0.1:18000

PYTHONPATH=src:. python -m gyms.browsecomp.run \
  --agent decomposer --tasks browsecomp-0001 browsecomp-0002 --concurrency 2
```

Each episode has its own Agent Server and checkpointed researcher state. Agents
can fetch only URLs they have searched; forks inherit that state. Found URLs persist across successive runs of one agent. There is no search-count
limit. Decomposer workers retain the per-run time budget (180 seconds of work plus
180 seconds to finish the current step and respond); standalone researchers use the episode timeout and
recursion limit.
Fetch returns the server's page body, not a query-specific LLM summary. The prompt uses native tools.

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
| `researcher_4b_thinking` | Qwen3.5-4B thinking | ReAct |
| `researcher_27b_thinking` | Qwen3.8-27B-FP8 thinking (medium) | ReAct |
| `decomposer_4b_non_thinking` | Qwen3.5-4B non-thinking | Decomposer |
| `decomposer_sft_1_0_0` | Decomposer manager SFT 1.0.0 | Decomposer |
| `decomposer_sft_1_2_0` | Decomposer manager SFT 1.2.0 | Decomposer |

All Decomposer variants use the `researcher` graph, a Qwen3.5-4B thinking
subagent with the same search and fetch tools as the standalone researcher.
The existing `researcher` and `decomposer` factories remain available;
`decomposer` uses the Flash Next manager.

SFT 1.0.0 is available for historical comparisons through `decomposer_sft_1_0_0`.
Its server must serve `decomposer-manager-qwen35-4b-sft-1.0.0` on
`127.0.0.1:18025`. It uses the same sampling and workers as SFT 1.2.0.

Sampling comes from `src/decomposer/models.py`. SFT 1.2.0 shares the base
Qwen3.5-4B non-thinking sampling settings. Its OpenAI-compatible server must
serve `decomposer-manager-qwen35-4b-sft-1.2.0` on `127.0.0.1:18024`.
Forward that port from the inference machine before running locally.
The other evaluation models use `https://lmrouter.2a2i.org/v1` and
`LLM_PROXY_MASTER_KEY`.

For example, after loading environment variables and forwarding retrieval:

```bash
python -m evals.browsecomp.run --agent decomposer_sft_1_2_0 \
  --tasks TASK_ID_1 TASK_ID_2 --retrieval-url http://127.0.0.1:18082
```

Queue recovery: infrastructure failures in model, worker and web logs are monitored
while episodes run. Authentication/quota failures pause scheduling immediately;
three consecutive infrastructure-failed episodes also pause it. Runs with remaining
infrastructure or judging errors have `needs_attention` status. Resume preserves
successful and model-limited attempts, archives infrastructure-failed attempts and
retries their task/repetition pairs. Resume requires the same commit, input hash,
backend, model and worker budgets. Judge failures can be rescored without generating
another answer (BrowseComp via resume; WideSearch via its scoring command).
