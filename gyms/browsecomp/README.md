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
Qwen3.5-4B unlooped thinking researchers. The standalone agent uses the same
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
can fetch only URLs they have searched; forks inherit that state. Search counts
and URLs persist across successive runs of one agent. Each agent has a 10-search
budget; concurrent searches started together may exceed the remaining budget.
Fetch returns the server's page body, not a query-specific LLM summary. These
details match the old HTTP adapter, with the prompt corrected to use native tools.

Runs save `manifest.json`, then `traces/<task>-rNNN/` with `task.json`, `trace.json`,
optional `trace.html` (when the shared renderer has captured agent runs),
`usage.json`, `result.json`, `runner.log` and append-only
`model_calls.jsonl`. Judge requests and responses go into `judge.json` when needed.
The shared capture helper cancels active agents before scoring timeout/error traces.
Each invocation creates a new run directory; there is no gym-level resume policy.

## Scoring Changes

The old evaluator accepted bidirectional token inclusion, which could mark
an underspecified answer correct. Only exact case-insensitive answers now bypass
the equivalence judge. Invalid judge responses and provider errors are unscored,
not ordinary wrong answers. This is our migrated answer-equivalence evaluator,
not a claim of reproducing the official BrowseComp scorer. Historical results
in `evals/browsecomp/history.md` also used different models and prompts.
