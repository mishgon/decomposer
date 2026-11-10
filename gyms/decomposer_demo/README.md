# decomposer-demo

Twenty frozen validation tasks derived from the Decomposer demo: **8 calendar,
6 estate and 6 frontend**. English and Russian are two views of the same task
IDs. Russian is the default. `--language en|ru` selects the task prompt and frontend seed, as in the demo.
Calendar event metadata stays in its original language.

## Demo Fidelity

`demo_prompts.json` pins the environment descriptions, workspace context, worker
systems and helper text from demo commit `8b7b050dc774aa324c6fdec43b5e23427b6287f6`,
with source-file SHA-256 hashes. The coordinator receives all three workspace
sections, as the demo does. All three `qwen-4b-calendar`, `qwen-4b-estate` and
`qwen-4b-frontend` worker types are available in every episode. Each environment
has fresh private state; agents of the same type share it. Execution/reporting,
parallel ownership and environment-specific guidance are copied verbatim.
The coordinator's core prompt matches the demo's pinned upstream prompt.
Worker models follow `selectWorkerModel`: Qwen3.5-4B-unlooped for calendar and
estate, Flash Next for frontend. `--worker-model` is an explicit all-worker
override. Hosted calls cap each output turn at 4096 tokens.
Teacher/student coordinator comparisons retain the experiment's non-thinking
profiles; the demo's provider-default coordinator can enable reasoning. This
model-mode difference remains separate from environment prompt fidelity.

Explicit capability notes explain synthetic listings and unavailable image,
preview and browser evidence. Frontend files use the demo's initial revision 1,
project versions, tombstones, read-before-write checks and size limits. Preview
inspection reports unavailable; generated code is never executed. UI line-author
annotations are not reproduced because tools do not expose them.

The current demo forces Russian reports. This benchmark deliberately extends
that presentation instruction to English for `language='en'`; Russian preserves
the original instruction. Task text and frontend seeds also follow the language
selection. Episodes save the assembled coordinator/worker system prompts and
source demo commit for inspection.

The Python LangChain transport preserves the upstream tool and budget logic but
is not the demo's JavaScript AI SDK runtime. Worker budgets are 60 seconds plus
60 seconds of reporting grace, as in the demo. Python graph recursion limits
and the 300-second episode timeout remain benchmark limits, rather than an
exact reproduction of the demo's 100-model-turn loop. Tasks are deterministic
adaptations: frontend quality covers explicit source requirements, not the
demo's open-ended visual redesigns. Estate matching uses fictional fields,
not photo evidence or real-market coverage.

All tools execute in-process against episode-private memory. There is no Docker,
sandbox service, browser, subprocess execution, SQLite or live property API.
Generated frontend source is stored and checked, never executed. Only model
inference needs a configured endpoint. Tools use short async wrappers to keep
shared-state mutations atomic on the event loop.

## Tasks and Rewards

Calendar tasks reuse the 40-event `src/timetable.json` snapshot from
`demo-decomposer` commit `d256f1fdee9a4b304b92a0c7eda3e02fc0098284`. The AIRI prompt comes
from `src/locales/en.json` / `ru.json`, `promptTemplates.airi`; the other tasks use
explicit filters inspired by the demo. Tools follow `src/calendar.ts`, including
six-hour overlapping reads, local clock times, and complete event metadata.
`get_events` exposes the demo's `from`/`to` names to the model; its Python method
uses `start`/`end`. Reward requires the exact retained event set, including
metadata. Reordering is allowed; duplicates, missing or added events fail.

Estate tasks use 96 deterministic **fictional** listings created in Python.
The search/fetch signatures, numeric IDs, 20-item pages, city aliases, exact room
filter and discovery sorting follow the demo. Default `diverse` ordering uses the
same deterministic price-band algorithm. `fetch` returns the demo's model-visible
`status`, `source`, `url`, `captured_at`, `coverage` and `text` fields. All records
are explicitly finished and habitable; unknown lease terms remain unknown.
`inspect_photos` honestly reports that this synthetic snapshot has no images.
It never invents visual evidence or calls an image service.

Tasks adapt apartment search scenarios to explicit budget, room, furniture,
workspace and lease constraints. The coordinator submits its ranked shortlist
through exact source URLs in its final answer, as in the demo. There is no extra
submission tool. The harness extracts unique URLs in first-appearance order;
reward checks the exact ranked oracle URLs and that each listing was fetched.
A no-match answer requires an explicit English/Russian no-match statement, no
URLs, and empty searches covering the requested rooms and budget. The phrase
check recognizes common formulations; it is not an unrestricted semantic judge.
The answer must contain only shortlisted listings, so links to rejected or
unrelated candidates fail this benchmark's exact-answer contract.

Frontend tasks use copies of the demo's EN/RU chat source files in
`frontend_seed/`, from the same commit. Tasks adapt notebook, confetti and
magazine themes into exact CSS requirements. The virtual filesystem preserves
per-worker read revisions and rejects stale edits. Reward checks parsed CSS
selector/property values and preservation of all other properties and files.
This is a **source-editing benchmark**, not visual or browser behavior evaluation.
The checker is not a CSS cascade or layout engine; it supports ordinary rules and
media rules, and rejects imports and other unsupported at-rules.

Every task returns a binary native reward. Natural-language claims never set the
score. Episode errors are reported separately; a correct final state can receive
score 1 after an error, while `passed` requires both correctness and clean completion.

## Standalone Evaluation

Install the repository's model dependencies and the small CSS parser dependency:

```sh
uv pip install -e . -r gyms/decomposer_demo/requirements.txt
PYTHONPATH=src:. python -m evals.decomposer_demo.run \
  --model lmrouter/qwen_3_5_4b_unlooped_non_thinking \
  --language ru --concurrency 4 -n 1 --output /tmp/demo-eval-ru
```

The default harness is **Decomposer**. `LocalWorkers` supplies an episode-local
async transport for the existing core `new`, `fork`, `run`, and `wait` tools;
worker histories, budgets, guards and coordinator stopping rules use the current
Decomposer code. No Agent Server is started. Forks copy conversation history and
share their episode's environment, matching the demo's workspace semantics.
Use `--harness react` for a direct single-agent comparison.

Each attempt saves assembled system prompts, coordinator messages, worker run reports, tool actions, final
state and reward. `manifest.json` records progress and settings; `summary.json`
reports overall and per-environment scores and errors. Existing output directories
are rejected. The two languages should be evaluated separately for comparison.

## Verification

```sh
PYTHONPATH=src:. python -m pytest tests/test_decomposer_demo*.py
```

Tests cover all task/language reference states, exact rewards, state isolation,
frontend revision conflicts, pagination, local Decomposer delegation/forking,
and worker error handling. They use scripted models, without
network inference. Live model quality requires
separate evaluation runs.

Expected tool input errors (for example an ISO timestamp instead of `HH:mm`, a
stale revision, or a missing event ID) are returned to the worker as error tool
observations, matching the demo's tool-error flow. Failed arguments are recorded
in `actions`. Non-input failures still stop the worker and mark the episode as an error, even
if the coordinator returns normally. `worker_errors` and `worker_runs` retain
the exception and partial messages. A subsequent worker run starts from the last
successful history.
