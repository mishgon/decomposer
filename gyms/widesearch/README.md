# WideSearch With BrowseComp Retrieval

This experiment uses the 100 English WideSearch questions and gold tables with
our BrowseComp harness and fixed BrowseComp-Plus corpus. It is a separate setting
from the official WideSearch web-search benchmark. No WideSearch search backend
or paid search key is needed.

Initialize `external/WideSearch` with `git submodule update --init external/WideSearch`.
Apply the empty-row scoring fix with
`git -C external/WideSearch apply ../../gyms/widesearch/upstream.patch`.
This two-line upstream fix returns zero metrics when no rows align, instead of
raising a pandas assignment error. Its diff hash is saved with evaluation provenance.

Install the project and Agent Server dependencies as for BrowseComp, then install
`gyms/widesearch/requirements.txt` in an isolated environment. The upstream
`pandarallel` import requires pandas 2.x; this environment pins pandas 2.2.3.

```bash
python -m gyms.widesearch.dataset
python -m gyms.widesearch.run --agent decomposer_sft_1_2_0 --all -n 3 \
  --retrieval-url http://127.0.0.1:18082 --concurrency 2
python -m evals.widesearch.run artifacts/evals/widesearch/<run-id>
```

Other assistant IDs are `researcher_4b_thinking`, `researcher_27b_thinking`, and
`decomposer_4b_non_thinking`. The 27B profile uses medium thinking. All sampling
parameters come from `src/decomposer/models.py`. Both Decomposers use the same
4B thinking worker, agent type ID and description as BrowseComp.

The harness reuses BrowseComp search/fetch, ten searches per worker, run budgets,
recursion limits, model logging, isolated Agent Servers, trace visualization and
resumption. Only the researcher's final-answer instruction changes from short
JSON to a Markdown table. The Decomposer system prompt is unchanged. The original
question supplies the table schema; gold tables and evaluation metadata never
enter the agent input. The exported `tasks.jsonl` contains only IDs and questions.

Each episode saves `answer.md`, `trace.json`, `trace.html`, `model_calls.jsonl`,
`usage.json` and `result.json`. Episode evaluation remains `pending` until the
separate scoring command runs. Interrupted runs can resume with the same arguments
and `--resume <run-dir>`; completed episodes are retained, including failures.

The dataset revision is `6531a7e5b497d44c8912407e0cb3dc95bd98cc09`.
The upstream evaluator revision is pinned by the submodule gitlink. Deployment
records must also identify the retrieval corpus revision, collection, model
checkpoint, runtime dependencies and server endpoints.
