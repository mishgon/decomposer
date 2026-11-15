# DeepDive

This gym adapts [zai-org/DeepDive](https://huggingface.co/datasets/zai-org/DeepDive)
to the same hosted Agentic RAG tools as WideSeek, using the Search API's
`wikipedia` source. Results describe this adaptation, not the original
DeepDive benchmark setting.

## Prepare Tasks

Use the existing WideSeek runtime and a Parquet reader (`pyarrow`, already included
through the project's `train` extra). From the repository root:

```bash
python -m gyms.deepdive.prepare --split qa_rl
python -m gyms.deepdive.prepare --split qa_sft
```

The dataset is pinned to `36b28945eaf271e64003e7fcd4ddb81af0828eb5`:
2,234 `qa_rl` tasks and 1,016 `qa_sft` tasks. Each split gets a separate directory
under `artifacts/gyms/deepdive/data/`, containing the original Parquet file,
normalized `tasks.jsonl`, and a provenance/checksum manifest. Preparation retains
every question, including duplicates and questions that may be unsolvable from
the 2018 corpus. Solvability filtering comes later.

IDs include the split and original dataset ID. Reference answers stay in task
data for grading; agent input contains only the question and the shared boxed-answer
format instruction. Supplied conversations and SFT trajectories are not passed to
the agent. The raw Parquet preserves original fields for audit.

## Run Episodes

Configure the [WideSeek Search API credentials](../wideseek/README.md).
The same workers can serve DeepDive, or start dedicated workers:

```bash
source gyms/wideseek/env.sh
python -m langgraph_cli dev --config gyms/deepdive/langgraph.json \
  --host 127.0.0.1 --port 18085 --no-browser --no-reload --n-jobs-per-worker 16
```

In another shell with the same model and Search API credentials:

```bash
python -m gyms.deepdive.run --agent decomposer --limit 2 --concurrency 2 \
  --worker-url http://127.0.0.1:18085 \
  --output artifacts/data/deepdive/smoke
```

`qa_rl` is the default dataset. Select the other split through
`--data artifacts/gyms/deepdive/data/qa_sft/tasks.jsonl`.

The gym reuses WideSeek's Decomposer/React agents and QA evaluator. The coordinator
and judge use hosted Qwen3.8 Flash Next; researchers use hosted `occ-4b`
without thinking. `react` uses the same researcher directly. Researchers use
`search`, `open`, `find`, and `finish`, with 180 seconds of work plus 180 seconds
shutdown grace. A valid `finish` is required. Each run allows 32 model calls,
with the last reserved for `finish`, as in the colleagues' harness;
context/output limits match the colleagues' harness (65,536 tokens).
The manager keeps recursion 410 and the episode deadline is 45 minutes.

The QA evaluator extracts the boxed final answer and asks the judge whether it
matches the reference, returning `qa_accuracy` (0 or 1). A missing boxed answer
scores zero. Judge failures remain unscored, with raw judge calls saved. This is
the shared WideSeek QA grading policy, not a claim to reproduce DeepDive's official
evaluation. Raw episode status/error, model calls, tool calls, researcher traces,
and usage remain available; infrastructure errors must be inspected separately
from scored model failures.

See [trace collection](../../sft/deepdive/README.md) for the dedicated launcher.
