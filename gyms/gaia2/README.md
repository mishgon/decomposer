# GAIA2 Evaluation

GAIA2 is used for evaluation only. Both Decomposer and the single-agent baseline
run on Agent Server. Models must already be served; the runner starts Agent Server
and ARE, and stops both when evaluation ends or is interrupted.

## Preparation

The runner uses the pinned `external/gaia2` submodule (`221a9de`) directly.
Preparation creates an isolated Python environment from its lockfile and installs
FastAPI 0.116.1 for the tool broker without ARE's optional GUI dependencies.

```bash
PYTHONPATH=src:. uv run --env-file .env python -m gyms.gaia2.prepare eval \
  --domain execution
```

Use `--domain search` for Search. Each domain contains 160 validation scenarios;
there are no training partitions. Preparation checks dataset and filesystem
checksums and needs Hugging Face access to both GAIA2 datasets.
`--reuse-source` requires existing data; `--skip-runtime` requires an existing ARE environment.
Data, preparation manifests, and the environment live in `artifacts/data/gaia2/`.

## Evaluation

Models use IDs from `src/decomposer/models.py`. The default manager is local
Qwen3.8; the default worker is local Qwen3.5-4B with thinking enabled.
The ARE judge uses `LLM_PROXY_URL` and `LLM_PROXY_MASTER_KEY` from the environment.
`--judge-model` selects its served model name, retaining the existing Llama-3.3-70B
judge by default. Judge credentials are passed through the environment, not saved
in configuration files.
`--agent` is the Agent Server assistant ID: `decomposer` or
`qwen_3_5_4b_thinking`. Both graphs and the agent type description are defined
directly in `agents.py`; the runner passes model and budget settings through
explicit environment variables.

```bash
PYTHONPATH=src:. uv run --env-file .env python -m evals.gaia2.run \
  --agent decomposer \
  --decomposer-model-id vllm/qwen_3_8_flash_next_non_thinking \
  --agent-model-id vllm/qwen_3_5_4b_thinking \
  --agent-base-url http://127.0.0.1:18024/v1 \
  --domains execution search --num-repeats 3 --concurrency 8 \
  --episode-timeout 3600 \
  --agent-run-budget-seconds 60 --agent-shutdown-grace-seconds 60
```

For the single-agent baseline, use `--agent qwen_3_5_4b_thinking --agent-model-id
vllm/qwen_3_5_4b_thinking`. It uses the same agent graph, system prompt, tools,
and simulated clock as Decomposer's workers. Only the episode deadline applies;
it receives no worker run budget. Both modes use `create_model` directly.

`--agent-base-url` and `--decomposer-base-url` override the respective vLLM
server addresses. Without an override, addresses come from `models.py`.
On this VM, port `18024` forwards to Qwen3.5-4B on `occ-cds-2:8024`;
port `8024` serves the local model. The example above uses `occ-cds-2`.

Use repeated `--task SCENARIO_ID` arguments to select tasks, or `--limit N` to
select the first N tasks per domain in stable scenario-ID order.
`--task` requires a single selected domain. `--dry` prints the plan
without creating files or starting services. Task selection requires a prepared dataset manifest.

The episode timeout is ARE's existing wall-clock deadline for each scenario.
The run budget and shutdown grace apply to individual Decomposer worker runs.
There is no harness-level token limit or separate 80-model-call limit. LangGraph
retains a recursion safety limit of 10,000 steps. Worker prompts use our task and reporting instructions, with workspace context,
simulation date, and scenario-specific instructions. Workers return ordinary final
messages; user-interface tools remain hidden.

Both modes use `frozen_turn`: model calls do not advance simulated time,
while tool calls advance it by their measured durations. Unsupported scenario
events are rejected before execution.

## Results and Cleanup

Each evaluation creates `artifacts/evals/gaia2/<timestamp>-<id>/`, with
`execution/` and `search/` subdirectories. Domains run sequentially; episodes
within each domain run with the requested concurrency. `--domains execution` or
`--domains search` selects one domain.

The lower-level `gyms.gaia2.run` runs one domain using `--domain`. An explicit
`--output-dir` must be empty, so previous results cannot be overwritten.
`--port-offset N` shifts the Agent Server port from 8123 for simultaneous
invocations. Each episode uses its own thread and one top-level run.
Only single-turn scenarios are supported.
Cleanup cancels the manager run, cancels its workers, and deletes their threads.

The output includes `output.jsonl`, ARE `lite/` and `hf/` traces, Decomposer
sidecars, service logs, configuration, selected scenario inputs, and
`run_status.json`. Successful collection writes `.eval_done.json` after checking
that every selected task has exactly the requested repeats. Failed episodes stay
in the score denominator; duplicate or missing rollout records fail validation.
Interrupted collections retain their raw artifacts and a failed status.

`evals.gaia2.run` writes `eval_metrics.json` for each domain and a combined
summary at the evaluation root, with pass@1, pass@k, pass^k, and error counts. Recompute metrics without rerunning episodes:

```bash
PYTHONPATH=src:. uv run python -m evals.gaia2.run --metrics-only RUN_DIR
```

`evals.gaia2.trace_stats` reports recorded usage, durations, outcomes, and worker
concurrency. It does not estimate missing token counts. Decomposer sidecars record
top-level agent usage in both modes; Decomposer worker usage is unavailable.
Historical native ARE usage remains available from `lite/` traces. `evals.gaia2.chat_trace` renders individual trajectories.

```bash
PYTHONPATH=src:. uv run python -m evals.gaia2.trace_stats --run-dir RUN_DIR
```

## Code Layout

For an existing Qwen3.5 server running vLLM 0.19.0 with `qwen3_xml`,
`model_proxy.py` provides client-side `qwen3_coder` parsing without restarting
the model server. Run it in that server's Python environment:

```bash
python model_proxy.py --tokenizer /path/to/Qwen3.5-4B --port 18025
```

Forward that loopback port over SSH and pass its URL as `--agent-base-url`.
The proxy uses the model's original chat template and `/v1/completions`, then
parses reasoning and tool calls with vLLM's parsers. A parser-only type hint
keeps nullable strings such as phone numbers from becoming numbers. The proxy
preserves the schemas sent to the model,
sampling parameters, usage, and the remaining context budget. It supports the
non-streaming `auto`/`none` calls used by this integration. Keep the proxy and
tunnel running for the duration of the evaluation.

- `prepare.py`: dependencies, pinned datasets, filesystem, and integrity checks.
- `run.py`: CLI, process lifecycle, parallel episode configuration, and result validation.
- `are.py`: ARE integration, tool HTTP server, episode state, simulated clock, and raw traces.
- `agents.py`: Agent Server entry point and graphs for both harnesses.
- `tools.py`: LangChain tool clients and episode context.
- `prompts.py`: agent instructions and scenario context.

ARE schedules parallel episodes. Each episode has an isolated tool session;
agents within one episode share its environment and serialize tool execution.
