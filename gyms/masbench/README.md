# MASBench Evaluation

This harness runs the five test axes from [Salesforce/MASBench](https://huggingface.co/datasets/Salesforce/MASBench)
at revision `f1d57e20c304c5e42fa77c1a0412de2bc7ad52a3` (4,583 tasks).
It compares `single`, `base_manager`, and `sft_manager` on identical task inputs.
The first two use Qwen3.5-4B; the SFT manager uses Decomposer 1.5.0.
Both managers are non-thinking. The single agent and workers are thinking agents.

The managers keep the core system prompt. MASBench rules are prepended to their
user input. The single agent and workers use `AGENT_SYSTEM_PROMPT` plus the rules
as their system prompt. Only managers have tools (the existing orchestration
tools). Workers receive the subtasks composed by their manager. Ground truth,
dependency graphs, and reference solutions never enter model inputs.

Arithmetic uses modulo 23 and unspecified parameters default to zero, as in
[the authors' prompt](https://github.com/SalesforceAIResearch/MAS-Orchestra/blob/0527c70a508eb49ed42d5dd49a91ea857551e242/mas_r1_reasoner/agents/blocks_harmony/igsm/cot.py).
Magic-number retrieval preserves the exact number. Multi-answer scoring follows
the authors' ordered, exact-cardinality boxed-answer protocol. All answers must
match for a task to pass. Single-answer scoring uses the last box, or the full
response when no box exists. Only final content is scored, never reasoning.

The comparator is `math-verify==0.8.0`, one backend used by the
[upstream scorer](https://github.com/SalesforceAIResearch/MAS-Orchestra/blob/0527c70a508eb49ed42d5dd49a91ea857551e242/mas_r1_reasoner/rewards/utils/compute_score.py).
This adapter omits upstream's additional MathScorer fallback and optional LLM
judge, so it is not an exact reproduction of the paper's scoring implementation.
We report answer correctness, excluding the training reward for executing a
generated orchestration graph.

## Run

Install the project and gym requirements in an existing Python 3.12 environment:

```bash
uv pip install --python .venv/bin/python -e . -r gyms/masbench/requirements.txt
export MASBENCH_BASE_URL=http://127.0.0.1:8024/v1
export MASBENCH_SFT_URL=http://127.0.0.1:18024/v1
.venv/bin/python -m gyms.masbench.prepare --output-dir artifacts/data/masbench/dataset
.venv/bin/python -m gyms.masbench.run --smoke \
  --data artifacts/data/masbench/dataset/tasks.jsonl \
  --output-dir artifacts/data/masbench/smoke --concurrency 3
.venv/bin/python -m gyms.masbench.run --all \
  --data artifacts/data/masbench/dataset/tasks.jsonl \
  --output-dir artifacts/data/masbench/full --concurrency 12
.venv/bin/python -m evals.masbench.summarize artifacts/data/masbench/full \
  --output artifacts/evals/masbench/full.json
```

The output directory must be new. Smoke runs select the first task in each axis.
To change concurrency on an existing run, send SIGINT to its coordinator and
wait for it to exit, then repeat the command with `--resume` and the new
`--concurrency`. Keep the same output directory, data, agents, repetitions,
endpoints and timeout. Completed attempts, including failures, are preserved.
Unfinished attempts are archived under `interrupted/` before they are retried.
The manifest records each resumption and its code revision and concurrency.
Full runs default to one sample per task and configuration (13,749 episodes).
`--repetitions 8` measures average correctness over eight attempts, not pass@8.
The default episode timeout is 1,800 seconds for all configurations, with a
recursion limit of 410. Workers have no separate time limit. This comparison does
not equalize token usage; usage is saved for subsequent cost comparisons.

Each episode has an isolated Agent Server with 64 execution slots. Cleanup stops
its outstanding worker runs. Three consecutive episode errors stop scheduling
new work. Errors and timeouts count as failures; summaries include their counts
and distinguish completed episodes from the planned denominator. Partial runs
must not be compared as full benchmark results.

Artifacts include source dataset rows, model input, append-only model calls,
Agent Server logs, full trace state, token usage, final answer and evaluation.
The manifest records the Git revision, dataset revision, task-file hash, endpoints,
package versions and limits. Retain vLLM startup logs and model provenance beside
the run to identify the weights and serving configuration.

## H200 Serving Example

Run from reviewed `main`. Recheck availability before selecting GPUs. The paths
below refer to existing model snapshots on hertz-2. Use a large data mount for
logs and evaluation output rather than the root filesystem.

```bash
CUDA_VISIBLE_DEVICES=2,4,5,6 .venv/bin/vllm serve \
  "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a" \
  --served-model-name Qwen/Qwen3.5-4B --host 127.0.0.1 --port 8024 \
  --data-parallel-size 4 --data-parallel-backend mp --data-parallel-rpc-port 19560 \
  --max-model-len 131072 --gpu-memory-utilization 0.90 --max-num-seqs 256 \
  --language-model-only --reasoning-parser qwen3 --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder --enable-prefix-caching --enforce-eager \
  --default-chat-template-kwargs '{"enable_thinking":true}'

CUDA_VISIBLE_DEVICES=7 .venv/bin/vllm serve \
  artifacts/models/decomposer-manager-qwen35-4b-sft-1.5.0/1.5.0/merged \
  --served-model-name decomposer-manager-qwen35-4b-sft-1.5.0 --host 127.0.0.1 --port 18024 \
  --max-model-len 131072 --gpu-memory-utilization 0.70 --max-num-seqs 64 \
  --language-model-only --reasoning-parser qwen3 --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder --enable-prefix-caching --enforce-eager \
  --default-chat-template-kwargs '{"enable_thinking":false}'
```

Four base-model replicas at 90% memory and one SFT replica at 70% allocate about
5.1 times as much GPU memory to the base model. This is a capacity allocation,
not a throughput guarantee. The base endpoint also serves the non-thinking
baseline manager; `create_model` overrides thinking per request.
