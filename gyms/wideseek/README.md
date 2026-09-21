# WideSeek gym

Simple agent vs Decomposer using hosted Qwen3.5-4B non-thinking and fixed
Wiki-2018 retrieval. Based on `dev`; no Toolathlon/RL changes are required.
The first comparison uses **width/table tasks**. Depth/hybrid data can be prepared,
but the runner deliberately rejects their non-table tasks until their native QA
scorer is wired. This is not yet a full WideSeek benchmark reproduction.

## Setup

On a Linux host with about 160 GB of disk available for the corpus/index and
ample RAM (the page lookup loads the 27 GB JSONL into memory):

```bash
bash gyms/wideseek/setup.sh
source .venv/bin/activate
export WS_ASSETS=/large/disk/wideseek/assets
python -m gyms.wideseek.assets --root "$WS_ASSETS"
```

The corpus and upstream implementation revisions are pinned. The E5 revision is
resolved once and recorded. Downloads resume using the Hugging Face cache.
All LLM calls use the hosted router; no local GPU inference, CUDA build, or
OpenRouter is used. The encoder runs on CPU with FP32, versus upstream's GPU
FP16: same corpus/index/E5 pipeline, not bit-for-bit retrieval equivalence.

Start these three foreground services in separate terminals (or with your usual
`nohup`/tmux wrapper and logs under `artifacts/gyms/wideseek/setup/`):

```bash
WS_ASSETS=/large/disk/wideseek/assets bash gyms/wideseek/serve.sh qdrant
WS_ASSETS=/large/disk/wideseek/assets bash gyms/wideseek/serve.sh retrieval
bash gyms/wideseek/serve.sh workers
```

They bind localhost ports 16333/16334 (Qdrant), 18080 (retrieval), 18081 (workers).
`env.sh` loads `LMROUTER_ENV`, defaulting to the home environment file. Override
`WS_MODEL_URL`/`WS_MODEL_HOST` for another deployment. The optional host/IP
mapping retains TLS hostname verification; it does not disable TLS validation.

## Data

```bash
python3 -m gyms.wideseek.prepare --source width
python3 -m unittest discover -s tests -p 'test_wideseek_data.py'
```

Uses only the Python standard library. Source choices: `width`, `depth`, `hybrid`.
Each has 20,000 examples; hybrid mixes the other sources, so do not concatenate
them as independent data. Preparation pins the HF revision, preserves raw rows,
checks the schema, and records file hashes. It refuses to overwrite a directory;
only a completed dataset has `manifest.json`.

Artifacts: `artifacts/gyms/wideseek/data/<source>/`. References stay in dataset
files for scoring; `agent_input` supplies only the question. Split future held-out
panels by question hash (and inspect paraphrase overlap), not by file membership.
Evaluation on this training dataset is a development measurement, not a published
WideSearch benchmark result.

## Run and resume

```bash
source gyms/wideseek/env.sh
.venv/bin/python -m gyms.wideseek.run \
  --output artifacts/gyms/wideseek/runs/qwen4b-smoke \
  --mode both --limit 2 -n 3 --concurrency 2
```

Repeat the exact command with `--resume` to skip completed attempts. Interrupted
attempt directories are retained and retried separately. For width collection,
choose `--mode decomposer --limit 20000` and the desired `-n`; the default is only
a two-task smoke, not a full dataset run. Do not run two writers on the same output.

Each attempt has a 15-minute agent timeout and shared **64 model calls / 64,000
generated tokens**, covering decomposer plus all workers. Per-call output cap is
4,096 tokens, an explicit smoke resource limit, not the model's recommended full
output allowance. CLI flags control total budgets/timeouts. Both modes receive
the same budgets and researcher tools. Recursion limit is 410 for every agent.
Sampling follows Qwen's general non-thinking recommendation: temperature .7,
top_p .8, top_k 20, min_p 0, presence penalty 1.5, repetition penalty 1.
Judge calls are greedy and separately accounted, outside the agent budget.

Under one run directory:

- `manifest.json`: task IDs, data/source hashes, code revision, package versions,
  model/generation/retrieval settings and budgets.
- `<mode>/<task>/attempt-NNN/`: full model requests/responses (including provider
  usage), tool I/O, final graph state, subagent states, judge I/O and `result.json`.
- `<mode>-summary.json`: item-F1 and completion/error accounting. Judge failures
  remain unscored; the explicitly named `infra_zero` aggregate also counts them
  as zero. Native partial score is not a binary pass rate.

The judge is also hosted Qwen3.5-4B. Its scores are **diagnostic, not directly
paper-comparable**. Reference answers never enter agent inputs. The scorer keeps
upstream strict Markdown extraction: an unfenced table may score zero even if it
looks readable. Do not silently relax this between agent modes.

## Tests and upstream references

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_wideseek*.py'
```

- Data: https://huggingface.co/datasets/RLinf/WideSeek-R1-train-data
- Tools: https://rlinf.readthedocs.io/en/latest/rst_source/examples/agentic/wideseek_r1/tools.html
- Scoring: `rlinf/agents/wideseek_r1/utils/reward.py` in https://github.com/RLinf/RLinf
- Benchmark: https://github.com/RLinf/WideSeek-R1-Eval

`vendor/table_reward.py` copies the table-scoring functions unchanged from the
pinned RLinf revision, retaining its Apache license. The wrapper detects malformed
judge replies/API errors instead of accepting upstream's silent zero fallback.
Tests cover perfect, incomplete, wrong and malformed answers and shared-budget
concurrency. Live retrieval and two-agent smoke must also pass before collection.
