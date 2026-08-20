# Gaia2 execution evaluation

This package owns Gaia2 evaluation data preparation, the Decomposer external
agent adapter, local execution, and MLSpace submission. It supports the
`validation` split of the `execution` domain only; Gaia2 is not used for trace
generation or SFT data.

The current integration uses Gaia commit
`3bee736488864e028231755ce2ee32a7065e8648`. Until that commit can be published
and added as `external/gaia2`, preparation reads it from the existing local Gaia
repository and materializes a clean, immutable checkout under Decomposer
artifacts. Every preparation and run manifest records both repository commits.

## Prepare data and runtimes

```bash
.venv/bin/python -m gyms.gaia2.prepare eval \
  --split validation \
  --domain execution
```

Preparation loads revision
`78ea3bdbdeec2bdcd6afa5420915d8a22f23ed99` of
`meta-agents-research-environments/gaia2`. It reuses the shared Hugging Face
cache when present and otherwise downloads the dataset, then writes the 160
serialized scenarios unchanged to:

```text
/mnt/shared_ru.ml.SZ-5_000264/sukhorukov/decomposer_artifacts/
  evaluation/data/gaia2/<dataset-revision>/validation/execution/
```

The manifest is outside the scenario directory, so ARE never interprets it as
a task. `--reuse-source` forbids Hugging Face access and only validates the
already materialized files. `--skip-runtime` validates the lock-keyed runtime
without creating it. Select one experiment with repeated `--experiment` flags.

Preparation also mirrors revision
`132e26376f5e963bb59f64bcccdd02188cb08dee` of
`meta-agents-research-environments/gaia2_filesystem` into the artifact tree and
points `DEMO_FS_PATH` at its local `demo_filesystem/`. Evaluation therefore
does not lazily fetch execution-task files from Hugging Face.

The Gaia runtime is created with `uv sync --frozen` under
`decomposer_artifacts/venvs/gaia2/<uv-lock-hash>`. Evaluation never falls back
to a live Hugging Face dataset.

## Run locally

The registered experiments are:

- `gemma4-e4b-sft-deepseek-e4b-v1-8k-non-thinking-gemma4-e4b-thinking`:
  tuned non-thinking E4B manager on the first GPU and vanilla thinking E4B
  worker on the second GPU.
- `gemma4-e4b-it-thinking`: vanilla thinking E4B simple agent on one GPU.

```bash
# Decomposer, three attempts per scenario.
.venv/bin/python -m gyms.gaia2.run \
  --experiment gemma4-e4b-sft-deepseek-e4b-v1-8k-non-thinking-gemma4-e4b-thinking \
  --num-repeats 3 \
  --cuda-visible-devices 6,7

# Simple-agent one-scenario smoke.
.venv/bin/python -m gyms.gaia2.run \
  --experiment gemma4-e4b-it-thinking \
  --num-repeats 1 \
  --limit 1 \
  --cuda-visible-devices 7
```

`LLM_PROXY_URL` and `LLM_PROXY_MASTER_KEY` must point to the canonical proxy
judge. Use `--dry` to print every service and ARE command without starting
processes. `--output-dir` isolates an ad-hoc run. A rerun skips a completed
marker; `--force` archives the previous attempt before starting again.

Decomposer evaluation always uses the current student prompt. Its workers see
the original scenario-bound ARE tools through an authenticated loopback broker;
the final user-interface tools remain manager-only. Multiple tool calls are
valid and each is executed once under the scenario lock.

ARE emits one `output.jsonl` row per attempted rollout. Missing/empty manager
answers, uncollected subagents, recursion limits, and scenario timeouts remain
failed rows with score zero and do not stop the other attempts. A service death,
invalid dataset/runtime, nonzero ARE exit, or incomplete rollout cardinality is
a systemic failure and does not create `.eval_done.json`.

## Submit MLSpace jobs

```bash
/mnt/shared_ru.ml.SZ-5_000264/sukhorukov/.venv-mls/bin/python \
  -m gyms.gaia2.run_eval \
  --experiment gemma4-e4b-it-thinking \
  --num-repeats 3 \
  --author-name sukhorukov \
  --dry
```

The launcher uses one `a100plus.1gpu.80vG.12C.182G` instance for the simple
agent and one `a100plus.2gpu.80vG.24C.364G` instance for Decomposer. It skips
completed results and matching Pending/Running jobs, refuses a dirty worktree
for real submissions, stages Decomposer by Git SHA, and reuses the independently
staged Gaia SHA. Dry-run payloads redact judge credentials and submit nothing.

Canonical results are stored under:

```text
decomposer_artifacts/evaluation/gaia2/results/
  validation/execution/<experiment>-n3/
```

Each completed directory contains native `lite/` and `hf/` traces,
`output.jsonl`, service logs, redacted Decomposer sidecars when applicable,
`metrics.json`, `run_status.json`, and `.eval_done.json`.
