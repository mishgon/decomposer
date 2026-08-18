# Workplace Assistant gym

This package owns Workplace Assistant dataset preparation, local execution,
MLSpace submission, experiment profiles, and conversion of successful
Decomposer rollouts into canonical SFT releases. The pinned `external/Gym`
submodule remains an upstream runtime dependency; this package does not use its
baseline job shell script.

## Prepare evaluation inputs

Preparation wraps the pinned NeMo-Gym dataset converter, writes the raw and
Decomposer-specific variants outside the repository, prepares lock-keyed Gym
and component environments, validates the selected models, and writes one
manifest per experiment and split.

```bash
# Prepare all registered train experiments.
.venv/bin/python -m gyms.workplace_assistant.prepare eval --split train

# Prepare one validation experiment, reusing an existing validated source file.
.venv/bin/python -m gyms.workplace_assistant.prepare eval \
  --split validation \
  --experiment gemma4-e2b-it-non-thinking \
  --reuse-source
```

The train and validation splits contain 1,255 and 545 tasks respectively.
`--split train` is the default.

## Run locally

The local runner does not submit an MLSpace job. It starts the selected model
services, agent service, and Gym servers on the current machine, performs one
Gym evaluation, validates the output, and stops every child process.

```bash
# Simple agent backed by one local policy vLLM.
.venv/bin/python -m gyms.workplace_assistant.run \
  --experiment gemma4-e2b-it-non-thinking \
  --split train

# Decomposer backed by an external teacher and local subagent services.
OPENROUTER_API_KEY_DECOMPOSER=... \
HTTPS_PROXY=... \
.venv/bin/python -m gyms.workplace_assistant.run \
  --experiment glm-5-2-gemma4-26b-a4b-non-thinking \
  --split train

# One-task local smoke run. There is no automatic smoke pass.
.venv/bin/python -m gyms.workplace_assistant.run \
  --experiment gemma4-e2b-it-non-thinking \
  --split validation \
  --limit 1

# Isolate a local run from shared MLSpace outputs and select physical GPU 2.
.venv/bin/python -m gyms.workplace_assistant.run \
  --experiment deepseek-v4-flash-0731-gemma4-e4b-thinking \
  --split train \
  --num-repeats 3 \
  --cuda-visible-devices 2 \
  --output-dir /path/to/local-results
```

Use `--dry` to print the complete service and Gym commands without starting
processes. `--output-dir` routes every result, log, status file, and completion
marker beneath an explicit directory. Partial outputs resume by default.
`--force` archives the previous attempt before starting fresh.

## Submit MLSpace jobs

The launcher selects experiments from the same registry, skips completed and
active runs, stages a clean Git revision, and dispatches the shared local runner
inside each worker. Run it with the Python environment that provides `mls`:

```bash
MLSPY=/mnt/shared_ru.ml.SZ-5_000264/sukhorukov/.venv-mls/bin/python

$MLSPY -m gyms.workplace_assistant.run_eval \
  --experiment gemma4-e2b-it-non-thinking \
  --split train \
  --author-name sukhorukov \
  --dry

$MLSPY -m gyms.workplace_assistant.run_eval \
  --filter qwen35 \
  --split validation \
  --author-name sukhorukov
```

Dry runs do not stage code or submit jobs and redact credentials from printed
payloads. The live allocation's selected instance types are kept in
`experiments.py` as the single source of truth.

## Prepare SFT releases

The benchmark-neutral canonical schema, builder, and NeMo-Gym adapter remain in
`data.sft`. Workplace-specific build specifications and their public commands
live here:

```bash
uv run --group train python -m gyms.workplace_assistant.prepare sft \
  --dataset workplace-all-v3

uv run --group train python -m gyms.workplace_assistant.prepare sft \
  --dataset workplace-26b-nonthinking-v3
```

## Artifact layout

All generated data is outside the worktree under
`/mnt/shared_ru.ml.SZ-5_000264/sukhorukov/decomposer_artifacts`:

```text
evaluation/data/workplace_assistant/       source files and preparation manifests
evaluation/results/<split>/<run-name>/     rollouts, logs, status, completion marker
datasets/sft/<dataset-id>/<version>/        immutable canonical SFT releases
venvs/gym/<lock-hash>/                      shared Gym CLI runtime
venvs/workplace-assistant/<lock-hash>/      shared Gym component runtimes
code/<git-commit>/                          immutable MLSpace code stages
```
