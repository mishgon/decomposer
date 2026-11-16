# OPD Mix

The full-run control recipe is **100% WideSeek depth**. `recipe.json` declares
that composition. A separate four-task smoke exercises DeepDive and both
WideSeek splits. Full-run mixture sampling and difficulty filtering are not
implemented in the control recipe. Unsupported control recipe changes fail explicitly.

## Frozen In-Band Selection

Freeze the qualifying tasks from completed attempts in running searches:

```bash
python -m opd.mix.select_inband \
  --runs /shared/known/eval /shared/unseen/eval \
  --pools /shared/known/tasks.jsonl /shared/unseen/tasks.jsonl \
  --output /shared/selected
python -m opd.mix.overfit --source /shared/selected/tasks.jsonl \
  --output /shared/experiments/mix-selected/train --steps 64 \
  --model /shared/models/decomposer-manager-sft-v1.7.0 \
  --worker-python /shared/runtime/workers/bin/python --gpus 0 1
```

Selection requires four resolved attempts, preserves each search's positive-success
rule, checks checkpoint and task-pool identity, and rejects frozen holdout overlap.
The snapshot records a cutoff, result checksums, outcomes, and source provenance.
Training visits every selected question with balanced exposure: 512 episodes at
batch 8 for 64 updates. Six fixed train probes and Workplace-20 are evaluated at
step zero and every eight updates. There is no separate in-domain heldout panel
for this selected-pool run. Infrastructure errors remain excluded from selection.

## Mixed-Gym Smoke

The smoke selects one task from each of `deepdive-qa_rl`, `deepdive-qa_sft`,
`depth`, and `width` from a normalized candidate JSONL, using seed 42. It excludes
the frozen WideSeek holdouts and duplicate questions. Each task appears twice
in one training epoch: eight episodes and one optimizer update. The manifest
records the selected IDs and source checksum.

Inspect the selection without allocating GPUs:

```bash
python -m opd.mix.smoke_data --source /shared/data/candidates.jsonl \
  --output /shared/experiments/mix-smoke-preview
```

Run from the prepared OPD runtime with two free GPUs, an explicit merged SFT
v1.7.0 model, and the usual lmrouter/ClearML credentials. The shared researcher
harness also requires `AGENTIC_RAG_API_KEY` for its Wikipedia API. Researchers
use OCC-4b without thinking; the teacher remains Qwen Flash Next.

```bash
python -m opd.mix.smoke --source /shared/data/candidates.jsonl \
  --output /shared/experiments/mix-smoke \
  --model /shared/models/decomposer-manager-sft-v1.7.0 \
  --worker-python /shared/runtime/workers/bin/python --gpus 4 5
```

The smoke inherits `opd/wideseek/full.yaml`: batch 8, learning rate 1.5e-5,
LoRA rank 32 / alpha 64, two PPO epochs, 32K total context, temperature 1,
top-p 1 and unrestricted top-k. Native QA accuracy and width item-F1 remain
separate gym scores; distillation supplies the optimization signal. Checkpoint
saving remains enabled. The standalone baseline and validation sweep are
disabled for this smoke; it cannot establish learning or generalization.
It does not run the full in-band candidate pool.

This entry point reuses the [WideSeek pipeline](../wideseek/README.md): the
baseline, teacher alignment checks, task adapters, veRL trainer, checkpoint and
resume logic, recovery, GPU layout, and four-page report generator. The trainer
configuration remains `opd/wideseek/full.yaml` (or `fast.yaml` when the existing
fast-runtime option is selected). The default logging project is
`decomposer-mix-opd`.

## Data and Validation

Pass the same pinned, normalized depth JSONL used by WideSeek OPD. Preparation
retains its task order, question deduplication, frozen hold-out split, four train
probes, and Workplace-20 validation. The existing trainer shuffles with seed 42
and rotates through the pool. No extra sampling or filtering is applied.

Training rows keep `data_source=wideseek/train` and
`agent_name=wideseek_decomposer`; validation retains its existing gym routes.
The prepared data files are identical to the WideSeek recipe. Preparation adds
`mixture.json` and records its checksum and composition in `data/manifest.json`.
This preserves the existing baseline source-hash checks and report format.

```bash
python -m opd.mix.prepare --source /shared/data/depth/tasks.jsonl \
  --output /shared/experiments/mix-data
```

## Launch

Use the existing OPD runtime and credential setup. On a Linux host with an
existing retrieval service and two free GPUs:

```bash
python -m opd.mix.local \
  --root /shared/experiments/mix-depth-001 \
  --assets /shared/wideseek/assets \
  --source /shared/data/depth/tasks.jsonl \
  --model /shared/models/merged-sft \
  --search-url http://127.0.0.1:18080 \
  --gpus 0 1 --training-gpus 1
```

As with WideSeek, `--baseline` reuses a completed baseline for that exact model
and source; `--resume-from` restores a full checkpoint into a new run directory.
Without `--baseline`, the launcher performs the baseline first. When services
are managed externally, `python -m opd.mix.run --help` exposes the existing
lower-level launch options and accepts the same Hydra overrides.

The initial entry points are the local launcher and the lower-level training
runner above. Separate MLSpace and OCCtopus submission wrappers can follow when
needed. The control preserves the existing 64-step, 32K-token recipe before
adding another gym or defining mixture-weight semantics.
