# OPD Mix

The initial recipe is **100% WideSeek depth**. `recipe.json` declares that
composition; additional gyms, mixture sampling, and difficulty filtering are
not implemented yet. Unsupported recipe changes fail explicitly.

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
