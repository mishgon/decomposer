# GAIA2 SFT trace preparation

GAIA2-specific pieces of the shared SFT pipeline in `sft/`:

- `adapter.py` — the `gaia2` source adapter. It reads finished GAIA2 runs from
  `gyms/gaia2` (evaluation runs with `.eval_done.json`, `output.jsonl` and
  `decomposer_sidecars/`, or trace-generation runs with `.trace_done.json` and
  `trace_manifest.jsonl`) into canonical rollouts. It is registered in
  `sft/adapters/registry.py` under the name `gaia2`.
- `snapshot_trace_prefix.py` — freezes the selected logical rounds of a
  trace-generation run into an immutable `trace_manifest.jsonl` snapshot that a
  spec can pin while the run keeps growing:

  ```bash
  python -m sft.gaia2.snapshot_trace_prefix \
    --source <trace-run-dir> --output <snapshot-dir> \
    --logical-rollout-number 1 --logical-rollout-number 2
  ```
- `specs/` and `split_manifests/` — GAIA2-only release specs and their pinned
  splits. Build them with
  `python -m sft.prepare --spec sft/gaia2/specs/<spec>.yaml --output-root <dir>`.
  Mixed-gym specs live in `sft/specs/`.
