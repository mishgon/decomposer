# SFT build specifications

These specifications, those in `sft/gaia2/specs/` and
`sft/workplace_assistant/specs/`, and the training configs in `sft/configs/`,
target the retired `spawn_subagent`/`wait` Decomposer core. Their datasets and
checkpoints were built at `gaia2-eval` commit `2b1bda8`; rebuild or validate
them only from that commit. The current preparation code accepts only
`new`/`fork`/`run`/`wait` trajectories, excludes `spawn_subagent` traces as
`excluded_legacy_tool_interface`, and resolves the `teacher` prompt profile to
a different prompt. The files stay here unchanged as experiment records.

Specifications for the current core should be added as new files. See
`sft/README.md` for the pipeline.
