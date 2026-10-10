# SFT build specifications

These specifications, those in `sft/gaia2/specs/` and
`sft/workplace_assistant/specs/`, and the training configs in `sft/configs/`,
target the retired `spawn_subagent`/`wait` Decomposer core. Their datasets and
checkpoints were built at `gaia2-eval` commit `2b1bda8`; rebuild or validate
them only from that commit. The current preparation code accepts only
`new`/`fork`/`run`/`wait` trajectories, excludes `spawn_subagent` traces as
`excluded_legacy_tool_interface`, and resolves the `teacher` prompt profile to
a different prompt. The files stay here unchanged as experiment records.

Specifications for the current core should be added as new files. The current
core names the tool arguments `agent_type_id` and `agent_id`; see
`sft/README.md` for the pipeline. The current code builds only
`spec_version` 4 specifications, which name each source by its snapshot digest;
the first is `decomposer_mixed_qwen38_qwen35_4b_unloop_nonthinking_v2.yaml`.
`decomposer_mixed_qwen38_qwen35_4b_unloop_nonthinking_v1_32k.yaml`, the first
specification for the current core, now loads only at the commit that built it.
Records keep each gym's native tool schemas, and a specification that names no
prompt profile trains with the `teacher` prompt.
New specifications name the dataset `decomposer-manager-sft` and set
`dataset.version` to the release's semantic version; see *Snapshots, releases
and versioning* in `sft/README.md`. The first is
`decomposer_manager_sft_1.0.0.yaml`.
