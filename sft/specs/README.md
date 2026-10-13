# SFT build specifications

Each file here is a `spec_version: 4` build specification; see `sft/README.md`
for the pipeline. New specifications name the dataset `decomposer-manager-sft`
and set `dataset.version` to the release's semantic version (see *Snapshots,
releases and versioning* in `sft/README.md`).

`decomposer_manager_sft_1.0.0.yaml` is the record of published release 1.0.0
(fingerprint `fb6d86f6…`), kept byte for byte as it built the release. Its
traces were collected with the old core prompt and the builder inserts the
current one, so it reproduces the release only at commit `53a88d8`.

Every other specification and every training config was removed on 2026-10-13;
see *Legacy specs and configs* in `sft/README.md`.
