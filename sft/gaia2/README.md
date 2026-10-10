# GAIA2 SFT records

`specs/` and `split_manifests/` record how the GAIA2 SFT releases were built:
the GAIA2-only release spec, and the pinned task split that it and the mixed
v2 and v3 GAIA2 specs in `sft/specs/` used. The GAIA2 adapter and the
trace-prefix snapshot tool that read those runs are gone from the pipeline, so
these specs build only at the commit that built their releases (see *Legacy
specs and configs* in `sft/README.md`).
