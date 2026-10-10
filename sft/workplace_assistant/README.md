# Workplace Assistant SFT records

`specs/` records how the Workplace-only SFT releases were built. These specs
build only at the commits that built their releases (see *Legacy specs and
configs* in `sft/README.md`). Five of them (`workplace-all-v3`,
`workplace-26b-nonthinking-v3` and the three DeepSeek E4B releases) were built
by name with the since-removed `sft.workplace_assistant.prepare`. Current
releases mix Workplace with the other gyms through a snapshot-pinned spec in
`sft/specs/`.
