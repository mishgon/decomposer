# AGENTS.md

## Code style guidelines

- Follow the agile methodology. Keep prototypes minimal. Do not add modes, abstractions, parameters, logging systems, or framework structure before there is a concrete need.
- Prefer small, incremental changes. Implement one logical block at a time and keep each change easy to review.
- Prefer simple, explicit functions over clever abstractions. If two concepts are different, name them separately; if one concept is enough, do not split it into multiple helper functions.
- Avoid legacy compatibility code.
- Prefer official dataset/source instructions and explicit scripts for reproducible setup.
- Run lightweight checks after edits, such as `uv run python -m py_compile ...`, before expensive model evaluations.
