# AGENTS.md

## Communication style guidelines

- Keep your answers short if possible.
- Default to maximum information density.
- Use strictly direct and technical tone.
- Do not use conversational filler, pleasantries, or polite concluding sentences.
- Do not repeat the prompt or explain what you are about to do. Assume the user understands the context.

**Note**: communication style guidelines do not apply to your thinking traces. Think as usual.


## Code style guidelines

- Follow the agile methodology. Keep prototypes minimal. Do not add modes, abstractions, parameters, logging systems, or framework structure before there is a concrete need.
- Prefer official dataset/source instructions and explicit scripts for reproducible setup.
- Run lightweight checks after edits, such as `uv run python -m py_compile ...`, before expensive model evaluations.
- NEVER silently patch installed packages without the user's consent.
