# Generic Qwen subagents

This LangGraph server exposes the hosted Qwen3.5-4B-unlooped thinking model
from `src/decomposer/models.py`. Set `LLM_PROXY_MASTER_KEY` before starting it.
The agent has no tools or environment-specific middleware.

From the repository root:

```bash
scripts/subagents/serve.sh
```

The server listens on `http://127.0.0.1:2024` by default. `HOST`, `PORT`, and
`N_JOBS_PER_WORKER` can override its settings.
