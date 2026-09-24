# Generic Qwen subagents

This self-contained LangGraph server exposes Qwen3.5-4B with thinking disabled.
The agent has no tools or environment-specific middleware.

Start the required vLLM server with `scripts/vllm/serve_qwen_3_5_4b.sh`
(port 8024), then run
from the repository root:

```bash
scripts/subagents/serve.sh
```

The server listens on `http://127.0.0.1:2024` by default. `HOST`, `PORT`, and
`N_JOBS_PER_WORKER` can override its settings.
