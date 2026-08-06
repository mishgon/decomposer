# Generic Gemma subagents

This self-contained LangGraph server exposes thinking and non-thinking variants
of Gemma-4-E2B, Gemma-4-E4B, Gemma-4-12B, and Gemma-4-26B-A4B. The agents have
no tools or environment-specific middleware.

Start the required vLLM servers, then run from the repository root:

```bash
scripts/subagents/serve.sh
```

The server listens on `http://127.0.0.1:2024` by default. `HOST`, `PORT`, and
`N_JOBS_PER_WORKER` can override its settings.
