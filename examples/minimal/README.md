# Minimal example

This example runs Decomposer directly and exposes one general-purpose subagent
through a local LangGraph server. Decomposer uses DeepSeek V4 Flash 0731 through OpenRouter;
the subagent uses Gemma-4-E4B through a local vLLM server.

Set `OPENROUTER_API_KEY` in the environment before running the example.

From the repository root, start vLLM for the subagent:

```bash
scripts/vllm/serve_gemma_4_e4b.sh
```

In another terminal, start the subagent server from the repository root:

```bash
scripts/subagents/serve.sh
```

Then run Decomposer from the repository root:

```bash
uv run python examples/minimal/run.py
```

The run prints the final answer and saves the Decomposer message history as
human-readable Markdown at `examples/minimal/messages.md`.
