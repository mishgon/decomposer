# Minimal example

This example runs Decomposer directly and exposes one general-purpose subagent
through a local LangGraph server. Decomposer uses GLM-5.2 through OpenRouter;
the subagent uses Qwen3.6-35B-A3B-FP8 through a local vLLM server.

Set `OPENROUTER_API_KEY` in the environment before running the example.

From the repository root, start vLLM for the subagent:

```bash
scripts/vllm_serve_qwen3_6_35b_a3b_fp8.sh
```

In another terminal, start the subagent server:

```bash
cd examples/minimal
uv run langgraph dev --no-browser
```

Then run Decomposer from the repository root:

```bash
uv run python examples/minimal/run.py
```

The run prints the final answer and saves the Decomposer message history as
human-readable Markdown at `examples/minimal/messages.md`.
