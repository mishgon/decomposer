# Minimal example

This example runs Decomposer directly and exposes one general-purpose subagent
through a local LangGraph server. Both model roles use hosted lmrouter inference.

Set `LLM_PROXY_MASTER_KEY` before starting Python. The example uses the explicit
Flash Next non-thinking teacher and Qwen4B-unlooped thinking worker in
`src/decomposer/models.py`.

Start the subagent server from the repository root:

```bash
scripts/subagents/serve.sh
```

Then run Decomposer from the repository root:

```bash
uv run python examples/minimal/run.py
```

The run prints the final answer and saves the Decomposer message history as
human-readable Markdown at `examples/minimal/messages.md`.
