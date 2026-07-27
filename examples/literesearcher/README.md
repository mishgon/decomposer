# Minimal example

This example runs Decomposer directly and exposes one general-purpose subagent
through a local LangGraph server. Decomposer uses GLM-5.2 through OpenRouter;
the subagent uses LiteResearcher-4B through a local vLLM server.

Set `OPENROUTER_API_KEY` in the environment before running the example.

From the repository root, start vLLM for the subagent:

```bash
scripts/vllm_serve_literesearcher_4b.sh
```

In another terminal, start the subagent server:

```bash
cd examples/literesearcher
uv run langgraph dev --no-browser
```

Then run Decomposer from the repository root:

```bash
uv run python examples/literesearcher/run.py
```

The run loads the first three BrowseComp-Plus prompts, runs them concurrently,
prints their final answers, and saves each Decomposer message history as
human-readable Markdown under `examples/literesearcher/messages/`.
