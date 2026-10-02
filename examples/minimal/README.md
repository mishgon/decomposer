# Minimal example

This example hosts Decomposer and a general-purpose agent on one local LangGraph
server. Both use hosted models through `decomposer.models.create_model`.
`agents.py` defines the factories; `langgraph.json` registers them.

From the repository root, with `LLM_PROXY_MASTER_KEY` configured:

```bash
uv run python examples/minimal/run.py
```

`run.py` uses `utils.agent_server` to start the server on port 2024 and wait
for readiness. It runs Decomposer through the SDK; the context manager stops
the server on exit, including errors. Port 2024 must be free.
The development environment includes `langgraph-cli[inmem]`.

The final answer is printed and the conversation is saved to `messages.md`.
For private model access, follow the [lmrouter setup](../../README.md#hosted-models-and-private-lmrouter-access)
before running the example.
