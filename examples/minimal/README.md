# Minimal example

This example hosts Decomposer and a general-purpose agent on one local LangGraph
server. Both use hosted models through `decomposer.models.create_model`.
`agents.py` defines the factories; `langgraph.json` registers them.

From the repository root, with `LLM_PROXY_MASTER_KEY` configured:

```bash
uv run python examples/minimal/run.py
```

`run.py` uses `decomposer.agent_server` to start the server on port 2024 and wait
for readiness. It runs Decomposer through the SDK; the context manager stops
the server on exit, including errors. Port 2024 must be free.
The development environment includes `langgraph-cli[inmem]`.

The final answer is printed. `invoke_and_capture` captures the latest state and
confirmed run statuses before the server stops, including on failure.
The state is written to `trace.json`; `decomposer.visualization.write_trace_html`
generates `trace.html`. An HTML error produces a warning; an agent error is
re-raised after saving its trace.
