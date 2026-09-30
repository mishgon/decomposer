# Toolathlon-Gym subagent server

This container-local LangGraph server exposes one assistant:

Assistant ID: `qwen_3_5_4b_unlooped_thinking`.

Its model is created by `create_model("qwen_3_5_4b_unlooped_thinking")` in
[models.py](../../../src/decomposer/models.py). Temperature is 0.6, top-p 0.95,
top-k 20. Thinking and reasoning preservation are enabled;
other sampling parameters retain provider defaults. There are no per-model
environment variables or model factories in this graph.

The server reads the prepared task configuration from
`$TOOLATHLON_DATA_DIR/runtime.json`. When it starts, it opens one persistent
stdio session for each required MCP server and shares the loaded tools between
the graphs. It closes every session when it stops.
