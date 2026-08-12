# Toolathlon-Gym subagent server

This container-local LangGraph server exposes the
`gemma_4_26b_a4b_thinking` assistant. Its model endpoint is configured with
`GEMMA_4_26B_A4B_BASE_URL`.

The server reads the prepared task configuration from
`$TOOLATHLON_DATA_DIR/runtime.json`. When it starts, it opens one persistent
stdio session for each required MCP server and shares the loaded tools between
the graph. It closes every session when it stops.
