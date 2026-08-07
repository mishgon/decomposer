# Toolathlon-Gym subagent server

This container-local LangGraph server exposes the eight Gemma assistant IDs.
Its model endpoints are configured with `GEMMA_4_E2B_BASE_URL`,
`GEMMA_4_E4B_BASE_URL`, `GEMMA_4_12B_BASE_URL`, and
`GEMMA_4_26B_A4B_BASE_URL`.

The server reads the prepared task configuration from
`$TOOLATHLON_DATA_DIR/runtime.json`. When it starts, it opens one persistent
stdio session for each required MCP server and shares the loaded tools between
all eight graphs. It closes every session when it stops.
