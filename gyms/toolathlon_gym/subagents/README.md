# Toolathlon-Gym subagent server

This container-local LangGraph server exposes these assistants:

| Assistant ID | Endpoint variable | Default host port |
| --- | --- | --- |
| `gemma_4_26b_a4b_thinking` | `GEMMA_4_26B_A4B_BASE_URL` | 8023 |
| `gemma_4_26b_a4b_non_thinking` | `GEMMA_4_26B_A4B_BASE_URL` | 8023 |
| `qwen_3_5_4b_non_thinking` | `QWEN_3_5_4B_BASE_URL` | 8024 |

The server reads the prepared task configuration from
`$TOOLATHLON_DATA_DIR/runtime.json`. When it starts, it opens one persistent
stdio session for each required MCP server and shares the loaded tools between
the graphs. It closes every session when it stops.
