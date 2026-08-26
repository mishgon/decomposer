# Toolathlon benchmark subagent server

Container-local LangGraph server exposing the `qwen_3_5_4b_non_thinking`
assistant. Its model endpoint is configured with `QWEN_3_5_4B_BASE_URL`
(defaulting to `http://127.0.0.1:8030/v1`; the task container runs with
`--network host`, so the host vLLM server is reachable on the container
loopback).

On startup the server opens one persistent MCP SSE session to the Toolathlon
container gateway at `TOOLATHLON_GATEWAY_URL` (for example
`http://127.0.0.1:43210/sse`) and shares the loaded tools between the
graphs. It closes the session when it stops.
