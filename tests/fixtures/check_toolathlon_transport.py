"""Against toolathlon_transport.json on localhost:2024: failed worker must not break wait."""
import asyncio
import json
from types import SimpleNamespace
from langgraph_sdk import get_client
from decomposer.core import _build_wait_tool


async def main():
    client = get_client(url="http://127.0.0.1:2024")
    thread = await client.threads.create()
    run = await client.runs.create(thread["thread_id"], "test",
        input={"messages": [{"role": "user", "content": "Read page"}]})
    for _ in range(100):
        state = await client.runs.get(thread["thread_id"], run["run_id"])
        if state["status"] in {"error", "success"}:
            break
        await asyncio.sleep(.2)
    assert state["status"] == "error", state
    history = await client.threads.get_history(thread["thread_id"])
    assert history
    assert "64-bit" in json.dumps(history), history
    runtime = SimpleNamespace(state={"subagent_runs": {run["run_id"]: {
        "subagent_run_id": run["run_id"], "subagent_type_id": "test", "assistant_id": "test",
        "thread_id": thread["thread_id"], "run_id": run["run_id"], "status": "pending", "prompt": "Read page"}}},
        tool_call_id="wait")
    tool = _build_wait_tool(SimpleNamespace(get_async=lambda _: client))
    result = await tool.coroutine(runtime=runtime)
    report = result.update["subagent_runs"][run["run_id"]]["report"]
    assert report["status"] == "error", report
    print(json.dumps({"history_http": "OK", "history_entries": len(history),
                      "wait_report": report, "trainer_exception": False}))


asyncio.run(main())
