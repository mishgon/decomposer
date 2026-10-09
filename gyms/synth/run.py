"""One synthetic episode, reusable by baseline evaluation and OPD."""
import asyncio
from datetime import datetime, timezone
import json

from langchain_core.messages import message_to_dict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph_sdk import get_client
from decomposer.core import create_decomposer_agent
from decomposer.usage import build_usage_summary
from decomposer.visualization import write_trace_html
from gyms.synth.executor import initialize, grade
from gyms.synth.worker import Context


async def episode(task, policy, directory, worker_url, *, timeout=180, recursion_limit=80):
    initialize(directory, task)
    (directory / "task.json").write_text(json.dumps(task, indent=2))
    agent = create_decomposer_agent(decomposer_model=policy,
        agent_types=[{"agent_type_id": "file_worker", "assistant_id": "file_worker",
                      "description": "Deterministic executor. Accepts only newline-separated COPY and CHECK commands as specified in the task.",
                      "url": worker_url}],
        context_schema=Context, checkpointer=InMemorySaver(), agent_recursion_limit=8)
    config = {"configurable": {"thread_id": directory.name}, "recursion_limit": recursion_limit}
    started = datetime.now(timezone.utc).isoformat()
    state, error = {}, None
    try:
        state = await asyncio.wait_for(agent.ainvoke(
            {"messages": [{"role": "user", "content": task["prompt"]}]}, config=config,
            context=Context(str(directory.resolve()))), timeout)
    except Exception as exc:
        error = exc
    finally:
        snapshot = await agent.aget_state(config)
        state = state or dict(snapshot.values)
        cleanup_errors = []
        # This server is shared; clean only this episode's threads.
        async with get_client(url=worker_url, timeout=30) as client:
            for worker in state.get("agents", {}).values():
                thread_id = worker["thread_id"]
                try:
                    for status in ("pending", "running"):
                        while runs := await client.runs.list(thread_id, status=status, limit=100):
                            for run in runs:
                                await client.runs.cancel(thread_id, run["run_id"], wait=True, action="interrupt")
                    checkpoint = await client.threads.get_state(thread_id)
                    (directory / f"worker-{thread_id}.json").write_text(json.dumps(checkpoint, default=str))
                    await client.threads.delete(thread_id)
                except Exception as exc:
                    cleanup_errors.append(repr(exc))
        trace = {**state, "task": task["task_id"], "episode_id": directory.name,
                 "harness": "decomposer", "started_at": started,
                 "finished_at": datetime.now(timezone.utc).isoformat(),
                 "messages": [message_to_dict(m) for m in state.get("messages", [])],
                 "agent_error": repr(error) if error else None, "cleanup_errors": cleanup_errors}
        (directory / "trace.json").write_text(json.dumps(trace, indent=2, default=str))
        write_trace_html(trace, directory / "trace.html")
    workspace = json.loads((directory / "workspace.json").read_text())
    result = {"task_id": task["task_id"], "split": task["split"],
              "status": "finished" if error is None and not cleanup_errors else "error",
              "error": repr(error) if error else None, "cleanup_errors": cleanup_errors,
              **grade(task, workspace, state.get("agent_runs", {}))}
    result["passed"] = result["status"] == "finished" and result["score"] == 1
    (directory / "result.json").write_text(json.dumps(result, indent=2))
    (directory / "usage.json").write_text(json.dumps(build_usage_summary(
        trace["messages"], state.get("agent_runs", {}), state.get("agents", {})), indent=2))
    return result


if __name__ == "__main__":
    import argparse
    import os
    from pathlib import Path
    import socket
    from uuid import uuid4
    from decomposer.agent_server import agent_server
    from decomposer.models import create_model
    from gyms.synth import ROOT
    from gyms.synth.task import make_task

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "eval"], default="eval")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--model", default="vllm/qwen_3_5_4b_non_thinking")
    parser.add_argument("--worker-python", type=Path, default=ROOT / ".venv-workers/bin/python")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/gyms/synth" / uuid4().hex[:8])
    args = parser.parse_args()

    async def main():
        args.output = args.output.resolve()
        os.environ["SYNTH_ARTIFACT_ROOT"] = str(args.output.parent)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        policy = create_model(args.model)
        try:
            async with agent_server(ROOT / "gyms/synth/langgraph.json", port=port,
                                    python_executable=args.worker_python) as url:
                print(json.dumps(await episode(make_task(args.split, args.index), policy, args.output, url), indent=2))
        finally:
            policy.http_client.close()
            await policy.http_async_client.aclose()

    asyncio.run(main())
