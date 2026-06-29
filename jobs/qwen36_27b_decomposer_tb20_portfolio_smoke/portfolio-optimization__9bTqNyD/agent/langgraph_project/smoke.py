from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from agent import BaselineSidecar, DEFAULT_BASELINE_PROJECT, _build_model
from decomposer.core import HISTORY_LIMIT, create_decomposer_agent
from langchain_core.messages import message_to_dict
from langgraph_sdk import get_sync_client


def _json_default(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def _serialize_state(state: dict):
    serialized = dict(state)
    serialized["messages"] = [
        message_to_dict(message) for message in serialized.get("messages", [])
    ]
    return serialized


def _capture_subagent_messages(sidecar: BaselineSidecar, subagent_runs: dict):
    client = get_sync_client(
        url=sidecar.url,
        headers={"x-auth-scheme": "langsmith"},
    )
    captured = {}
    for subagent_run_id, subagent_run in subagent_runs.items():
        history = client.threads.get_history(
            thread_id=subagent_run["thread_id"],
            limit=HISTORY_LIMIT,
            metadata={"run_id": subagent_run["run_id"]},
        )
        if not history:
            captured[subagent_run_id] = {"history": [], "run_messages": []}
            continue

        before_messages = history[-1]["values"].get("messages", [])
        after_messages = history[0]["values"].get("messages", [])
        run_messages = after_messages[len(before_messages):]
        captured[subagent_run_id] = {
            "thread_id": subagent_run["thread_id"],
            "run_id": subagent_run["run_id"],
            "history_len": len(history),
            "before_message_count": len(before_messages),
            "after_message_count": len(after_messages),
            "run_messages": run_messages,
            "history": history,
        }
    return captured


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="decomposer-smoke-"))
    os.environ["T_BENCH_WORKDIR"] = str(workdir)

    prompt = (
        "Use the available baseline subagent to create a file named "
        "`decomposer_smoke.txt` in the task workspace. The file must contain "
        "exactly `decomposer sidecar ok` followed by a newline. Spawn exactly "
        "one subagent, wait for its report, and then answer briefly whether "
        "the file was created."
    )

    sidecar = BaselineSidecar(DEFAULT_BASELINE_PROJECT).start()
    try:
        agent = create_decomposer_agent(
            decomposer_model=_build_model({
                "configurable": {
                    "model": os.environ.get("OPENAI_MODEL", "Qwen/Qwen3.6-27B")
                }
            }),
            subagent_types=[
                {
                    "subagent_type_id": "baseline",
                    "assistant_id": sidecar.assistant_id or "",
                    "url": sidecar.url,
                    "description": (
                        "A Terminal-Bench baseline agent with a bash tool. It can "
                        "inspect and modify files in the task workspace."
                    ),
                }
            ],
        )
        result = agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config={"recursion_limit": 50},
        )
        state = _serialize_state(result)
        subagent_messages = _capture_subagent_messages(
            sidecar,
            state.get("subagent_runs", {}),
        )
    finally:
        sidecar.stop()

    state_path = workdir / "decomposer_state.json"
    messages_path = workdir / "messages.json"
    subagent_runs_path = workdir / "subagent_runs.json"
    subagent_messages_path = workdir / "subagent_messages.json"
    state_path.write_text(json.dumps(state, indent=2, default=_json_default))
    messages_path.write_text(json.dumps(state.get("messages", []), indent=2, default=_json_default))
    subagent_runs_path.write_text(json.dumps(state.get("subagent_runs", {}), indent=2, default=_json_default))
    subagent_messages_path.write_text(json.dumps(subagent_messages, indent=2, default=_json_default))

    target = workdir / "decomposer_smoke.txt"
    content = target.read_text() if target.exists() else None
    print("workdir:", workdir)
    print("state_path:", state_path)
    print("messages_path:", messages_path)
    print("subagent_runs_path:", subagent_runs_path)
    print("subagent_messages_path:", subagent_messages_path)
    print("sidecar_log_path:", sidecar.log_path)
    print("file_exists:", target.exists())
    print("file_content:", repr(content))
    print("subagent_runs:", json.dumps(state.get("subagent_runs", {}), indent=2, default=_json_default))
    print("subagent_messages_summary:")
    for subagent_run_id, item in subagent_messages.items():
        print("  run:", subagent_run_id)
        for index, message in enumerate(item.get("run_messages", [])):
            print(
                f"    {index}: type={message.get('type')} "
                f"tool_calls={len(message.get('tool_calls') or [])} "
                f"name={message.get('name')}"
            )
    print("final_message:", result["messages"][-1].content)
    if content != "decomposer sidecar ok\n":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
