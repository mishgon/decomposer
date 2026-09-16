"""Exercise Python via the real LangGraph server and hosted subagent model."""
import argparse
import asyncio
import json
import os
from pathlib import Path

from langgraph_sdk import get_client
from gyms.toolathlon_gym.episode import Episode


async def check(episode):
    client = get_client(url=episode.url)
    result = await client.runs.wait(None, "configured_non_thinking", input={"messages": [{
        "role": "user", "content": "Use python_execute to run print(73129 * 17). "
        "Then use python_execute with timeout=1 to run import time; time.sleep(10). "
        "Then call save_overlong_output with content='overlong-roundtrip-73129' and label='probe'. "
        "Use its returned ID to call view_overlong_output and read back the saved text. "
        "Use only these three tools. Report the results."}]}, config={"recursion_limit": 16})
    messages = result.get("messages", [])
    outputs = [str(m.get("content", "")) for m in messages
               if m.get("type") == "tool" and m.get("name") == "python_execute"]
    assert any("1243193" in s and "Return code: 0" in s for s in outputs), result
    assert any("TIMEOUT" in s for s in outputs), result
    saved = [str(m.get("content", "")) for m in messages
             if m.get("type") == "tool" and m.get("name") == "view_overlong_output"]
    assert any("overlong-roundtrip-73129" in s for s in saved), result
    return {"python_tool_results": outputs, "overlong_results": saved, "passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--webapp", type=Path, help="Test a staged webapp without changing live workers")
    args = parser.parse_args()
    episode = Episode("ppt-snowflake-executive", args.output,
                      subagent_url=os.environ["SUBAGENT_URL"],
                      subagent_host=os.environ.get("SUBAGENT_HOST"))
    if args.webapp:
        start = episode.start_task_container
        def staged_start(*arguments):
            return start(*(f"{args.webapp.resolve()}:/opt/decomposer/gyms/toolathlon_gym/subagents/webapp.py:ro"
                           if str(arg).endswith("/subagents/webapp.py:ro") else arg
                           for arg in arguments))
        episode.start_task_container = staged_start
    try:
        episode.start()
        result = asyncio.run(asyncio.wait_for(check(episode), 180))
        (args.output / "integration.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result))
    finally:
        episode.close()


if __name__ == "__main__":
    main()
