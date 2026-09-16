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
        "Do not use any other tools. Report both results."}]}, config={"recursion_limit": 12})
    messages = result.get("messages", [])
    outputs = [str(m.get("content", "")) for m in messages
               if m.get("type") == "tool" and m.get("name") == "python_execute"]
    assert any("1243193" in s and "Return code: 0" in s for s in outputs), result
    assert any("TIMEOUT" in s for s in outputs), result
    return {"python_tool_results": outputs, "passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    episode = Episode("ppt-snowflake-executive", args.output,
                      subagent_url=os.environ["SUBAGENT_URL"],
                      subagent_host=os.environ.get("SUBAGENT_HOST"))
    try:
        episode.start()
        result = asyncio.run(asyncio.wait_for(check(episode), 180))
        (args.output / "integration.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result))
    finally:
        episode.close()


if __name__ == "__main__":
    main()
