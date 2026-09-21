"""Replay the malformed Notion call that killed RL; no model or GPU needed."""

import argparse
import json
from pathlib import Path

from gyms.toolathlon_gym.episode import Episode
from tests.toolathlon_gym.check_environment import MCP_PROBE


# Exact properties emitted in smoke-qwen38-n8-10epochs-20260921-v3.
MALFORMED = '{"title": [{"text": {"content": "Employee Tenure Dashboard"}, "type": "text"}], "type": "title"}}'
VALID = {"title": {"title": [{"text": {"content": "Employee Tenure Dashboard"}}]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    episode = Episode("sf-hr-experience-notion", args.output, image=args.image)
    episode.start()
    try:
        def call(tool, arguments):
            return episode.command(
                "exec", episode.container, "/opt/subagents/bin/python", "-c", MCP_PROBE,
                "/opt/local_servers/notion-mcp-server/bin/cli.mjs", tool,
                json.dumps(arguments), check=False)

        def query(sql):
            return episode.command("exec", episode.pg, "psql", "-U", "eigent", "-d",
                                   "toolathlon_gym", "-tAc", sql).stdout.strip()

        baseline = query("SELECT count(*) FROM notion.pages")
        for bad in (MALFORMED, "[]", "null", "42"):
            result = call("API-post-page", {"parent": {"page_id": "user-pm-168aae21"},
                                            "properties": bad})
            assert result.returncode != 0 and "properties must" in result.stderr, result
            assert query("SELECT count(*) FROM notion.pages") == baseline

        # The agent can correct its call; serialized valid objects stay objects.
        result = call("API-post-page", {"parent": {"page_id": "user-pm-168aae21"},
                                        "properties": json.dumps(VALID)})
        assert result.returncode == 0, result.stderr
        response = json.loads(result.stdout.strip().splitlines()[-1])
        page = json.loads(response["content"][0]["text"])
        assert page["properties"] == VALID, page
        page_id = page["id"]
        result = call("API-patch-page", {"page_id": page_id, "properties": MALFORMED})
        assert result.returncode != 0 and "properties must" in result.stderr, result
        assert json.loads(query(f"SELECT properties FROM notion.pages WHERE id = '{page_id}'")) == VALID
        result = call("API-patch-page", {"page_id": page_id, "properties": VALID})
        assert result.returncode == 0, result.stderr
        assert query(f"SELECT jsonb_typeof(properties) FROM notion.pages WHERE id = '{page_id}'") == "object"
        evaluation = episode.score()
        assert isinstance(evaluation["reward"], (int, float))
        print("PASS: malformed creates/updates rejected without writes; valid retry accepted; native evaluator scored.")
    finally:
        episode.close()


if __name__ == "__main__":
    main()
