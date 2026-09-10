"""Check real MCP fixture access and two-episode isolation without using a GPU."""

import argparse
import json
import uuid
from pathlib import Path

from gyms.toolathlon_gym.episode import Episode


MCP_PROBE = r'''
import asyncio, json, os, sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    env = dict(os.environ)
    for source, target in {"PGHOST":"PG_HOST", "PGPORT":"PG_PORT",
                           "PGDATABASE":"PG_DATABASE", "PGUSER":"PG_USER",
                           "PGPASSWORD":"PG_PASSWORD"}.items():
        env[target] = env[source]
    env.update(WORDPRESS_SITE_URL="http://localhost:8081",
               WOOCOMMERCE_CONSUMER_KEY="placeholder", WOOCOMMERCE_CONSUMER_SECRET="placeholder",
               CANVAS_API_TOKEN="placeholder", CANVAS_DOMAIN="localhost:8080")
    async with stdio_client(StdioServerParameters(command="node", args=[sys.argv[1]], env=env)) as io:
        async with ClientSession(*io) as client:
            await client.initialize()
            result = await client.call_tool(sys.argv[2], json.loads(sys.argv[3]))
            if result.isError:
                raise RuntimeError(str(result))
            print(json.dumps(result.model_dump()))
asyncio.run(main())
'''


def probe(episode, server, entry, tool, arguments=None):
    result = episode.command("exec", episode.container, "/opt/subagents/bin/python", "-c",
                             MCP_PROBE, f"/opt/local_servers/{server}/{entry}", tool,
                             json.dumps(arguments or {}))
    response = json.loads(result.stdout.strip().splitlines()[-1])
    text = json.dumps(response)
    if any(error in text.lower() for error in ("econnrefused", "cloudflare", "connection refused")):
        raise RuntimeError(text)
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    episodes = [Episode("wc-refund-analysis-notion", args.output / str(i), subagent_port=8025)
                for i in range(2)]
    started = []
    try:
        for episode in episodes:
            episode.start()
            started.append(episode)
        marker = "rl-isolation-" + uuid.uuid4().hex
        episodes[0].command("exec", episodes[0].pg, "psql", "-U", "eigent", "-d", "toolathlon_gym", "-c",
                            f"UPDATE wc.orders SET customer_note = '{marker}';")
        first = probe(episodes[0], "woocommerce-mcp", "dist/index.js", "woo_orders_list")
        second = probe(episodes[1], "woocommerce-mcp", "dist/index.js", "woo_orders_list")
        if marker not in first or marker in second:
            raise RuntimeError("WooCommerce fixture access or episode isolation failed")
        notion = probe(episodes[0], "notion-mcp-server", "bin/cli.mjs", "API-post-search")
        body = json.loads(json.loads(notion)["content"][0]["text"])
        if body.get("object") != "list" or not isinstance(body.get("results"), list):
            raise RuntimeError("Notion did not return a valid search result")
        episodes[0].command("exec", episodes[0].pg, "psql", "-U", "eigent", "-d", "toolathlon_gym", "-c",
                            f"UPDATE canvas.users SET name = '{marker}' WHERE id = (SELECT min(id) FROM canvas.users);")
        canvas = [probe(episode, "mcp-canvas-lms", "build/index.js", "canvas_list_account_users",
                        {"account_id": 1}) for episode in episodes]
        if marker not in canvas[0] or marker in canvas[1]:
            raise RuntimeError("Canvas account users bypass local fixtures or are not isolated")
        report = {"woocommerce_reads_local_database": True, "duplicate_task_isolated": True,
                  "canvas_account_users_reads_local_database": True,
                  "notion_response": json.loads(notion)}
        (args.output / "result.json").write_text(json.dumps(report, indent=2))
        print("PASS: WooCommerce and Canvas read isolated local fixtures; Notion MCP responded.")
    finally:
        for episode in started:
            episode.close()


if __name__ == "__main__":
    main()
