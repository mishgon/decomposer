"""Toolathlon gateway extension for benchmark-requested local tools.

The upstream decoupled gateway exposes MCP servers and ``claim_done`` only.
Our LangGraph executors do not instantiate Toolathlon's ``TaskAgent``, so they
otherwise lose native tools such as ``python_execute``, ``web_search``, and the
helpers for reading full outputs after Toolathlon's 100K truncation boundary.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

sys.path.insert(0, "/workspace")

from aiohttp import web
from agents import RunContextWrapper
from scripts.decoupled import container_tool_gateway as upstream
from utils.roles.task_agent import local_tool_mappings


EXPOSED_LOCAL_TOOLS = frozenset(
    {
        "handle_overlong_tool_outputs",
        "python_execute",
        "sleep",
        "web_search",
    }
)


class ContainerToolGateway(upstream.ContainerToolGateway):
    def __init__(
        self,
        bundle_file: str,
        *,
        include_local_tools: bool = False,
        debug: bool = False,
    ) -> None:
        super().__init__(bundle_file=bundle_file, debug=debug)
        self.include_local_tools = include_local_tools
        self._local_callbacks: dict[str, Any] = {}

    async def startup(self, app: web.Application) -> None:
        await super().startup(app)
        if not self.include_local_tools:
            return

        requested = set(self.bundle.get("needed_local_tools") or [])
        for config_name in sorted(requested.intersection(EXPOSED_LOCAL_TOOLS)):
            configured = local_tool_mappings[config_name]
            tools = configured if isinstance(configured, list) else [configured]
            for tool in tools:
                exposed_name = self.registry._allocate_name(
                    tool.name, "local", always_prefix=False
                )
                self.registry._records[exposed_name] = upstream.ToolRecord(
                    exposed_name=exposed_name,
                    backend_type="native_local",
                    backend_name=tool.name,
                    description=tool.description,
                    input_schema=tool.params_json_schema,
                    server_name=None,
                )
                self._local_callbacks[exposed_name] = tool.on_invoke_tool

    async def _remote_call(
        self, tool_record: upstream.ToolRecord, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        if tool_record.backend_type != "native_local":
            return await super()._remote_call(tool_record, arguments)

        callback = self._local_callbacks[tool_record.exposed_name]
        workspace = self.bundle["container_paths"]["agent_workspace"]
        result = await callback(
            RunContextWrapper(context={"_agent_workspace": workspace}),
            json.dumps(arguments, ensure_ascii=False),
        )
        text = result if isinstance(result, str) else json.dumps(
            result, ensure_ascii=False, default=str
        )
        return {
            "content": [{"type": "text", "text": text}],
            "isError": False,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle_file", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10086)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--include_local_tools", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gateway = ContainerToolGateway(
        bundle_file=args.bundle_file,
        include_local_tools=args.include_local_tools,
        debug=args.debug,
    )
    web.run_app(gateway.create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
