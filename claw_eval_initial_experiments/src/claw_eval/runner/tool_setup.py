"""Build task tools and tool dispatchers for agent loops."""

from __future__ import annotations

from typing import Any

from ..config import MediaConfig
from ..models.task import TaskDefinition
from ..models.tool import ToolSpec
from .agent_tools import build_agent_tools
from .dispatcher import ToolDispatcher


def build_task_tools_and_dispatcher(
    task: TaskDefinition,
    *,
    sandbox_tools: bool = False,
    sandbox_url: str | None = None,
    media_cfg: MediaConfig | None = None,
) -> tuple[list[ToolSpec], Any, list[ToolSpec] | None]:
    """Return (task_tools, dispatcher, sandbox_tool_list)."""
    _mcfg = media_cfg or MediaConfig()
    endpoint_map = task.get_endpoint_map()
    http_dispatcher = ToolDispatcher(endpoint_map)

    sandbox_tool_list = None
    if sandbox_tools:
        from .sandbox_dispatcher import SandboxToolDispatcher
        from .sandbox_tools import SANDBOX_TOOLS

        existing_names = {t.name for t in task.tools}
        sandbox_tool_list = [t for t in SANDBOX_TOOLS if t.name not in existing_names]
        task_tools = list(task.tools) + sandbox_tool_list
        dispatcher = SandboxToolDispatcher(
            http_dispatcher,
            sandbox_url=sandbox_url,
            max_images_per_turn=_mcfg.max_images_per_turn,
            tool_image_max_dimension=_mcfg.tool_image_max_dimension,
            tool_image_quality=_mcfg.tool_image_quality,
        )
    else:
        task_tools = list(task.tools)
        dispatcher = http_dispatcher

    agent_tool_list = build_agent_tools(
        enable_todo=task.environment.enable_todo,
        enable_compact=task.environment.enable_compact,
    )
    task_tools = task_tools + agent_tool_list
    return task_tools, dispatcher, sandbox_tool_list
