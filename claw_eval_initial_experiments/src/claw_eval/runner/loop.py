"""Core agent execution loop: Think -> Act -> Observe -> Repeat."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from .user_agent import UserAgent

from ..config import MediaConfig, ModelConfig, PromptConfig, ReActRunConfig
from ..models.content import TextBlock
from ..models.message import Message
from ..models.task import TaskDefinition
from ..models.trace import AuditSnapshot, TraceEnd, TraceMessage, TraceStart, TokenUsage
from ..trace.writer import TraceWriter
from .agent_tools import build_agent_tools
from .loop_common import build_initial_user_content, log
from .providers.openai_compat import OpenAICompatProvider
from .react_loop import run_react_loop
from .system_prompt import build_system_prompt
from .todo import TodoManager
from .tool_setup import build_task_tools_and_dispatcher

# Re-export for backward compatibility
from .loop_common import brief as _brief  # noqa: F401
from .loop_common import build_initial_user_content as _build_initial_user_content  # noqa: F401
from .loop_common import log as _log  # noqa: F401


def run_task(
    task: TaskDefinition,
    provider: OpenAICompatProvider,
    trace_dir: str | Path = "traces",
    *,
    sandbox_tools: bool = False,
    sandbox_url: str | None = None,
    prompt_cfg: PromptConfig | None = None,
    model_cfg: ModelConfig | None = None,
    media_cfg: MediaConfig | None = None,
    user_agent: UserAgent | None = None,
    react_cfg: ReActRunConfig | None = None,
) -> Path:
    """Execute one trial of a task and write JSONL trace."""
    trace_id = str(uuid4())
    trace_path = Path(trace_dir) / f"{task.task_id}_{trace_id[:8]}.jsonl"

    task_tools, dispatcher, sandbox_tool_list = build_task_tools_and_dispatcher(
        task,
        sandbox_tools=sandbox_tools,
        sandbox_url=sandbox_url,
        media_cfg=media_cfg,
    )

    agent_tool_list = build_agent_tools(
        enable_todo=task.environment.enable_todo,
        enable_compact=task.environment.enable_compact,
    )
    todo_mgr = TodoManager() if task.environment.enable_todo else None

    wall_start = time.monotonic()

    log(f"[start] task={task.task_id} model={provider.model_id} trace={trace_path.name}")
    log(f"[config] max_turns={task.environment.max_turns} timeout={task.environment.timeout_seconds}s sandbox_tools={sandbox_tools}")
    if agent_tool_list:
        log(f"[agent tools] {', '.join(t.name for t in agent_tool_list)}")

    with TraceWriter(trace_path) as writer:
        writer.write_event(TraceStart(
            trace_id=trace_id,
            task_id=task.task_id,
            model=provider.model_id,
            run_mode="flat",
        ))

        system_prompt = build_system_prompt(task, prompt_cfg, extra_tools=sandbox_tool_list)
        if model_cfg and model_cfg.system_prompt_prefix:
            system_prompt = model_cfg.system_prompt_prefix + "\n\n" + system_prompt
        ua_cfg = task.user_agent
        ua_enabled = ua_cfg.enabled and user_agent is not None
        if ua_enabled and ua_cfg.system_prompt_suffix:
            system_prompt = system_prompt + "\n\n" + ua_cfg.system_prompt_suffix

        user_content = build_initial_user_content(
            task,
            trace_id=trace_id,
            writer=writer,
            model_cfg=model_cfg,
            media_cfg=media_cfg,
        )
        initial_messages: list[Message] = [
            Message(role="system", content=[TextBlock(text=system_prompt)]),
            Message(role="user", content=user_content),
        ]
        writer.write_event(TraceMessage(trace_id=trace_id, message=initial_messages[-1]))

        loop_error: str | None = None
        loop_exc: Exception | None = None
        loop_result = None
        try:
            loop_result = run_react_loop(
                initial_messages=initial_messages,
                task=task,
                provider=provider,
                task_tools=task_tools,
                dispatcher=dispatcher,
                trace_id=trace_id,
                writer=writer,
                model_cfg=model_cfg,
                media_cfg=media_cfg,
                user_agent=user_agent,
                todo_mgr=todo_mgr,
                wall_start=wall_start,
                max_turns=react_cfg.max_turns if react_cfg else None,
                max_environment_tool_calls=(
                    react_cfg.max_environment_tool_calls if react_cfg else None
                ),
                react_cfg=react_cfg,
            )
        except Exception as exc:
            loop_error = f"{type(exc).__name__}: {exc}"
            loop_exc = exc
            log(f"[error] agent loop failed: {loop_error}")

        import httpx as _httpx

        for svc in task.services:
            if svc.reset_endpoint:
                audit_url = svc.reset_endpoint.rsplit("/reset", 1)[0] + "/audit"
                try:
                    resp = _httpx.get(audit_url, timeout=5)
                    writer.write_event(AuditSnapshot(
                        trace_id=trace_id,
                        service_name=svc.name,
                        audit_url=audit_url,
                        audit_data=resp.json(),
                    ))
                except Exception:
                    pass

        wall_time = time.monotonic() - wall_start
        if loop_result:
            total_usage = loop_result.usage
            turn_count = loop_result.turn_count
            model_time_s = loop_result.model_time_s
            tool_time_s = loop_result.tool_time_s
            user_agent_rounds = loop_result.user_agent_rounds
            ua_done = loop_result.user_agent_done
        else:
            total_usage = TokenUsage()
            turn_count = 0
            model_time_s = 0.0
            tool_time_s = 0.0
            user_agent_rounds = 0
            ua_done = False

        input_tok = total_usage.input_tokens
        output_tok = total_usage.output_tokens
        total_tok = input_tok + output_tok
        other_time_s = max(0.0, wall_time - model_time_s - tool_time_s)
        failure_modes = [loop_error] if loop_error else []
        ua_max_rounds = ua_cfg.max_rounds if ua_enabled else 0

        writer.write_event(TraceEnd(
            trace_id=trace_id,
            total_turns=turn_count,
            model_input_tokens=input_tok,
            model_output_tokens=output_tok,
            input_tokens=input_tok,
            output_tokens=output_tok,
            total_tokens=total_tok,
            model_time_s=round(model_time_s, 2),
            tool_time_s=round(tool_time_s, 2),
            other_time_s=round(other_time_s, 2),
            wall_time_s=round(wall_time, 2),
            failure_modes=failure_modes,
            user_agent_rounds=user_agent_rounds,
            user_agent_max_rounds=ua_max_rounds,
            user_agent_done=ua_done,
        ))

        if loop_error:
            raise loop_exc

    log(
        f"[end] turns={turn_count} tokens={total_tok} "
        f"({input_tok}in/{output_tok}out) "
        f"time=model {model_time_s:.1f}s tool {tool_time_s:.1f}s wall {wall_time:.1f}s"
    )

    dispatcher.close()
    return trace_path
