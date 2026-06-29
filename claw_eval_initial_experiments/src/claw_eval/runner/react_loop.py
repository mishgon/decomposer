"""Shared ReAct loop used by flat baseline and executor subtasks."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .user_agent import UserAgent

from ..config import MediaConfig, ModelConfig, ReActRunConfig
from ..models.content import ContentBlock, TextBlock, ToolResultBlock
from ..models.message import Message
from ..models.task import TaskDefinition
from ..models.tool import ToolSpec
from ..models.trace import CompactEvent, TokenUsage, TraceMessage
from ..trace.writer import TraceWriter
from .compact import (
    _estimate_tokens,
    do_auto_compact,
    micro_compact,
    should_auto_compact,
)
from .loop_common import (
    brief,
    cap_conversation_images,
    log,
    make_local_tool_result,
    strip_old_turn_images,
)
from .executor_report_tools import SUBMIT_EXECUTOR_REPORT_NAME, format_executor_report
from .providers.openai_compat import OpenAICompatProvider
from .todo import TodoManager


@dataclass
class ReActLoopResult:
    messages: list[Message]
    turn_count: int
    usage: TokenUsage = field(default_factory=TokenUsage)
    model_time_s: float = 0.0
    tool_time_s: float = 0.0
    stopped_reason: str = "no_tools"
    error: str | None = None
    user_agent_rounds: int = 0
    user_agent_done: bool = False
    environment_tool_count: int = 0
    environment_tool_names: list[str] = field(default_factory=list)
    environment_tool_summaries: list[str] = field(default_factory=list)
    tool_budget_exhausted: bool = False
    transitional_tool_retry_count: int = 0
    empty_visible_response_count: int = 0
    assistant_text_message_count: int = 0
    assistant_tool_message_count: int = 0
    submitted_report: str = ""
    submitted_report_count: int = 0

    @property
    def final_report(self) -> str:
        """Last assistant text (used as executor subtask report)."""
        for msg in reversed(self.messages):
            if msg.role == "assistant" and msg.text.strip():
                return msg.text.strip()
        return ""


def _looks_like_transitional_tool_text(text: str, phrases: list[str]) -> bool:
    """Return True when text says the model is about to use tools but did not."""
    normalized = " ".join(text.lower().split())
    if not normalized:
        return False

    normalized_phrases = [" ".join(p.lower().split()) for p in phrases if p.strip()]
    if not normalized_phrases:
        return False

    # Check each sentence-ish segment so messages like
    # "I found records. Let me retrieve details." still trigger.
    for separator in ("\n", ".", "!", "?", ";"):
        normalized = normalized.replace(separator, "|")
    for segment in normalized.split("|"):
        candidate = segment.strip()
        if any(candidate.startswith(phrase) for phrase in normalized_phrases):
            return True
    return False


def _tool_result_summary(tool_name: str, result: ToolResultBlock, max_chars: int = 500) -> str:
    text = " ".join(block.text for block in result.content if block.type == "text").strip()
    text = " ".join(text.split()) or "(no text result)"
    if len(text) > max_chars:
        text = text[:max_chars - 3] + "..."
    status = "ERR" if result.is_error else "OK"
    return f"{tool_name} [{status}]: {text}"


def run_react_loop(
    *,
    initial_messages: list[Message],
    task: TaskDefinition,
    provider: OpenAICompatProvider,
    task_tools: list[ToolSpec],
    dispatcher,
    trace_id: str,
    writer: TraceWriter | None = None,
    dispatch_writer: TraceWriter | None = None,
    log_messages: bool = True,
    log_tool_dispatches: bool = True,
    max_turns: int | None = None,
    timeout_seconds: int | None = None,
    max_output_tokens: int | None = None,
    wall_start: float | None = None,
    model_cfg: ModelConfig | None = None,
    media_cfg: MediaConfig | None = None,
    user_agent: UserAgent | None = None,
    todo_mgr: TodoManager | None = None,
    auto_compact_count: int = 0,
    min_environment_tool_calls: int = 0,
    max_environment_tool_calls: int | None = None,
    react_cfg: ReActRunConfig | None = None,
    enable_submit_report_tool: bool = False,
) -> ReActLoopResult:
    """Run a ReAct tool-use loop until stop, timeout, or max turns.

    Args:
        writer: Trace writer for messages and compact events.
        dispatch_writer: Separate writer for tool_dispatch events (defaults to writer).
        log_messages: When False, skip TraceMessage and CompactEvent writes.
        log_tool_dispatches: When False, skip ToolDispatch writes.
    """
    _mcfg = media_cfg or MediaConfig()
    _react_cfg = react_cfg or ReActRunConfig()
    _dispatch_writer = dispatch_writer or writer
    turn_limit = max_turns if max_turns is not None else task.environment.max_turns
    timeout = timeout_seconds if timeout_seconds is not None else task.environment.timeout_seconds
    _wall_start = wall_start if wall_start is not None else time.monotonic()
    context_window = model_cfg.context_window if model_cfg else 200_000

    messages = list(initial_messages)
    total_usage = TokenUsage()
    turn_count = 0
    model_time_s = 0.0
    tool_time_s = 0.0
    dispatched_tool_count = 0
    environment_tool_names: list[str] = []
    environment_tool_summaries: list[str] = []
    empty_response_retry_used = False
    missing_required_tool_retry_used = False
    transitional_tool_retries = 0
    empty_visible_response_count = 0
    assistant_text_message_count = 0
    assistant_tool_message_count = 0
    submitted_report = ""
    submitted_report_count = 0
    force_required_tool_next_turn = False
    tool_budget_exhausted = False
    stopped_reason = "max_turns"
    loop_error: str | None = None
    loop_exc: Exception | None = None

    ua_cfg = task.user_agent
    ua_enabled = ua_cfg.enabled and user_agent is not None
    ua_max_rounds = ua_cfg.max_rounds if ua_enabled else 0
    user_agent_rounds = 0
    ua_done = False

    if todo_mgr is None and task.environment.enable_todo:
        todo_mgr = TodoManager()

    try:
        while turn_count < turn_limit:
            elapsed = time.monotonic() - _wall_start
            if elapsed > timeout:
                log(f"[timeout] {elapsed:.1f}s exceeded limit {timeout}s")
                stopped_reason = "timeout"
                break

            if task.environment.enable_compact:
                micro_compact(
                    messages,
                    keep_recent=task.environment.compact_keep_recent,
                    min_chars=task.environment.compact_min_chars,
                )

            if (
                task.environment.enable_compact
                and auto_compact_count < task.environment.compact_max_auto_compacts
                and should_auto_compact(messages, context_window, task.environment.compact_threshold_pct)
            ):
                tokens_before = _estimate_tokens(messages)
                msgs_before = len(messages)
                log(f"[auto-compact] triggering (est. {tokens_before} tokens, {msgs_before} msgs)")
                messages = do_auto_compact(
                    messages,
                    provider,
                    keep_recent_on_summary=task.environment.compact_keep_recent_on_summary,
                    protect_tokens=task.environment.compact_protect_tokens,
                    todo_mgr=todo_mgr,
                )
                auto_compact_count += 1
                tokens_after = _estimate_tokens(messages)
                if writer and log_messages:
                    writer.write_event(CompactEvent(
                        trace_id=trace_id,
                        layer="auto",
                        estimated_tokens_before=tokens_before,
                        estimated_tokens_after=tokens_after,
                        messages_before=msgs_before,
                        messages_after=len(messages),
                    ))
                log(f"[auto-compact] done: {tokens_before} → {tokens_after} tokens, {msgs_before} → {len(messages)} msgs")

            n_old = strip_old_turn_images(messages, _mcfg.image_keep_recent_turns)
            if n_old > 0:
                log(f"  [image-strip] stripped {n_old} image(s) from old turns, keeping last {_mcfg.image_keep_recent_turns} turns")

            n_dropped = cap_conversation_images(messages, _mcfg.max_conversation_images)
            if n_dropped > 0:
                log(f"  [image-cap] dropped {n_dropped} oldest image(s), keeping last {_mcfg.max_conversation_images}")

            log(f"[turn {turn_count + 1}/{turn_limit}] calling model ...")
            model_t0 = time.monotonic()
            require_tool = bool(task_tools) and (
                force_required_tool_next_turn
                or (
                    min_environment_tool_calls > 0
                    and dispatched_tool_count < min_environment_tool_calls
                )
            )
            force_required_tool_next_turn = False
            response, usage = provider.chat(
                messages,
                tools=task_tools,
                tool_choice="required" if require_tool else None,
                max_tokens=max_output_tokens,
            )
            model_time_s += time.monotonic() - model_t0
            total_usage.input_tokens += usage.input_tokens
            total_usage.output_tokens += usage.output_tokens
            turn_count += 1

            if writer and log_messages:
                writer.write_event(TraceMessage(
                    trace_id=trace_id,
                    message=response,
                    usage=usage,
                ))

            messages.append(response)

            text_blocks = [b for b in response.content if b.type == "text"]
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            visible_text = response.text.strip()
            if visible_text:
                assistant_text_message_count += 1
            if tool_uses:
                assistant_tool_message_count += 1
            if not visible_text and not tool_uses and usage.output_tokens > 0:
                empty_visible_response_count += 1
            text_preview = text_blocks[0].text[:120].replace("\n", " ") if text_blocks else ""
            log(f"[turn {turn_count}] assistant: {len(text_blocks)} text, {len(tool_uses)} tool_use | tokens: +{usage.input_tokens}in +{usage.output_tokens}out")
            if text_preview:
                log(f"  text: {text_preview}{'...' if len(text_blocks[0].text) > 120 else ''}")

            if not tool_uses:
                empty_model_response = (
                    not visible_text
                    and usage.output_tokens > 0
                )
                if (
                    empty_model_response
                    and task_tools
                    and _react_cfg.retry_empty_model_response
                    and not empty_response_retry_used
                ):
                    empty_response_retry_used = True
                    if dispatched_tool_count == 0:
                        correction_text = (
                            "Protocol error: the previous assistant turn produced tokens but no text "
                            "and no native tool call. Call exactly one available native tool now. "
                            "Do not answer in plain text."
                        )
                    else:
                        correction_text = (
                            "Protocol error: the previous assistant turn produced tokens but no text "
                            "and no native tool call after receiving a tool result. Either call another "
                            "available native tool if more information is needed, or provide a compact "
                            "text report. Do not return an empty message."
                        )
                    correction_msg = Message(role="user", content=[TextBlock(text=correction_text)])
                    messages.append(correction_msg)
                    if writer and log_messages:
                        writer.write_event(TraceMessage(trace_id=trace_id, message=correction_msg))
                    log("[protocol] empty model response; retrying once")
                    continue
                if (
                    task_tools
                    and dispatched_tool_count < min_environment_tool_calls
                    and _react_cfg.retry_missing_required_tool
                    and not missing_required_tool_retry_used
                ):
                    missing_required_tool_retry_used = True
                    correction_msg = Message(role="user", content=[TextBlock(text=(
                        "Protocol error: the previous assistant turn did not call a native tool. "
                        f"Use the available tools now; at least {min_environment_tool_calls} "
                        "environment tool call(s) are required before reporting completion."
                    ))])
                    messages.append(correction_msg)
                    if writer and log_messages:
                        writer.write_event(TraceMessage(trace_id=trace_id, message=correction_msg))
                    log("[protocol] missing required tool call; retrying once")
                    continue

                if task_tools and dispatched_tool_count < min_environment_tool_calls:
                    stopped_reason = "missing_required_tool"
                    log("[done] missing required tool call")
                    break

                retry_limit = max(0, _react_cfg.transitional_tool_retry_limit)
                if (
                    task_tools
                    and dispatched_tool_count > 0
                    and _react_cfg.retry_transitional_tool_text
                    and _looks_like_transitional_tool_text(
                        response.text,
                        _react_cfg.transitional_tool_phrases,
                    )
                ):
                    if transitional_tool_retries < retry_limit:
                        transitional_tool_retries += 1
                        force_required_tool_next_turn = True
                        correction_msg = Message(role="user", content=[TextBlock(text=(
                            "Protocol correction: your previous assistant turn said you would "
                            "use or inspect tool results, but it did not include a native tool "
                            "call. Call the appropriate available native tool now. Do not "
                            "describe the next step in plain text."
                        ))])
                        messages.append(correction_msg)
                        if writer and log_messages:
                            writer.write_event(TraceMessage(trace_id=trace_id, message=correction_msg))
                        log(
                            "[protocol] transitional text without tool call; "
                            f"retrying with required tool ({transitional_tool_retries}/{retry_limit})"
                        )
                        continue

                    stopped_reason = "transitional_tool_retry_exhausted"
                    log("[done] transitional tool-call retry budget exhausted")
                    break

                if ua_enabled and user_agent_rounds < ua_max_rounds:
                    ua_text = user_agent.generate_response(
                        persona=ua_cfg.persona,
                        conversation_messages=messages,
                    )
                    if ua_text is None:
                        ua_done = True
                        stopped_reason = "no_tools"
                        log(f"[user-agent] user satisfied — ending at turn {turn_count}")
                        break
                    user_agent_rounds += 1
                    ua_msg = Message(role="user", content=[TextBlock(text=f"[user_agent]\n{ua_text}")])
                    messages.append(ua_msg)
                    if writer and log_messages:
                        writer.write_event(TraceMessage(trace_id=trace_id, message=ua_msg))
                    log(f"[user-agent] round {user_agent_rounds}/{ua_max_rounds}: {ua_text[:100]}")
                    continue
                stopped_reason = "no_tools"
                log(f"[done] no tool calls — agent finished at turn {turn_count}")
                break

            if (
                enable_submit_report_tool
                and any(tu.name == SUBMIT_EXECUTOR_REPORT_NAME for tu in tool_uses)
                and len(tool_uses) != 1
            ):
                result_blocks = [
                    make_local_tool_result(
                        tu,
                        (
                            "Error: `submit_report` must be called alone after observing "
                            "environment tool results. Do not combine it with other tool calls."
                        ),
                        is_error=True,
                    )
                    for tu in tool_uses
                ]
                tool_msg = Message(role="user", content=result_blocks)
                messages.append(tool_msg)
                if writer and log_messages:
                    writer.write_event(TraceMessage(trace_id=trace_id, message=tool_msg))
                log("[protocol] submit_report mixed with another tool call; retrying")
                continue

            result_blocks: list[ToolResultBlock] = []
            media_blocks: list[ContentBlock] = []
            report_submitted_this_turn = False
            for tu in tool_uses:
                log(f"  -> tool: {tu.name}({brief(tu.input)})")

                if enable_submit_report_tool and tu.name == SUBMIT_EXECUTOR_REPORT_NAME:
                    if dispatched_tool_count < min_environment_tool_calls:
                        result_blocks.append(make_local_tool_result(
                            tu,
                            (
                                "Error: submit_report rejected. Complete at least "
                                f"{min_environment_tool_calls} environment tool call(s) first."
                            ),
                            is_error=True,
                        ))
                        log("  <- submit_report: ERR (environment tool required first)")
                        continue
                    submitted_report = format_executor_report(tu.input)
                    submitted_report_count += 1
                    report_submitted_this_turn = True
                    result_blocks.append(make_local_tool_result(tu, "Executor report submitted."))
                    log("  <- submit_report: OK (local)")
                    continue

                if tu.name == "todo" and todo_mgr:
                    result_text = todo_mgr.update(tu.input.get("items", []))
                    result = make_local_tool_result(tu, result_text)
                    result_blocks.append(result)
                    log("  <- todo: OK (local)")
                    continue

                if tu.name == "compact" and task.environment.enable_compact:
                    tokens_before = _estimate_tokens(messages)
                    msgs_before = len(messages)
                    messages = do_auto_compact(
                        messages,
                        provider,
                        keep_recent_on_summary=task.environment.compact_keep_recent_on_summary,
                        protect_tokens=task.environment.compact_protect_tokens,
                        todo_mgr=todo_mgr,
                        focus=tu.input.get("focus"),
                    )
                    auto_compact_count += 1
                    tokens_after = _estimate_tokens(messages)
                    if writer and log_messages:
                        writer.write_event(CompactEvent(
                            trace_id=trace_id,
                            layer="manual",
                            estimated_tokens_before=tokens_before,
                            estimated_tokens_after=tokens_after,
                            messages_before=msgs_before,
                            messages_after=len(messages),
                        ))
                    result = make_local_tool_result(
                        tu, f"Context compacted. {tokens_before} → {tokens_after} est. tokens."
                    )
                    result_blocks.append(result)
                    log(f"  <- compact: OK (local, {tokens_before} → {tokens_after} tokens)")
                    continue

                if (
                    max_environment_tool_calls is not None
                    and dispatched_tool_count >= max_environment_tool_calls
                ):
                    tool_budget_exhausted = True
                    result_blocks.append(make_local_tool_result(
                        tu,
                        (
                            "Error: environment tool budget exhausted "
                            f"({max_environment_tool_calls}). Stop using tools and report what is known."
                        ),
                        is_error=True,
                    ))
                    log(f"  <- {tu.name}: ERR (tool budget exhausted)")
                    continue

                dispatch_result = dispatcher.dispatch(tu, trace_id)
                dispatched_tool_count += 1
                environment_tool_names.append(tu.name)
                if len(dispatch_result) == 3:
                    result, dispatch_event, extra_media = dispatch_result
                else:
                    result, dispatch_event = dispatch_result
                    extra_media = None
                if _dispatch_writer and log_tool_dispatches:
                    _dispatch_writer.write_event(dispatch_event)
                result_blocks.append(result)
                environment_tool_summaries.append(_tool_result_summary(tu.name, result))
                if extra_media:
                    media_blocks.extend(extra_media)
                tool_time_s += dispatch_event.latency_ms / 1000.0
                status_tag = "OK" if not result.is_error else "ERR"
                log(f"  <- {tu.name}: {status_tag} ({dispatch_event.latency_ms:.0f}ms)")

            tool_msg = Message(role="user", content=result_blocks)
            messages.append(tool_msg)
            if writer and log_messages:
                writer.write_event(TraceMessage(trace_id=trace_id, message=tool_msg))

            if media_blocks:
                caption = TextBlock(text=f"[Visual content from tool results: {len(media_blocks)} image(s)]")
                media_msg = Message(role="user", content=[caption] + media_blocks)
                messages.append(media_msg)
                if writer and log_messages:
                    writer.write_event(TraceMessage(trace_id=trace_id, message=media_msg))
                log(f"  [media] injected {len(media_blocks)} image(s) into conversation")

            if report_submitted_this_turn:
                stopped_reason = "submitted_report"
                log("[done] executor submitted structured report")
                break

            if tool_budget_exhausted:
                stopped_reason = "tool_budget"
                log("[tool-budget] environment tool budget exhausted; ending action loop")
                break

    except Exception as exc:
        loop_error = f"{type(exc).__name__}: {exc}"
        loop_exc = exc
        stopped_reason = "error"
        log(f"[error] agent loop failed: {loop_error}")

    result = ReActLoopResult(
        messages=messages,
        turn_count=turn_count,
        usage=total_usage,
        model_time_s=model_time_s,
        tool_time_s=tool_time_s,
        stopped_reason=stopped_reason,
        error=loop_error,
        user_agent_rounds=user_agent_rounds,
        user_agent_done=ua_done,
        environment_tool_count=dispatched_tool_count,
        environment_tool_names=environment_tool_names,
        environment_tool_summaries=environment_tool_summaries,
        tool_budget_exhausted=tool_budget_exhausted,
        transitional_tool_retry_count=transitional_tool_retries,
        empty_visible_response_count=empty_visible_response_count,
        assistant_text_message_count=assistant_text_message_count,
        assistant_tool_message_count=assistant_tool_message_count,
        submitted_report=submitted_report,
        submitted_report_count=submitted_report_count,
    )

    if loop_error:
        raise loop_exc  # type: ignore[misc]

    return result
