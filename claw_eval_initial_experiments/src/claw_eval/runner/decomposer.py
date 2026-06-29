"""Hierarchical decomposer + executor task runner."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from .user_agent import UserAgent

from ..config import DecomposerRunConfig, MediaConfig, ModelConfig, PromptConfig, ReActRunConfig
from ..models.content import TextBlock, ToolResultBlock
from ..models.message import Message
from ..models.task import TaskDefinition
from ..models.trace import (
    AuditSnapshot,
    DecomposerSummary,
    DelegationEnd,
    DelegationStart,
    TokenUsage,
    TraceEnd,
    TraceMessage,
    TraceStart,
)
from ..trace.writer import TraceWriter
from .decomposer_prompts import build_decomposer_system_prompt, build_executor_subtask_user_content
from .decomposer_tools import (
    DECOMPOSER_TOOLS,
    DELEGATE_SUBTASK_NAME,
    SUBMIT_FINAL_ANSWER_NAME,
)
from .executor_report_tools import SUBMIT_EXECUTOR_REPORT_NAME, SUBMIT_EXECUTOR_REPORT_TOOL
from .loop_common import build_initial_user_content, log, make_local_tool_result
from .providers.openai_compat import OpenAICompatProvider
from .react_loop import ReActLoopResult, run_react_loop
from .system_prompt import build_system_prompt
from .todo import TodoManager
from .tool_setup import build_task_tools_and_dispatcher


def run_decomposer_task(
    task: TaskDefinition,
    decomposer_provider: OpenAICompatProvider,
    executor_provider: OpenAICompatProvider,
    trace_dir: str | Path = "traces",
    *,
    decomposer_cfg: DecomposerRunConfig | None = None,
    sandbox_tools: bool = False,
    sandbox_url: str | None = None,
    prompt_cfg: PromptConfig | None = None,
    decomposer_model_cfg: ModelConfig | None = None,
    executor_model_cfg: ModelConfig | None = None,
    media_cfg: MediaConfig | None = None,
    user_agent: UserAgent | None = None,
    react_cfg: ReActRunConfig | None = None,
) -> Path:
    """Execute one hierarchical decomposer trial and write a grading-compatible trace."""
    if task.user_agent.enabled and user_agent is not None:
        raise ValueError(
            "Decomposer runner does not support user-agent multi-turn tasks in v1 "
            f"(task {task.task_id} has user_agent.enabled=true)"
        )

    cfg = decomposer_cfg or DecomposerRunConfig()
    trace_id = str(uuid4())
    trace_path = Path(trace_dir) / f"{task.task_id}_{trace_id[:8]}.jsonl"

    task_tools, dispatcher, sandbox_tool_list = build_task_tools_and_dispatcher(
        task,
        sandbox_tools=sandbox_tools,
        sandbox_url=sandbox_url,
        media_cfg=media_cfg,
    )

    executor_max_turns = cfg.executor_max_turns or task.environment.max_turns
    min_delegations_before_final = max(0, cfg.min_delegations_before_final)
    combined_model = f"{decomposer_provider.model_id}+{executor_provider.model_id}"

    decomposer_usage = TokenUsage()
    executor_usage = TokenUsage()
    decomposer_model_time_s = 0.0
    executor_model_time_s = 0.0
    executor_tool_time_s = 0.0
    decomposer_turn_count = 0
    delegation_count = 0
    executor_turn_count = 0
    executor_report_status_counts: dict[str, int] = {}
    executor_environment_tool_count = 0
    executor_empty_visible_response_count = 0
    executor_assistant_text_message_count = 0
    executor_assistant_tool_message_count = 0
    executor_transitional_tool_retry_count = 0
    wall_start = time.monotonic()

    log(
        f"[decomposer start] task={task.task_id} "
        f"decomposer={decomposer_provider.model_id} executor={executor_provider.model_id} "
        f"trace={trace_path.name}"
    )

    with TraceWriter(trace_path) as main_writer:
        main_writer.write_event(TraceStart(
            trace_id=trace_id,
            task_id=task.task_id,
            model=combined_model,
            run_mode="decomposer",
            decomposer_model=decomposer_provider.model_id,
            executor_model=executor_provider.model_id,
        ))

        system_prompt = build_decomposer_system_prompt(
            task,
            prompt_cfg,
            manager_valid_tool_guidance=cfg.manager_valid_tool_guidance,
        )
        if decomposer_model_cfg and decomposer_model_cfg.system_prompt_prefix:
            system_prompt = decomposer_model_cfg.system_prompt_prefix + "\n\n" + system_prompt

        user_content = build_initial_user_content(
            task,
            trace_id=trace_id,
            writer=main_writer,
            model_cfg=decomposer_model_cfg,
            media_cfg=media_cfg,
        )
        messages: list[Message] = [
            Message(role="system", content=[TextBlock(text=system_prompt)]),
            Message(role="user", content=user_content),
        ]
        main_writer.write_event(TraceMessage(trace_id=trace_id, message=messages[-1]))

        loop_error: str | None = None
        loop_exc: Exception | None = None
        final_answer: str | None = None
        failure_modes: list[str] = []
        pending_protocol_errors: list[str] = []

        def remember_protocol_error(*codes: str) -> None:
            for code in ("decomposer_protocol_violation", *codes):
                if code not in pending_protocol_errors:
                    pending_protocol_errors.append(code)

        def clear_protocol_errors() -> None:
            pending_protocol_errors.clear()

        def append_failure_modes_once(codes: list[str]) -> None:
            for code in codes:
                if code not in failure_modes:
                    failure_modes.append(code)

        def protocol_correction(text: str) -> Message:
            return Message(role="user", content=[TextBlock(text="\n".join([
                "Protocol error in previous Decomposer response.",
                text,
                "Call exactly one native control tool now: `delegate_subtask` or `submit_final_answer`.",
                "If no Executor delegation has completed yet, call `delegate_subtask`.",
                "Do not answer in plain text.",
            ]))])

        try:
            while decomposer_turn_count < cfg.max_decomposer_turns:
                elapsed = time.monotonic() - wall_start
                if elapsed > task.environment.timeout_seconds:
                    log(f"[decomposer timeout] {elapsed:.1f}s exceeded limit")
                    failure_modes.append("timeout")
                    break

                log(f"[decomposer turn {decomposer_turn_count + 1}/{cfg.max_decomposer_turns}] calling model ...")
                model_t0 = time.monotonic()
                response, usage = decomposer_provider.chat(
                    messages,
                    tools=DECOMPOSER_TOOLS,
                    tool_choice="required",
                    strict_native_tools=True,
                    max_tokens=cfg.decomposer_max_output_tokens,
                )
                decomposer_model_time_s += time.monotonic() - model_t0
                decomposer_usage.input_tokens += usage.input_tokens
                decomposer_usage.output_tokens += usage.output_tokens
                decomposer_turn_count += 1

                main_writer.write_event(TraceMessage(
                    trace_id=trace_id,
                    message=response,
                    usage=usage,
                ))
                messages.append(response)

                tool_uses = [b for b in response.content if b.type == "tool_use"]
                text_blocks = [b for b in response.content if b.type == "text"]

                if not tool_uses:
                    remember_protocol_error("no_control_tool_call")
                    log("[decomposer protocol] no native control tool call")
                    correction_msg = protocol_correction(
                        "Your response did not include a native control tool call."
                    )
                    messages.append(correction_msg)
                    main_writer.write_event(TraceMessage(trace_id=trace_id, message=correction_msg))
                    continue

                result_blocks: list[ToolResultBlock] = []
                done = False
                if len(tool_uses) != 1:
                    remember_protocol_error("multiple_control_tool_calls")
                    log(f"[decomposer protocol] expected 1 control tool, got {len(tool_uses)}")
                    for tu in tool_uses:
                        result_blocks.append(make_local_tool_result(
                            tu,
                            "Error: call exactly one Decomposer control tool per turn.",
                            is_error=True,
                        ))
                    tool_msg = Message(role="user", content=result_blocks)
                    messages.append(tool_msg)
                    main_writer.write_event(TraceMessage(trace_id=trace_id, message=tool_msg))
                    continue

                for tu in tool_uses:
                    if tu.name == SUBMIT_FINAL_ANSWER_NAME:
                        if delegation_count < min_delegations_before_final:
                            remember_protocol_error("final_before_min_delegations")
                            result_blocks.append(make_local_tool_result(
                                tu,
                                (
                                    "Error: final answer rejected. Complete at least "
                                    f"{min_delegations_before_final} Executor delegation(s) first."
                                ),
                                is_error=True,
                            ))
                            log("[decomposer protocol] submit_final_answer before delegation")
                            continue

                        candidate_answer = str(tu.input.get("answer", "")).strip()
                        if not candidate_answer and text_blocks:
                            candidate_answer = text_blocks[0].text.strip()
                        if not candidate_answer:
                            remember_protocol_error("empty_final_answer")
                            result_blocks.append(make_local_tool_result(
                                tu,
                                (
                                    "Error: final answer rejected. `submit_final_answer` must include "
                                    "a non-empty `answer` string that satisfies the user's original task."
                                ),
                                is_error=True,
                            ))
                            log("[decomposer protocol] empty submit_final_answer")
                            continue

                        clear_protocol_errors()
                        final_answer = candidate_answer
                        result_blocks.append(make_local_tool_result(tu, "Final answer submitted."))
                        done = True
                        log("[decomposer] submit_final_answer")
                        break

                    if tu.name == DELEGATE_SUBTASK_NAME:
                        if delegation_count >= cfg.max_delegations:
                            result_blocks.append(make_local_tool_result(
                                tu,
                                f"Error: max delegations ({cfg.max_delegations}) reached.",
                                is_error=True,
                            ))
                            continue

                        subtask = str(tu.input.get("subtask", "")).strip()
                        context = str(tu.input.get("context", "") or "").strip()
                        if not subtask:
                            remember_protocol_error("empty_delegate_subtask")
                            result_blocks.append(make_local_tool_result(
                                tu, "Error: subtask must be non-empty.", is_error=True,
                            ))
                            continue

                        clear_protocol_errors()
                        delegation_count += 1
                        idx = delegation_count
                        sidecar_path = trace_path.with_name(
                            f"{trace_path.stem}_exec_{idx}.jsonl"
                        )
                        main_writer.write_event(DelegationStart(
                            trace_id=trace_id,
                            delegation_index=idx,
                            subtask=subtask,
                            context=context,
                        ))
                        log(f"[delegation {idx}] subtask: {subtask[:120]}")

                        report, exec_stats = _run_executor_delegation(
                            task=task,
                            subtask=subtask,
                            context=context,
                            executor_provider=executor_provider,
                            task_tools=task_tools,
                            dispatcher=dispatcher,
                            sandbox_tool_list=sandbox_tool_list,
                            trace_id=trace_id,
                            main_writer=main_writer,
                            sidecar_path=sidecar_path,
                            prompt_cfg=prompt_cfg,
                            executor_model_cfg=executor_model_cfg,
                            media_cfg=media_cfg,
                            max_turns=executor_max_turns,
                            timeout_seconds=task.environment.timeout_seconds,
                            wall_start=wall_start,
                            min_tool_calls=cfg.executor_min_tool_calls,
                            max_environment_tool_calls=cfg.executor_max_environment_tool_calls,
                            executor_max_output_tokens=cfg.executor_max_output_tokens,
                            report_max_tokens=cfg.executor_report_max_tokens,
                            report_mode=cfg.executor_report_mode,
                            synthetic_failure_report=cfg.executor_synthetic_failure_report,
                            executor_prompt_mode=cfg.executor_prompt_mode,
                            evidence_mode=cfg.executor_evidence_mode,
                            evidence_max_chars=cfg.executor_evidence_max_chars,
                            react_cfg=react_cfg,
                        )
                        executor_usage.input_tokens += exec_stats.usage.input_tokens
                        executor_usage.output_tokens += exec_stats.usage.output_tokens
                        executor_model_time_s += exec_stats.model_time_s
                        executor_tool_time_s += exec_stats.tool_time_s
                        executor_turn_count += exec_stats.turn_count
                        executor_report_status_counts[exec_stats.report_status] = (
                            executor_report_status_counts.get(exec_stats.report_status, 0) + 1
                        )
                        executor_environment_tool_count += exec_stats.environment_tool_count
                        executor_empty_visible_response_count += exec_stats.empty_visible_response_count
                        executor_assistant_text_message_count += exec_stats.assistant_text_message_count
                        executor_assistant_tool_message_count += exec_stats.assistant_tool_message_count
                        executor_transitional_tool_retry_count += exec_stats.transitional_tool_retry_count

                        main_writer.write_event(DelegationEnd(
                            trace_id=trace_id,
                            delegation_index=idx,
                            report=report,
                            executor_turns=exec_stats.turn_count,
                            executor_tokens=exec_stats.usage,
                            sidecar_trace=str(sidecar_path.name),
                            report_status=exec_stats.report_status,
                            executor_stopped_reason=exec_stats.executor_stopped_reason,
                            executor_environment_tool_count=exec_stats.environment_tool_count,
                            executor_environment_tool_names=exec_stats.environment_tool_names,
                            executor_empty_visible_response_count=exec_stats.empty_visible_response_count,
                            executor_assistant_text_message_count=exec_stats.assistant_text_message_count,
                            executor_assistant_tool_message_count=exec_stats.assistant_tool_message_count,
                            executor_transitional_tool_retry_count=exec_stats.transitional_tool_retry_count,
                        ))
                        result_blocks.append(make_local_tool_result(tu, report))
                        continue

                    remember_protocol_error("unknown_decomposer_tool")
                    result_blocks.append(make_local_tool_result(
                        tu,
                        f"Error: unknown decomposer tool {tu.name!r}.",
                        is_error=True,
                    ))

                if done:
                    break

                tool_msg = Message(role="user", content=result_blocks)
                messages.append(tool_msg)
                main_writer.write_event(TraceMessage(trace_id=trace_id, message=tool_msg))

            else:
                failure_modes.append("max_decomposer_turns")

            if not final_answer and pending_protocol_errors:
                append_failure_modes_once(pending_protocol_errors)

            if final_answer:
                final_msg = Message(role="assistant", content=[TextBlock(text=final_answer)])
                main_writer.write_event(TraceMessage(trace_id=trace_id, message=final_msg))
            elif not failure_modes:
                failure_modes.append("no_final_answer")

            if delegation_count == 0 and not final_answer:
                failure_modes.append("zero_delegations")

        except Exception as exc:
            loop_error = f"{type(exc).__name__}: {exc}"
            loop_exc = exc
            failure_modes.append(loop_error)
            log(f"[decomposer error] {loop_error}")

        import httpx as _httpx

        for svc in task.services:
            if svc.reset_endpoint:
                audit_url = svc.reset_endpoint.rsplit("/reset", 1)[0] + "/audit"
                try:
                    resp = _httpx.get(audit_url, timeout=5)
                    main_writer.write_event(AuditSnapshot(
                        trace_id=trace_id,
                        service_name=svc.name,
                        audit_url=audit_url,
                        audit_data=resp.json(),
                    ))
                except Exception:
                    pass

        wall_time = time.monotonic() - wall_start
        total_input = decomposer_usage.input_tokens + executor_usage.input_tokens
        total_output = decomposer_usage.output_tokens + executor_usage.output_tokens
        total_model_time = decomposer_model_time_s + executor_model_time_s

        main_writer.write_event(DecomposerSummary(
            trace_id=trace_id,
            decomposer_turns=decomposer_turn_count,
            delegation_count=delegation_count,
            executor_turns=executor_turn_count,
            decomposer_tokens=decomposer_usage,
            executor_tokens=executor_usage,
            decomposer_model_time_s=round(decomposer_model_time_s, 2),
            executor_model_time_s=round(executor_model_time_s, 2),
            executor_report_status_counts=executor_report_status_counts,
            executor_environment_tool_count=executor_environment_tool_count,
            executor_empty_visible_response_count=executor_empty_visible_response_count,
            executor_assistant_text_message_count=executor_assistant_text_message_count,
            executor_assistant_tool_message_count=executor_assistant_tool_message_count,
            executor_transitional_tool_retry_count=executor_transitional_tool_retry_count,
        ))

        main_writer.write_event(TraceEnd(
            trace_id=trace_id,
            total_turns=decomposer_turn_count + executor_turn_count,
            model_input_tokens=total_input,
            model_output_tokens=total_output,
            input_tokens=total_input,
            output_tokens=total_output,
            total_tokens=total_input + total_output,
            model_time_s=round(total_model_time, 2),
            tool_time_s=round(executor_tool_time_s, 2),
            other_time_s=round(max(0.0, wall_time - total_model_time - executor_tool_time_s), 2),
            wall_time_s=round(wall_time, 2),
            failure_modes=failure_modes,
        ))

        if loop_error:
            raise loop_exc

    log(
        f"[decomposer end] delegations={delegation_count} decomposer_turns={decomposer_turn_count} "
        f"executor_turns={executor_turn_count} tokens={total_input + total_output}"
    )
    dispatcher.close()
    return trace_path


class _ExecutorStats:
    __slots__ = (
        "turn_count",
        "usage",
        "model_time_s",
        "tool_time_s",
        "report_status",
        "executor_stopped_reason",
        "environment_tool_count",
        "environment_tool_names",
        "empty_visible_response_count",
        "assistant_text_message_count",
        "assistant_tool_message_count",
        "transitional_tool_retry_count",
    )

    def __init__(
        self,
        turn_count: int,
        usage: TokenUsage,
        model_time_s: float,
        tool_time_s: float,
        report_status: str = "natural",
        executor_stopped_reason: str = "",
        environment_tool_count: int = 0,
        environment_tool_names: list[str] | None = None,
        empty_visible_response_count: int = 0,
        assistant_text_message_count: int = 0,
        assistant_tool_message_count: int = 0,
        transitional_tool_retry_count: int = 0,
    ) -> None:
        self.turn_count = turn_count
        self.usage = usage
        self.model_time_s = model_time_s
        self.tool_time_s = tool_time_s
        self.report_status = report_status
        self.executor_stopped_reason = executor_stopped_reason
        self.environment_tool_count = environment_tool_count
        self.environment_tool_names = list(environment_tool_names or [])
        self.empty_visible_response_count = empty_visible_response_count
        self.assistant_text_message_count = assistant_text_message_count
        self.assistant_tool_message_count = assistant_tool_message_count
        self.transitional_tool_retry_count = transitional_tool_retry_count


_EXECUTOR_REPORT_REQUEST = """Stop using tools. Produce only the compact report for the coordinator.

Use this structure:
- What you did
- Key findings / outputs
- Any blockers or open questions

If the subtask could not be completed, state exactly what was attempted and what blocked it.
""".strip()


def _looks_like_non_report(text: str) -> bool:
    stripped = text.strip()
    return len(stripped.split()) < 4


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _synthetic_executor_failure_report(loop_result: ReActLoopResult, reason: str) -> str:
    blockers = [
        "executor_failed_no_report",
        "executor_finished_without_report",
        f"reason={_one_line(reason)}",
    ]
    if loop_result.tool_budget_exhausted:
        blockers.append("environment_tool_budget_exhausted")
    if loop_result.stopped_reason == "missing_required_tool":
        blockers.append("executor_did_not_satisfy_required_tool_use")
    if loop_result.error:
        blockers.append(_one_line(loop_result.error))

    return "\n".join([
        "- What you did: Executor attempted the delegated subtask but did not return a valid coordinator report.",
        "- Key findings / outputs: unavailable from Executor report.",
        "- Tool results that matter for next steps: not provided to coordinator.",
        "- Blockers / open questions: " + "; ".join(blockers),
        "- Coordinator note: this is a report-generation failure, not evidence that tools or data are unavailable.",
    ])


def _executor_report_from_action_loop(
    loop_result: ReActLoopResult,
    *,
    require_structured_report: bool = False,
    synthetic_failure_report: bool = True,
) -> tuple[str, str]:
    def failure_report(reason: str, fallback_report: str = "") -> tuple[str, str]:
        fallback = fallback_report.strip()
        if synthetic_failure_report:
            return _synthetic_executor_failure_report(loop_result, reason), "synthetic_failure"
        if fallback:
            return fallback, "natural"
        return "", "missing_report"

    if loop_result.stopped_reason == "submitted_report":
        report = loop_result.submitted_report.strip()
        if _looks_like_non_report(report):
            return failure_report(
                "executor submit_report returned empty or too-short text",
                fallback_report=report,
            )
        return report, "structured"

    if require_structured_report:
        return failure_report(
            f"executor action loop did not call submit_report: {loop_result.stopped_reason}",
        )

    if loop_result.stopped_reason != "no_tools":
        return failure_report(
            f"executor action loop stopped before a final report: {loop_result.stopped_reason}",
            fallback_report=loop_result.final_report,
        )

    report = loop_result.final_report.strip()
    if _looks_like_non_report(report):
        return failure_report(
            "executor action loop returned empty or too-short text",
            fallback_report=report,
        )
    return report, "natural"


def _compact_tool_evidence(loop_result: ReActLoopResult, max_chars: int) -> str:
    summaries = loop_result.environment_tool_summaries
    if not summaries or max_chars <= 0:
        return ""

    lines = ["[Compact tool evidence passed through from Executor trace]"]
    lines.extend(f"{idx}. {summary}" for idx, summary in enumerate(summaries, start=1))
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars - 3] + "..."
    return text


def _append_executor_evidence(
    report: str,
    loop_result: ReActLoopResult,
    *,
    evidence_mode: str,
    evidence_max_chars: int,
) -> str:
    if evidence_mode == "none":
        return report
    if evidence_mode != "tool_summary":
        raise ValueError(f"Unknown executor_evidence_mode: {evidence_mode!r}")

    evidence = _compact_tool_evidence(loop_result, evidence_max_chars)
    if not evidence:
        return report
    return f"{report}\n\n{evidence}"


def _produce_executor_report(
    *,
    loop_result: ReActLoopResult,
    provider: OpenAICompatProvider,
    trace_id: str,
    writer: TraceWriter,
    report_max_tokens: int | None,
) -> tuple[str, str]:
    report_prompt = Message(role="user", content=[TextBlock(text=_EXECUTOR_REPORT_REQUEST)])
    loop_result.messages.append(report_prompt)
    writer.write_event(TraceMessage(trace_id=trace_id, message=report_prompt))

    try:
        model_t0 = time.monotonic()
        response, usage = provider.chat(
            loop_result.messages,
            tools=[],
            max_tokens=report_max_tokens,
        )
        loop_result.model_time_s += time.monotonic() - model_t0
        loop_result.usage.input_tokens += usage.input_tokens
        loop_result.usage.output_tokens += usage.output_tokens
        loop_result.turn_count += 1
        loop_result.messages.append(response)
        writer.write_event(TraceMessage(trace_id=trace_id, message=response, usage=usage))
    except Exception as exc:
        reason = f"report phase failed: {type(exc).__name__}: {exc}"
        log(f"[executor report] {reason}")
        return _synthetic_executor_failure_report(loop_result, reason), "synthetic_failure"

    report = response.text.strip()
    if _looks_like_non_report(report):
        return (
            _synthetic_executor_failure_report(
                loop_result,
                "report phase returned empty or too-short text",
            ),
            "synthetic_failure",
        )
    return report, "repaired"


def _run_executor_delegation(
    *,
    task: TaskDefinition,
    subtask: str,
    context: str,
    executor_provider: OpenAICompatProvider,
    task_tools,
    dispatcher,
    sandbox_tool_list,
    trace_id: str,
    main_writer: TraceWriter,
    sidecar_path: Path,
    prompt_cfg: PromptConfig | None,
    executor_model_cfg: ModelConfig | None,
    media_cfg: MediaConfig | None,
    max_turns: int,
    timeout_seconds: int,
    wall_start: float,
    min_tool_calls: int = 1,
    max_environment_tool_calls: int | None = 20,
    executor_max_output_tokens: int | None = 1024,
    report_max_tokens: int | None = 512,
    report_mode: str = "strict",
    synthetic_failure_report: bool = True,
    executor_prompt_mode: str = "report_wrapper",
    evidence_mode: str = "none",
    evidence_max_chars: int = 2000,
    react_cfg: ReActRunConfig | None = None,
) -> tuple[str, _ExecutorStats]:
    """Run one executor subtask; return (report, stats)."""
    executor_system = build_system_prompt(task, prompt_cfg, extra_tools=sandbox_tool_list)
    if executor_model_cfg and executor_model_cfg.system_prompt_prefix:
        executor_system = executor_model_cfg.system_prompt_prefix + "\n\n" + executor_system

    structured_report = report_mode in {"structured", "structured_repair"}
    executor_tools = list(task_tools)
    if structured_report:
        executor_tools.append(SUBMIT_EXECUTOR_REPORT_TOOL)
    has_environment_tools = bool(task.tools or sandbox_tool_list)
    effective_min_tool_calls = min_tool_calls if has_environment_tools else 0

    subtask_text = build_executor_subtask_user_content(
        subtask,
        context,
        report_tool_name=SUBMIT_EXECUTOR_REPORT_NAME if structured_report else None,
        prompt_mode=executor_prompt_mode,
    )
    initial_messages = [
        Message(role="system", content=[TextBlock(text=executor_system)]),
        Message(role="user", content=[TextBlock(text=subtask_text)]),
    ]

    todo_mgr = TodoManager() if task.environment.enable_todo else None

    with TraceWriter(sidecar_path) as sidecar_writer:
        sidecar_writer.write_event(TraceStart(
            trace_id=trace_id,
            task_id=task.task_id,
            model=executor_provider.model_id,
            run_mode="executor_subtask",
        ))
        sidecar_writer.write_event(TraceMessage(trace_id=trace_id, message=initial_messages[-1]))

        loop_result = run_react_loop(
            initial_messages=initial_messages,
            task=task,
            provider=executor_provider,
            task_tools=executor_tools,
            dispatcher=dispatcher,
            trace_id=trace_id,
            writer=sidecar_writer,
            dispatch_writer=sidecar_writer,
            log_messages=True,
            log_tool_dispatches=True,
            max_turns=max_turns,
            timeout_seconds=timeout_seconds,
            wall_start=wall_start,
            model_cfg=executor_model_cfg,
            media_cfg=media_cfg,
            todo_mgr=todo_mgr,
            min_environment_tool_calls=effective_min_tool_calls,
            max_environment_tool_calls=max_environment_tool_calls,
            max_output_tokens=executor_max_output_tokens,
            react_cfg=react_cfg,
            enable_submit_report_tool=structured_report,
        )

        if report_mode == "strict":
            report, report_status = _executor_report_from_action_loop(
                loop_result,
                synthetic_failure_report=synthetic_failure_report,
            )
        elif report_mode == "repair":
            report, report_status = _produce_executor_report(
                loop_result=loop_result,
                provider=executor_provider,
                trace_id=trace_id,
                writer=sidecar_writer,
                report_max_tokens=report_max_tokens,
            )
        elif report_mode == "structured":
            report, report_status = _executor_report_from_action_loop(
                loop_result,
                require_structured_report=True,
                synthetic_failure_report=synthetic_failure_report,
            )
        elif report_mode == "structured_repair":
            report, report_status = _executor_report_from_action_loop(
                loop_result,
                require_structured_report=True,
                synthetic_failure_report=synthetic_failure_report,
            )
            if report_status in {"synthetic_failure", "missing_report"}:
                report, report_status = _produce_executor_report(
                    loop_result=loop_result,
                    provider=executor_provider,
                    trace_id=trace_id,
                    writer=sidecar_writer,
                    report_max_tokens=report_max_tokens,
                )
        else:
            raise ValueError(f"Unknown executor_report_mode: {report_mode!r}")

        report = _append_executor_evidence(
            report,
            loop_result,
            evidence_mode=evidence_mode,
            evidence_max_chars=evidence_max_chars,
        )

        sidecar_failure_modes = [loop_result.error] if loop_result.error else []
        if report_status in {"synthetic_failure", "missing_report"}:
            sidecar_failure_modes.append("executor_failed_no_report")

        sidecar_writer.write_event(TraceEnd(
            trace_id=trace_id,
            total_turns=loop_result.turn_count,
            model_input_tokens=loop_result.usage.input_tokens,
            model_output_tokens=loop_result.usage.output_tokens,
            input_tokens=loop_result.usage.input_tokens,
            output_tokens=loop_result.usage.output_tokens,
            total_tokens=loop_result.usage.input_tokens + loop_result.usage.output_tokens,
            model_time_s=round(loop_result.model_time_s, 2),
            tool_time_s=round(loop_result.tool_time_s, 2),
            wall_time_s=0.0,
            failure_modes=sidecar_failure_modes,
        ))

    return report, _ExecutorStats(
        turn_count=loop_result.turn_count,
        usage=loop_result.usage,
        model_time_s=loop_result.model_time_s,
        tool_time_s=loop_result.tool_time_s,
        report_status=report_status,
        executor_stopped_reason=loop_result.stopped_reason,
        environment_tool_count=loop_result.environment_tool_count,
        environment_tool_names=loop_result.environment_tool_names,
        empty_visible_response_count=loop_result.empty_visible_response_count,
        assistant_text_message_count=loop_result.assistant_text_message_count,
        assistant_tool_message_count=loop_result.assistant_tool_message_count,
        transitional_tool_retry_count=loop_result.transitional_tool_retry_count,
    )
