"""Tests for decomposer prompt builders."""

from __future__ import annotations

from claw_eval.models.task import Environment, Prompt, TaskDefinition
from claw_eval.models.tool import ToolSpec
from claw_eval.runner.decomposer_prompts import (
    build_decomposer_system_prompt,
    build_executor_subtask_user_content,
)


def test_decomposer_system_prompt_mentions_control_tools():
    task = TaskDefinition(
        task_id="T_test",
        task_name="Test",
        prompt=Prompt(text="Do something"),
    )
    prompt = build_decomposer_system_prompt(task, None)
    assert "delegate_subtask" in prompt
    assert "submit_final_answer" in prompt
    assert "cannot call environment tools" in prompt.lower() or "cannot" in prompt.lower()
    assert "first action must be a native `delegate_subtask`" in prompt
    assert "Plain text is never a valid control action" in prompt
    assert "exactly one native control tool call" in prompt
    assert "Never call both `delegate_subtask` and `submit_final_answer`" in prompt
    assert "Never emit an empty `submit_final_answer`" in prompt


def test_decomposer_system_prompt_includes_mock_today_for_manager():
    task = TaskDefinition(
        task_id="T_test",
        task_name="Test",
        prompt=Prompt(text="Do something tomorrow"),
        environment=Environment(mock_today="2026-03-26"),
    )
    prompt = build_decomposer_system_prompt(task, None)

    assert "## Current Date" in prompt
    assert "2026-03-26" in prompt
    assert "Use this date for all relative-date reasoning" in prompt
    assert "include the exact absolute date in the subtask" in prompt


def test_decomposer_system_prompt_includes_final_answer_checklist():
    task = TaskDefinition(
        task_id="T_test",
        task_name="Test",
        prompt=Prompt(text="Do something"),
    )
    prompt = build_decomposer_system_prompt(task, None)

    assert "## Final Answer Checklist" in prompt
    assert "Every numbered or explicit requirement" in prompt
    assert "Required side effects were completed" in prompt
    assert "saved draft IDs" in prompt
    assert "No requirement depends only on an Executor failure report" in prompt


def test_decomposer_system_prompt_delegation_contract_is_report_aligned():
    task = TaskDefinition(
        task_id="T_test",
        task_name="Test",
        prompt=Prompt(text="Do something"),
    )
    prompt = build_decomposer_system_prompt(task, None)

    assert "## Delegation Report Contract" in prompt
    assert "Delegate natural-language subtasks" in prompt
    assert "concrete completion evidence" in prompt
    assert "created or updated artifact IDs" in prompt
    assert "Do not ask for raw tool outputs" in prompt
    assert "verify progress from its summarized report artifacts" in prompt


def test_decomposer_system_prompt_can_include_valid_executor_tools():
    task = TaskDefinition(
        task_id="T_test",
        task_name="Test",
        prompt=Prompt(text="Do something"),
        tools=[
            ToolSpec(name="kb_search", description="Search the KB"),
            ToolSpec(name="kb_get_article", description="Get a KB article"),
        ],
    )
    default_prompt = build_decomposer_system_prompt(task, None)
    strict_prompt = build_decomposer_system_prompt(
        task,
        None,
        manager_valid_tool_guidance=True,
    )

    assert "Executor Environment Tools" not in default_prompt
    assert "Executor Environment Tools" in strict_prompt
    assert "`kb_search`: Search the KB" in strict_prompt
    assert "`kb_get_article`: Get a KB article" in strict_prompt
    assert "Do not ask the Executor to list tools" in strict_prompt

def test_executor_subtask_envelope():
    text = build_executor_subtask_user_content("Search the KB. Focus on VPN.", "ignored context")
    assert "Search the KB. Focus on VPN." in text
    assert "ignored context" not in text
    assert "Context from coordinator" not in text
    assert "First use the available tools to complete the subtask." in text
    assert "compact report" in text.lower()
    assert "Tool results that matter" not in text


def test_executor_subtask_flat_prompt_mode():
    text = build_executor_subtask_user_content(
        "  Search the KB. Focus on VPN.  ",
        "ignored context",
        report_tool_name="submit_report",
        prompt_mode="flat_subtask",
    )
    assert text == "Search the KB. Focus on VPN."
    assert "ignored context" not in text
    assert "You are the Executor" not in text
    assert "compact report" not in text.lower()
    assert "submit_report" not in text


def test_executor_subtask_envelope_with_structured_report_tool():
    text = build_executor_subtask_user_content(
        "Search the KB. Focus on VPN.",
        "ignored context",
        report_tool_name="submit_report",
    )
    assert "`submit_report`" in text
    assert "plain text final reports are invalid" in text
    assert "Call it alone" in text
    assert "ignored context" not in text
