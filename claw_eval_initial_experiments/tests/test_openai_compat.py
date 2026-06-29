"""Tests for OpenAI-compatible provider response parsing."""

from __future__ import annotations

from claw_eval.runner.providers.openai_compat import OpenAICompatProvider


class _Obj:
    pass


def _response_with_text(content: str):
    msg = _Obj()
    msg.content = content
    msg.tool_calls = None
    msg.reasoning_content = None

    choice = _Obj()
    choice.message = msg

    response = _Obj()
    response.choices = [choice]
    response.usage = None
    return response


def test_pseudo_tool_markup_not_accepted_for_strict_native_tool_mode():
    provider = OpenAICompatProvider(model_id="test")
    response = _response_with_text(
        "<tool_call><function=delegate_subtask>"
        "<parameter=subtask>Search records</parameter>"
        "</tool_call>"
    )

    loose_msg, _ = provider._parse_response(response)
    strict_msg, _ = provider._parse_response(response, strict_native_tools=True)

    assert any(block.type == "tool_use" for block in loose_msg.content)
    assert not any(block.type == "tool_use" for block in strict_msg.content)
    assert "<tool_call>" in strict_msg.text


def test_native_tool_argument_value_error_falls_back_to_empty_args():
    provider = OpenAICompatProvider(model_id="test")

    fn = _Obj()
    fn.name = "gmail_list_messages"
    fn.arguments = "{\"days_back\": " + ("9" * 4301) + "}"

    tc = _Obj()
    tc.id = "tool-1"
    tc.function = fn

    msg = _Obj()
    msg.content = None
    msg.tool_calls = [tc]
    msg.reasoning_content = None

    choice = _Obj()
    choice.message = msg

    response = _Obj()
    response.choices = [choice]
    response.usage = None

    parsed, _ = provider._parse_response(response, strict_native_tools=True)

    tool = next(block for block in parsed.content if block.type == "tool_use")
    assert tool.name == "gmail_list_messages"
    assert tool.input == {}
