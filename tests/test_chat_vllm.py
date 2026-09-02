import pytest
from langchain_core.messages import AIMessage

from decomposer.chat_vllm import ChatVLLM


def _response(*, finish_reason: str = "stop") -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "answer",
                    "reasoning": "reasoning",
                },
                "finish_reason": finish_reason,
            }
        ],
    }


def test_reasoning_round_trip() -> None:
    model = ChatVLLM(
        model="test",
        api_key="test",
        preserve_reasoning=True,
        use_responses_api=False,
    )

    result = model._create_chat_result(_response())
    message = result.generations[0].message
    assert message.additional_kwargs["reasoning_content"] == "reasoning"
    assert message.content_blocks[0] == {
        "type": "reasoning",
        "reasoning": "reasoning",
    }

    payload = model._get_request_payload([message])
    assert payload["messages"][0]["reasoning"] == "reasoning"


def test_legacy_reasoning_round_trip() -> None:
    model = ChatVLLM(
        model="test",
        api_key="test",
        preserve_reasoning=True,
        use_responses_api=False,
    )
    message = AIMessage(
        content="answer",
        additional_kwargs={"reasoning": "legacy reasoning"},
    )

    payload = model._get_request_payload([message])

    assert payload["messages"][0]["reasoning"] == "legacy reasoning"


def test_length_limit_is_an_error() -> None:
    model = ChatVLLM(model="test", api_key="test")

    with pytest.raises(RuntimeError, match="max_completion_tokens"):
        model._create_chat_result(_response(finish_reason="length"))


def test_qwen_xml_tool_calls_are_parsed_client_side() -> None:
    model = ChatVLLM(
        model="test",
        api_key="test",
        parse_qwen_xml_tool_calls=True,
        use_responses_api=False,
    )
    response = _response()
    response["choices"][0]["message"]["content"] = """
Before the call.
<tool_call>
<function=spawn_subagent>
<parameter=subagent_type_id>
qwen_3_5_4b_non_thinking
</parameter>
<parameter=prompt>
Solve the task.
</parameter>
</function>
</tool_call>
"""

    result = model._create_chat_result(response)
    message = result.generations[0].message

    assert message.content == "Before the call."
    assert message.tool_calls == [
        {
            "name": "spawn_subagent",
            "args": {
                "subagent_type_id": "qwen_3_5_4b_non_thinking",
                "prompt": "Solve the task.",
            },
            "id": message.tool_calls[0]["id"],
            "type": "tool_call",
        }
    ]


def test_qwen_xml_mode_forces_tool_choice_none() -> None:
    model = ChatVLLM(
        model="test",
        api_key="test",
        parse_qwen_xml_tool_calls=True,
        use_responses_api=False,
    )

    payload = model._get_request_payload(
        [{"role": "user", "content": "delegate"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "wait",
                    "description": "Wait",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        tool_choice="auto",
    )

    assert payload["tool_choice"] == "none"
