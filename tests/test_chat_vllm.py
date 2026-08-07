import pytest

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
    assert message.additional_kwargs["reasoning"] == "reasoning"

    payload = model._get_request_payload([message])
    assert payload["messages"][0]["reasoning"] == "reasoning"


def test_length_limit_is_an_error() -> None:
    model = ChatVLLM(model="test", api_key="test")

    with pytest.raises(RuntimeError, match="max_completion_tokens"):
        model._create_chat_result(_response(finish_reason="length"))
