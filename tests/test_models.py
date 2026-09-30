import pytest
from langchain_core.messages import AIMessage

from decomposer.models import ChatVLLM, create_model


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


def test_flash_next_uses_nonthinking_sampling():
    model = create_model("qwen_3_8_flash_next_non_thinking")
    payload = model._get_request_payload("test")
    assert "reasoning_effort" not in payload
    assert payload["temperature"] == .7
    assert payload["top_p"] == .8
    assert payload["presence_penalty"] == 1.5
    assert payload["extra_body"] == {
        "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    assert model.preserve_reasoning is False
    assert "max_tokens" not in payload
    assert "max_completion_tokens" not in payload


@pytest.mark.parametrize("profile, thinking, temperature, top_p", [
    ("qwen_3_5_4b_unlooped_thinking", True, .6, .95),
    ("qwen_3_5_4b_unlooped_non_thinking", False, .7, .8),
])
def test_unlooped_uses_checkpoint_sampling(profile, thinking, temperature, top_p):
    model = create_model(profile)
    payload = model._get_request_payload("test")
    assert payload["model"] == "Qwen/Qwen3.5-4B-unlooped"
    assert payload["temperature"] == temperature
    assert payload["top_p"] == top_p
    assert "presence_penalty" not in payload
    assert payload["extra_body"] == {"top_k": 20, "chat_template_kwargs": {"enable_thinking": thinking}}
    assert model.preserve_reasoning is thinking
