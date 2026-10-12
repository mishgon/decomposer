import copy
import json

import pytest

from gyms.gaia2.model_proxy import chat_response, completion_payload


class Tokenizer:
    def get_vocab(self):
        return {"<tool_call>": 1, "</tool_call>": 2, "<think>": 3, "</think>": 4}

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.options = kwargs
        return {"input_ids": [1, 2, 3]}


def test_completion_preserves_schema_sampling_and_tool_history():
    payload = {
        "model": "Qwen/Qwen3.5-4B",
        "messages": [
            {"role": "assistant", "content": "", "reasoning": "Check recipients",
             "tool_calls": [{"function": {"name": "send", "arguments": '{"recipients": ["a@example.com"], "cc": null}'}}]},
            {"role": "tool", "tool_call_id": "call-1", "content": "Sent"},
        ],
        "tools": [{"type": "function", "function": {"name": "send", "parameters": {
            "type": "object", "properties": {"recipients": {"anyOf": [
                {"type": "array", "items": {"type": "string"}}, {"type": "null"}
            ]}}
        }}}],
        "tool_choice": "auto", "stream": False,
        "temperature": 1.0, "top_p": .95, "top_k": 20, "presence_penalty": 1.5,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    original = copy.deepcopy(payload)
    tokenizer = Tokenizer()
    result = completion_payload(payload, tokenizer, 131072)
    assert payload == original
    assert tokenizer.options["tools"] == original["tools"]
    assert tokenizer.messages[0]["reasoning_content"] == "Check recipients"
    assert tokenizer.messages[0]["tool_calls"][0]["function"]["arguments"] == {
        "recipients": ["a@example.com"], "cc": None,
    }
    assert tokenizer.messages[1] == original["messages"][1]
    assert result["max_tokens"] == 131069
    for key in ("temperature", "top_p", "top_k", "presence_penalty"):
        assert result[key] == original[key]
    assert not {"tools", "messages", "tool_choice", "chat_template_kwargs"} & result.keys()


def test_explicit_generation_limit_and_disabled_thinking():
    tokenizer = Tokenizer()
    result = completion_payload({
        "messages": [], "max_completion_tokens": 100,
        "tool_choice": "none", "chat_template_kwargs": {"enable_thinking": False},
    }, tokenizer, 131072)
    assert result["max_tokens"] == 100
    assert tokenizer.options["enable_thinking"] is False


@pytest.mark.parametrize("options", [{"stream": True}, {"tool_choice": "required"}])
def test_unsupported_modes_fail_explicitly(options):
    with pytest.raises(ValueError):
        completion_payload({"messages": [], **options}, Tokenizer(), 131072)


@pytest.mark.parametrize("phone", ["+441234567890", "0012345", "123456", "null"])
def test_vllm_019_parser_preserves_nullable_values(phone):
    if not pytest.importorskip("vllm").__version__.startswith("0.19."):
        pytest.skip("Run parser regression tests in the vLLM 0.19 server environment")
    payload = {
        "model": "Qwen/Qwen3.5-4B", "messages": [],
        "tools": [{"type": "function", "function": {"name": "send", "parameters": {
            "type": "object", "properties": {
                "phone": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "recipients": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]},
                "price": {"anyOf": [{"type": "integer"}, {"type": "number"}, {"type": "null"}]},
            },
        }}}],
    }
    original = copy.deepcopy(payload)
    response = {
        "id": "test", "created": 1, "model": payload["model"], "usage": {},
        "choices": [{"index": 0, "finish_reason": "stop", "text": (
            "Reasoning</think>\n<tool_call>\n<function=send>\n"
            f"<parameter=phone>\n{phone}\n</parameter>\n"
            '<parameter=recipients>\n["a@example.com"]\n</parameter>\n'
            '<parameter=price>\n12.5\n</parameter>\n</function>\n</tool_call>'
        )}],
    }
    result = chat_response(response, payload, Tokenizer())
    args = json.loads(result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    assert args == {"phone": None if phone == "null" else phone,
                    "recipients": ["a@example.com"], "price": 12.5}
    assert payload == original
    response["choices"][0]["finish_reason"] = "length"
    assert chat_response(response, payload, Tokenizer())["choices"][0]["finish_reason"] == "length"
