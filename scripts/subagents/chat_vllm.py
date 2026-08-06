from typing import Any

import openai
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI
from pydantic import Field


class ChatVLLM(ChatOpenAI):
    """ChatOpenAI adapter for vLLM's Chat Completions reasoning field."""

    preserve_reasoning: bool = Field(default=False, exclude=True)

    def _create_chat_result(
        self,
        response: dict[str, Any] | openai.BaseModel,
        generation_info: dict[str, Any] | None = None,
    ) -> ChatResult:
        result = super()._create_chat_result(response, generation_info)
        response_dict = response if isinstance(response, dict) else response.model_dump(warnings=False)

        for choice in response_dict.get("choices") or []:
            if choice.get("finish_reason") == "length":
                raise RuntimeError("vLLM exhausted max_completion_tokens before completing its response.")

        if not self.preserve_reasoning:
            return result

        for generation, choice in zip(result.generations, response_dict.get("choices") or [], strict=True):
            message = generation.message
            response_message = choice.get("message") or {}
            reasoning = response_message.get("reasoning")
            if reasoning is None:
                reasoning = response_message.get("reasoning_content")

            if isinstance(message, AIMessage) and isinstance(reasoning, str) and reasoning:
                message.additional_kwargs["reasoning"] = reasoning

        return result

    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        request_messages = payload.get("messages")
        if not isinstance(request_messages, list):
            raise RuntimeError("ChatVLLM requires the Chat Completions API.")

        if not self.preserve_reasoning:
            return payload

        messages = self._convert_input(input_).to_messages()
        for message, request_message in zip(messages, request_messages, strict=True):
            reasoning = message.additional_kwargs.get("reasoning")
            if isinstance(message, AIMessage) and isinstance(reasoning, str) and reasoning:
                request_message["reasoning"] = reasoning

        return payload
