import os
from typing import Any, Literal

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class ChatVLLM(ChatOpenAI):
    """ChatOpenAI adapter for vLLM's Chat Completions reasoning field."""

    preserve_reasoning: bool = Field(default=False, exclude=True)

    def _create_chat_result(
        self,
        response: dict[str, Any] | BaseModel,
        generation_info: dict[str, Any] | None = None,
    ) -> ChatResult:
        result = super()._create_chat_result(response, generation_info)
        response_dict = (
            response
            if isinstance(response, dict)
            else response.model_dump(warnings=False)
        )

        for choice in response_dict.get("choices") or []:
            if choice.get("finish_reason") == "length":
                raise RuntimeError(
                    "vLLM exhausted max_completion_tokens before completing "
                    "its response."
                )

        if not self.preserve_reasoning:
            return result

        for generation, choice in zip(
            result.generations,
            response_dict.get("choices") or [],
            strict=True,
        ):
            message = generation.message
            response_message = choice.get("message") or {}
            reasoning = response_message.get("reasoning")
            if reasoning is None:
                reasoning = response_message.get("reasoning_content")

            if (
                isinstance(message, AIMessage)
                and isinstance(reasoning, str)
                and reasoning
            ):
                message.additional_kwargs["reasoning_content"] = reasoning

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
        for message, request_message in zip(
            messages,
            request_messages,
            strict=True,
        ):
            reasoning = message.additional_kwargs.get("reasoning_content")
            if reasoning is None:
                reasoning = message.additional_kwargs.get("reasoning")
            if (
                isinstance(message, AIMessage)
                and isinstance(reasoning, str)
                and reasoning
            ):
                request_message["reasoning"] = reasoning

        return payload


def create_model(
    model: Literal[
        "qwen_3_5_4b_unlooped_thinking",
        "qwen_3_5_4b_unlooped_non_thinking",
        "qwen_3_8_flash_next_non_thinking",
    ],
) -> BaseChatModel:
    socket = os.environ.get("LLM_PROXY_UNIX_SOCKET")
    sync_client = httpx.Client(
        transport=httpx.HTTPTransport(uds=socket) if socket else None,
        trust_env=False,
    )
    client = httpx.AsyncClient(
        transport=httpx.AsyncHTTPTransport(uds=socket) if socket else None,
        trust_env=False,
    )
    match model:
        case "qwen_3_5_4b_unlooped_thinking":
            return ChatVLLM(
                model="Qwen/Qwen3.5-4B-unlooped",
                base_url="https://lmrouter.2a2i.org/v1",
                api_key=os.environ.get("LLM_PROXY_MASTER_KEY", "EMPTY"),
                temperature=0.6,
                top_p=0.95,
                extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": True}},
                preserve_reasoning=True,
                timeout=600,
                max_retries=2,
                http_client=sync_client,
                http_async_client=client,
                disable_streaming=True,
                use_responses_api=False,
            )
        case "qwen_3_5_4b_unlooped_non_thinking":
            return ChatVLLM(
                model="Qwen/Qwen3.5-4B-unlooped",
                base_url="https://lmrouter.2a2i.org/v1",
                api_key=os.environ.get("LLM_PROXY_MASTER_KEY", "EMPTY"),
                temperature=0.7,
                top_p=0.8,
                extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}},
                preserve_reasoning=False,
                timeout=600,
                max_retries=2,
                http_client=sync_client,
                http_async_client=client,
                disable_streaming=True,
                use_responses_api=False,
            )
        case "qwen_3_8_flash_next_non_thinking":
            return ChatVLLM(
                model="Qwen/Qwen3.8-Flash-Next-NVFP4",
                base_url="https://lmrouter.2a2i.org/v1",
                api_key=os.environ.get("LLM_PROXY_MASTER_KEY", "EMPTY"),
                # https://huggingface.co/Qwen/Qwen3.8-Flash-Next#api-usage
                temperature=0.7,
                top_p=0.8,
                presence_penalty=1.5,
                extra_body={
                    "top_k": 20,
                    "min_p": 0.0,
                    "repetition_penalty": 1.0,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                preserve_reasoning=False,
                timeout=600,
                max_retries=5,
                http_client=sync_client,
                http_async_client=client,
                disable_streaming=True,
                use_responses_api=False,
            )
        case _:
            raise ValueError(f"Unknown model: {model}")
