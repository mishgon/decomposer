import html
import json
import re
import uuid
from typing import Any, ClassVar

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class ChatVLLM(ChatOpenAI):
    """ChatOpenAI adapter for vLLM's Chat Completions reasoning field."""

    preserve_reasoning: bool = Field(default=False, exclude=True)
    preserve_reasoning_on_tool_calls_only: bool = Field(default=False, exclude=True)
    parse_qwen_xml_tool_calls: bool = Field(default=False, exclude=True)

    _TOOL_CALL_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"<tool_call>\s*<function=([^>\s]+)>\s*(.*?)\s*</function>\s*</tool_call>",
        re.DOTALL,
    )
    _PARAMETER_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>",
        re.DOTALL,
    )

    @classmethod
    def _parse_qwen_xml(cls, content: str) -> tuple[str, list[dict[str, Any]]]:
        tool_calls: list[dict[str, Any]] = []
        for match in cls._TOOL_CALL_PATTERN.finditer(content):
            arguments: dict[str, Any] = {}
            for parameter in cls._PARAMETER_PATTERN.finditer(match.group(2)):
                value = html.unescape(parameter.group(2).strip())
                if value.startswith(("{", "[")):
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        pass
                arguments[html.unescape(parameter.group(1))] = value
            tool_calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex}",
                    "name": html.unescape(match.group(1)),
                    "args": arguments,
                    "type": "tool_call",
                }
            )

        if "<tool_call>" in content and not tool_calls:
            raise ValueError("Could not parse Qwen XML tool call")

        return cls._TOOL_CALL_PATTERN.sub("", content).strip(), tool_calls

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

        if self.parse_qwen_xml_tool_calls:
            for generation in result.generations:
                message = generation.message
                if not isinstance(message, AIMessage):
                    continue
                content = message.text
                clean_content, tool_calls = self._parse_qwen_xml(content)
                if tool_calls:
                    message.content = clean_content
                    message.tool_calls = tool_calls

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

        # Some hosted vLLM deployments expose Qwen's XML tool template but
        # were started without a server-side tool parser. Asking for
        # tool_choice=none avoids their 400 response while preserving the tool
        # definitions in the prompt; _create_chat_result parses the XML.
        if self.parse_qwen_xml_tool_calls and payload.get("tools"):
            payload["tool_choice"] = "none"

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
                and (
                    not self.preserve_reasoning_on_tool_calls_only
                    or bool(message.tool_calls)
                )
            ):
                request_message["reasoning"] = reasoning

        return payload
