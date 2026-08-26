"""A GAIA2 subagent whose tools are supplied per episode by runtime context."""

from __future__ import annotations

import json
import os
from types import NoneType
from typing import Any, TypedDict

import httpx
from langchain.agents import create_agent
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.runtime import Runtime
from pydantic import ConfigDict, Field, ValidationError, create_model

HIDDEN_AUI_TOOLS = frozenset(
    {
        "AgentUserInterface__send_message_to_user",
        "AgentUserInterface__get_last_message_from_user",
        "AgentUserInterface__get_last_message_from_agent",
        "AgentUserInterface__get_last_unread_messages",
        "AgentUserInterface__get_all_messages",
    }
)
WAIT_FOR_NOTIFICATION_TOOL = "SystemApp__wait_for_notification"


class EpisodeContext(TypedDict):
    tool_schemas: list[dict[str, Any]]
    broker_url: str
    session_token: str
    policy: str
    scenario_id: str
    run_number: int | None
    notification_cursor: int


def _annotation(schema: dict[str, Any]) -> Any:
    if "anyOf" in schema:
        members = [_annotation(item) for item in schema["anyOf"]]
        if not members:
            return Any
        annotation = members[0]
        for member in members[1:]:
            annotation = annotation | member
        return annotation
    if "enum" in schema:
        from typing import Literal

        return Literal.__getitem__(tuple(schema["enum"]))

    simple = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "null": NoneType,
    }
    schema_type = schema.get("type")
    if schema_type in simple:
        return simple[schema_type]
    if schema_type == "array":
        return list[_annotation(schema.get("items") or {})]
    if schema_type == "object":
        additional = schema.get("additionalProperties", {})
        value_type = _annotation(additional) if isinstance(additional, dict) else Any
        return dict[str, value_type]
    if schema_type is None:
        return Any
    raise TypeError(f"Unsupported Gaia2 JSON schema: {schema!r}")


def _arguments_model(name: str, parameters: dict[str, Any]):
    required = set(parameters.get("required") or [])
    fields: dict[str, Any] = {}
    for key, schema in (parameters.get("properties") or {}).items():
        annotation = _annotation(schema)
        default = ... if key in required else schema.get("default", None)
        fields[key] = (
            annotation,
            Field(default=default, description=schema.get("description")),
        )
    return create_model(
        f"{name}Arguments",
        __config__=ConfigDict(strict=True, extra="forbid"),
        **fields,
    )


def _validation_error(error: ValidationError) -> str:
    return _serialize_tool_result(
        {
            "error": "Invalid tool arguments",
            "details": error.errors(include_url=False),
        }
    )


def _serialize_tool_result(result: Any) -> str:
    """Return broker results as text, not accidental multimodal content blocks.

    LangChain treats a list of dictionaries containing a ``type`` key as
    OpenAI-style multimodal content.  ARE tools such as ``Files__ls`` naturally
    return records whose domain field is ``type: file``.  JSON text preserves
    the exact result while keeping those records inside an ordinary tool
    message.
    """
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False, default=str)


def _broker_tool_result(response: httpx.Response) -> str:
    """Translate model-correctable ARE errors into ordinary tool feedback."""
    if response.status_code in {400, 422}:
        try:
            error = response.json().get("error", response.text)
        except (json.JSONDecodeError, AttributeError):
            error = response.text
        return _serialize_tool_result(
            {"error": error, "http_status": response.status_code}
        )
    # Authentication, session-isolation, and unknown-tool failures remain hard
    # errors; a model must never reason its way around broker security checks.
    response.raise_for_status()
    return _serialize_tool_result(response.json()["result"])


def _tool_from_schema(
    schema: dict[str, Any],
    context: EpisodeContext,
    *,
    coroutine: Any | None = None,
):
    function = schema["function"]
    name = function["name"]
    if name in HIDDEN_AUI_TOOLS:
        raise ValueError(f"Hidden AUI tool leaked into episode context: {name}")

    async def broker_invoke(**arguments: Any) -> Any:
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(
                f"{context['broker_url']}/tools/{name}/invoke",
                headers={"Authorization": f"Bearer {context['session_token']}"},
                json={"arguments": arguments},
            )
            return _broker_tool_result(response)

    return StructuredTool.from_function(
        coroutine=coroutine or broker_invoke,
        name=name,
        description=function.get("description") or "",
        args_schema=_arguments_model(name, function.get("parameters") or {}),
        infer_schema=False,
        handle_validation_error=_validation_error,
    )


def _model() -> ChatOpenAI:
    thinking = os.environ.get("GAIA2_SUBAGENT_THINKING", "1") == "1"
    extra_body: dict[str, Any] = {
        "top_k": int(os.environ.get("GAIA2_SUBAGENT_TOP_K", "64")),
    }
    if min_p := os.environ.get("GAIA2_SUBAGENT_MIN_P"):
        extra_body["min_p"] = float(min_p)
    if repetition_penalty := os.environ.get("GAIA2_SUBAGENT_REPETITION_PENALTY"):
        extra_body["repetition_penalty"] = float(repetition_penalty)
    if not thinking:
        extra_body.update(
            {
                "include_reasoning": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )
    if os.environ.get("GAIA2_SUBAGENT_EXTRA_BODY"):
        extra_body.update(json.loads(os.environ["GAIA2_SUBAGENT_EXTRA_BODY"]))
    kwargs: dict[str, Any] = {
        "model": os.environ["GAIA2_SUBAGENT_MODEL"],
        "base_url": os.environ.get("GAIA2_SUBAGENT_ENDPOINT"),
        "api_key": os.environ.get(
            "GAIA2_SUBAGENT_API_KEY", os.environ.get("OPENAI_API_KEY", "EMPTY")
        ),
        "temperature": float(os.environ.get("GAIA2_SUBAGENT_TEMPERATURE", "1.0")),
        "top_p": float(os.environ.get("GAIA2_SUBAGENT_TOP_P", "0.95")),
        "max_completion_tokens": int(
            os.environ.get("GAIA2_SUBAGENT_MAX_COMPLETION_TOKENS", "4096")
        ),
        "use_responses_api": False,
        "timeout": float(os.environ.get("GAIA2_SUBAGENT_TIMEOUT", "300")),
        "max_retries": int(os.environ.get("GAIA2_SUBAGENT_MAX_RETRIES", "2")),
    }
    if presence_penalty := os.environ.get("GAIA2_SUBAGENT_PRESENCE_PENALTY"):
        kwargs["presence_penalty"] = float(presence_penalty)
    if extra_body:
        kwargs["extra_body"] = extra_body
    return ChatOpenAI(**kwargs)


def _worker_tools(context: EpisodeContext, consumer: str) -> list[StructuredTool]:
    """Build one worker's broker tools around an independent journal cursor."""

    cursor = int(context["notification_cursor"])
    authorization = {"Authorization": f"Bearer {context['session_token']}"}

    def advance_cursor(result: dict[str, Any]) -> None:
        nonlocal cursor
        cursor = max(cursor, int(result["next_cursor"]))

    async def read_notifications() -> str:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                f"{context['broker_url']}/notifications",
                params={"cursor": cursor, "consumer": consumer},
                headers=authorization,
            )
            response.raise_for_status()
            result = response.json()
            advance_cursor(result)
            return _serialize_tool_result(result)

    async def wait_for_notification(**arguments: Any) -> str:
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(
                f"{context['broker_url']}/notifications/wait",
                headers=authorization,
                json={
                    "cursor": cursor,
                    "consumer": consumer,
                    "arguments": arguments,
                },
            )
            if response.status_code in {400, 422}:
                return _broker_tool_result(response)
            response.raise_for_status()
            result = response.json()
            advance_cursor(result)
            return _serialize_tool_result(result)

    tools = []
    for schema in context["tool_schemas"]:
        name = schema["function"]["name"]
        tools.append(
            _tool_from_schema(
                schema,
                context,
                coroutine=(
                    wait_for_notification
                    if name == WAIT_FOR_NOTIFICATION_TOOL
                    else None
                ),
            )
        )
    tools.append(
        StructuredTool.from_function(
            coroutine=read_notifications,
            name="Gaia2Broker__read_notifications",
            description=(
                "Read newly ready ARE notifications using this subagent's independent "
                "cursor. Use when the environment may have changed; the wrapped "
                "SystemApp__wait_for_notification both waits and returns notifications."
            ),
        )
    )
    return tools


async def run_subagent(
    state: MessagesState, runtime: Runtime[EpisodeContext]
) -> dict[str, Any]:
    context = runtime.context
    tools = _worker_tools(context, consumer=f"subagent-{id(state)}")
    system_prompt = os.environ.get(
        "GAIA2_SUBAGENT_SYSTEM_PROMPT",
        (
            "You are a GAIA2 worker. Solve the delegated subtask using the supplied "
            "ARE tools. State is shared between workers and tool calls are serialized. "
            "Never call the user-interface final-response tool; report findings and "
            "actions back to the manager. SystemApp__wait_for_notification is the "
            "canonical way to advance simulated time."
        ),
    )
    agent = create_agent(model=_model(), tools=tools, system_prompt=system_prompt)
    result = await agent.ainvoke({"messages": state["messages"]})
    return {"messages": result["messages"][len(state["messages"]) :]}


builder = StateGraph(MessagesState, context_schema=EpisodeContext)
builder.add_node("subagent", run_subagent)
builder.add_edge(START, "subagent")
builder.add_edge("subagent", END)
gaia2_worker = builder.compile()
