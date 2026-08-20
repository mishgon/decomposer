"""A GAIA2 subagent whose tools are supplied per episode by runtime context."""

from __future__ import annotations

import json
import os
from typing import Any, TypedDict

import httpx
from langchain.agents import create_agent
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.runtime import Runtime
from pydantic import Field, create_model

HIDDEN_AUI_TOOLS = frozenset(
    {
        "AgentUserInterface__send_message_to_user",
        "AgentUserInterface__get_last_message_from_user",
        "AgentUserInterface__get_last_message_from_agent",
        "AgentUserInterface__get_last_unread_messages",
        "AgentUserInterface__get_all_messages",
    }
)


class EpisodeContext(TypedDict):
    tool_schemas: list[dict[str, Any]]
    broker_url: str
    session_token: str
    policy: str
    scenario_id: str
    run_number: int | None
    notification_cursor: int


def _annotation(schema: dict[str, Any]) -> Any:
    return {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list[Any],
        "object": dict[str, Any],
    }.get(schema.get("type"), Any)


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
    return create_model(f"{name}Arguments", **fields)


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


def _tool_from_schema(schema: dict[str, Any], context: EpisodeContext):
    function = schema["function"]
    name = function["name"]
    if name in HIDDEN_AUI_TOOLS:
        raise ValueError(f"Hidden AUI tool leaked into episode context: {name}")

    async def invoke(**arguments: Any) -> Any:
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(
                f"{context['broker_url']}/tools/{name}/invoke",
                headers={"Authorization": f"Bearer {context['session_token']}"},
                json={"arguments": arguments},
            )
            return _broker_tool_result(response)

    return StructuredTool.from_function(
        coroutine=invoke,
        name=name,
        description=function.get("description") or "",
        args_schema=_arguments_model(name, function.get("parameters") or {}),
        infer_schema=False,
    )


def _model() -> ChatOpenAI:
    thinking = os.environ.get("GAIA2_SUBAGENT_THINKING", "1") == "1"
    extra_body: dict[str, Any] = {
        "top_k": int(os.environ.get("GAIA2_SUBAGENT_TOP_K", "64")),
    }
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
    if extra_body:
        kwargs["extra_body"] = extra_body
    return ChatOpenAI(**kwargs)


async def run_subagent(
    state: MessagesState, runtime: Runtime[EpisodeContext]
) -> dict[str, Any]:
    context = runtime.context
    # Each worker owns an independent journal cursor. Starting at zero makes
    # the initial scenario/user state readable by every parallel worker.
    cursor = 0
    tools = [_tool_from_schema(schema, context) for schema in context["tool_schemas"]]

    async def read_notifications() -> dict[str, Any]:
        nonlocal cursor
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                f"{context['broker_url']}/notifications",
                params={"cursor": cursor, "consumer": f"subagent-{id(state)}"},
                headers={"Authorization": f"Bearer {context['session_token']}"},
            )
            response.raise_for_status()
            result = response.json()
            cursor = int(result["next_cursor"])
            return result

    tools.append(
        StructuredTool.from_function(
            coroutine=read_notifications,
            name="Gaia2Broker__read_notifications",
            description=(
                "Read newly ready ARE notifications using this subagent's independent "
                "cursor. Use after waiting or when the environment may have changed."
            ),
        )
    )
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
