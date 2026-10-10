"""NeMo-Gym resources server exposing a tau2 domain to the Decomposer.

Hosted outside the ``external/Gym`` submodule: ``gym env start`` finds it because
``NEMO_GYM_EXTRA_ROOTS`` points at ``gyms/tau2_gym/gym_components`` and extra roots
are searched before the built-ins (``nemo_gym/__init__.py:52-79``).

Shape follows ``resources_servers/workplace_assistant/app.py``: one session per
rollout holding a live tau2 ``Environment`` and a ``POST /{tool_name}`` catch-all.
Unlike Workplace, ``verify`` does not score the calls the Decomposer reports. Every
subagent calls this server with the rollout's session cookie, so the server logs
each executed call with its result, in execution order, and ``verify`` scores that
log -- as tau2-gym's own environment records its trajectory. Calls of a subagent
that died, and whose history the Decomposer therefore cannot report, still count.
"""

import json
import uuid
from collections import Counter
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.server_utils import SESSION_ID_KEY
from resources_servers.tau2_gym.tau2_bridge import (
    DEFAULT_LANGUAGE,
    score_logged_calls,
    seed_environment,
)

from tau2.data_model.message import ToolCall, ToolMessage

UNREPORTED_NAMES_SHOWN = 10


class Tau2GymResourcesServerConfig(BaseResourcesServerConfig):
    language: str = DEFAULT_LANGUAGE


class Tau2GymToolRequest(BaseModel):
    model_config = ConfigDict(extra="allow")


class Tau2GymToolResponse(BaseModel):
    model_config = ConfigDict(extra="allow")


class Tau2GymSeedSessionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    domain: str
    task_id: str


class Tau2GymVerifyRequest(BaseVerifyRequest):
    domain: str
    task_id: str


class Tau2GymVerifyResponse(BaseVerifyResponse):
    # Declared explicitly: pydantic drops undeclared extras, so without this the
    # rollout row loses which task it came from.
    domain: str
    task_id: str
    category: Optional[str] = None
    environment_name: Optional[str] = None
    breakdown: Dict[str, Any] = Field(default_factory=dict)


class Tau2GymResourcesServer(SimpleResourcesServer):
    config: Tau2GymResourcesServerConfig
    session_id_to_environment: Dict[str, Any] = Field(default_factory=dict)
    # Every call the session's environment executed, with its result, in execution order.
    session_id_to_calls: Dict[str, list] = Field(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/{path}")(self.route_to_python_function)
        return app

    async def seed_session(
        self, request: Request, body: Tau2GymSeedSessionRequest
    ) -> BaseSeedSessionResponse:
        session_id = request.session[SESSION_ID_KEY]
        self.session_id_to_environment[session_id] = seed_environment(
            body.domain, body.task_id, language=self.config.language
        )
        self.session_id_to_calls[session_id] = []
        return BaseSeedSessionResponse()

    async def route_to_python_function(
        self, path: str, body: Tau2GymToolRequest, request: Request
    ) -> Tau2GymToolResponse:
        session_id = request.session[SESSION_ID_KEY]

        if session_id not in self.session_id_to_environment:
            raise HTTPException(
                status_code=400,
                detail="Session not initialized. Please call seed_session first.",
            )

        environment = self.session_id_to_environment[session_id]
        arguments = {
            key: value
            for key, value in body.model_dump(exclude_unset=True).items()
            if value is not None
        }

        call = ToolCall(id=uuid.uuid4().hex, name=path, arguments=arguments, requestor="assistant")
        # Tool failures come back as ordinary output so the subagent can correct
        # itself, matching workplace_assistant/app.py:94-97. get_response already turns
        # tool errors into an error ToolMessage; this only catches failures around it.
        # No await before the call is logged: handlers run one at a time on the event
        # loop, so the log order is the order the environment executed the calls.
        try:
            message = environment.get_response(call)
        except Exception as error:  # noqa: BLE001
            message = ToolMessage(
                id=call.id,
                role="tool",
                content=f"Error executing tool '{path}': {error}",
                requestor="assistant",
                error=True,
            )
        self.session_id_to_calls[session_id].append((call, message))
        return Tau2GymToolResponse(output=message.content)

    async def verify(self, body: Tau2GymVerifyRequest, request: Request) -> Tau2GymVerifyResponse:
        # The session's call log is what gets scored. body.response.output (the
        # Decomposer's flattened view: reported subagent calls, then the final
        # assistant message) supplies the final answer and a diagnostic count.
        session_id = request.session[SESSION_ID_KEY]
        logged = self.session_id_to_calls.pop(session_id, None)
        self.session_id_to_environment.pop(session_id, None)
        if logged is None:
            raise HTTPException(
                status_code=400,
                detail="Session not initialized. Please call seed_session first.",
            )

        reported: list[tuple[str, dict]] = []
        final_message: str | None = None
        for item in body.response.output:
            if item.type == "message" and getattr(item, "role", None) == "assistant":
                final_message = _message_text(item.model_dump())
                continue
            if item.type == "function_call":
                call = item.model_dump()
                reported.append((call["name"], _parse_arguments(call.get("arguments"))))

        reward, breakdown = score_logged_calls(
            body.domain,
            body.task_id,
            logged,
            final_message,
            language=self.config.language,
        )
        breakdown.update(_call_diagnostics(logged, reported))
        return Tau2GymVerifyResponse(**body.model_dump(), reward=reward, breakdown=breakdown)


def _call_key(name: str, arguments: dict) -> tuple[str, str]:
    # The tool route drops None-valued arguments before executing, so compare without them.
    kept = {key: value for key, value in arguments.items() if value is not None}
    return name, json.dumps(kept, sort_keys=True, ensure_ascii=False, default=str)


def _call_diagnostics(
    logged: list[tuple[ToolCall, ToolMessage]], reported: list[tuple[str, dict]]
) -> dict[str, Any]:
    """How the executed calls compare with the calls the Decomposer reported.

    `unreported_calls` are calls the environment executed that the Decomposer's view
    lacks, typically those of a subagent that died before its history was read.
    """
    executed = Counter(_call_key(call.name, call.arguments) for call, _ in logged)
    claimed = Counter(_call_key(name, arguments) for name, arguments in reported)
    unreported = executed - claimed
    return {
        "reported_calls": len(reported),
        "unreported_calls": sum(unreported.values()),
        "unreported_call_names": [name for name, _ in sorted(unreported.elements())][:UNREPORTED_NAMES_SHOWN],
        "reported_not_executed_calls": sum((claimed - executed).values()),
    }


def _message_text(message: dict) -> str | None:
    """Text of a Responses message item: ``content`` is a string or a list of parts."""
    content = message.get("content")
    if isinstance(content, str):
        return content or None
    text = "".join(
        part.get("text") or ""
        for part in content or []
        if isinstance(part, dict) and part.get("type") in ("output_text", "input_text", "text")
    )
    return text or None


def _parse_arguments(arguments: Any) -> dict:
    """Responses function calls carry arguments as a JSON string."""
    if isinstance(arguments, dict):
        return arguments
    if not arguments:
        return {}
    import json

    try:
        parsed = json.loads(arguments)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


if __name__ == "__main__":
    Tau2GymResourcesServer.run_webserver()
