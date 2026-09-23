"""NeMo-Gym resources server exposing a tau2 domain to the Decomposer.

Hosted outside the ``external/Gym`` submodule: ``gym env start`` finds it because
``NEMO_GYM_EXTRA_ROOTS`` points at ``gyms/tau2_gym/gym_components`` and extra roots
are searched before the built-ins (``nemo_gym/__init__.py:52-79``).

Shape follows ``resources_servers/workplace_assistant/app.py``: one session per
rollout holding a live tau2 ``Environment``, a ``POST /{tool_name}`` catch-all, and
a ``verify`` that scores the flattened Decomposer trajectory.
"""

import uuid
from typing import Any, Dict

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
    score_trajectory,
    seed_environment,
)

from tau2.data_model.message import ToolCall


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
    # All three are declared explicitly: pydantic drops undeclared extras, so
    # without this the rollout row loses which task it came from.
    domain: str
    task_id: str
    breakdown: Dict[str, Any] = Field(default_factory=dict)


class Tau2GymResourcesServer(SimpleResourcesServer):
    config: Tau2GymResourcesServerConfig
    session_id_to_environment: Dict[str, Any] = Field(default_factory=dict)

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

        # Tool failures come back as ordinary output so the subagent can correct
        # itself, matching workplace_assistant/app.py:94-97.
        try:
            message = environment.get_response(
                ToolCall(
                    id=uuid.uuid4().hex,
                    name=path,
                    arguments=arguments,
                    requestor="assistant",
                )
            )
        except Exception as error:  # noqa: BLE001
            return Tau2GymToolResponse(output=f"Error executing tool '{path}': {error}")
        return Tau2GymToolResponse(output=message.content)

    async def verify(self, body: Tau2GymVerifyRequest) -> Tau2GymVerifyResponse:
        # response.output is the flattened view built by
        # decomposer_agent/app.py:287-311: every subagent function_call in report
        # order, then the final assistant message.
        predicted_tool_calls: list[ToolCall] = []
        for item in body.response.output:
            if item.type != "function_call":
                continue
            call = item.model_dump()
            predicted_tool_calls.append(
                ToolCall(
                    id=call.get("call_id") or uuid.uuid4().hex,
                    name=call["name"],
                    arguments=_parse_arguments(call.get("arguments")),
                    requestor="assistant",
                )
            )

        reward, breakdown = score_trajectory(
            body.domain,
            body.task_id,
            predicted_tool_calls,
            language=self.config.language,
        )
        return Tau2GymVerifyResponse(**body.model_dump(), reward=reward, breakdown=breakdown)


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
