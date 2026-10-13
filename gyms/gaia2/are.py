from __future__ import annotations

import hashlib
import json
import os
import secrets
import socket
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

import uvicorn
from are.simulation.agents.agent_builder import AbstractAgentBuilder
from are.simulation.agents.agent_config_builder import AbstractAgentConfigBuilder
from are.simulation.agents.agent_execution_result import AgentExecutionResult
from are.simulation.agents.are_simulation_agent import RunnableARESimulationAgent
from are.simulation.agents.are_simulation_agent_config import (
    ARESimulationReactAgentConfig,
    ARESimulationReactBaseAgentConfig,
)
from are.simulation.apps.agent_user_interface import AgentUserInterface
from are.simulation.notification_system import Message, MessageType
from are.simulation.tool_utils import (
    AppTool,
    app_tool_to_openai_schema,
    sanitize_app_tool_error,
)
from are.simulation.types import (
    CapabilityTag,
    EnvironmentState,
    EventType,
    OracleEvent,
)
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, StrictInt

from gyms.gaia2.prompts import render_worker_system_prompt

# Advance before every tool call while frozen, so events never share a timestamp.
TOOL_CALL_STEP_SECONDS = 0.001
# ARE's judge checks an oracle event's timing only above this relative delay
# (validation/configs.py check_time_threshold_seconds, event_judge.py:check_time).
JUDGE_CHECK_TIME_THRESHOLD_SECONDS = 1.0


class UnsupportedScenarioError(RuntimeError):
    pass


def frozen_turn_refusals(scenario: Any) -> list[str]:
    """Why a frozen turn would change this scenario's outcome; empty if it cannot.

    Checks the scenario's own events rather than the live queue: ARE notifies the
    agent of the task message before it queues the events that follow it.
    """

    reasons = []
    turns = getattr(scenario, "nb_turns", None) or 1
    if turns != 1:
        reasons.append(f"{turns} turns (turn triggers are CONDITION events)")
    if CapabilityTag.Time in (getattr(scenario, "tags", None) or ()):
        reasons.append("Time capability tag")
    if getattr(scenario, "env_events_config", None) is not None:
        reasons.append("environment noise events (env_events_config)")
    task_messages = 0
    for event in getattr(scenario, "events", None) or ():
        if type(event) is OracleEvent:
            # Inert in agent mode; only the judge's time check could depend on it.
            relative = event.event_relative_time
            if (
                relative is None
                or relative > JUDGE_CHECK_TIME_THRESHOLD_SECONDS
                or event.event_time_comparator is not None
            ):
                reasons.append(f"time-checked oracle event {event.event_id}")
            continue
        is_task_message = (
            event.event_type == EventType.USER
            and not event.dependencies
            and not event.event_relative_time
            and getattr(event, "app_class_name", lambda: None)() == "AgentUserInterface"
            and getattr(event, "function_name", lambda: None)()
            == "send_message_to_agent"
        )
        if is_task_message and task_messages == 0:
            task_messages += 1
            continue
        reasons.append(f"{event.event_type.value} event {event.event_id}")
    return reasons


class FrozenTurnClock:
    """Freezes ARE's clock for a turn; the broker advances it per tool call."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self.frozen = False
        self.cap_reached = False
        self._summary: dict[str, Any] | None = None

    def _time(self) -> float:
        return self.env.time_manager.time()

    def freeze(self) -> None:
        """Pause the environment for a turn, as ARE's agent does for a model call."""

        if self.env.state == EnvironmentState.RUNNING:
            self.env.pause()
        self.frozen = self.env.state == EnvironmentState.PAUSED
        self.cap_reached = False
        self._summary = {
            "frozen": self.frozen,
            "start": self._time(),
            "tool_calls": 0,
            "tool_seconds": 0.0,
            "waits": 0,
            "wait_seconds": 0.0,
        }

    def before_tool(self) -> float:
        """Advance 1 ms and return the time the tool's event will carry."""

        if self.frozen:
            self.env.time_manager.add_offset(TOOL_CALL_STEP_SECONDS)
        return self._time()

    def after_tool(self, elapsed: float, *, started: float, waited: bool) -> None:
        """Re-freeze after a wait, then add the tool's measured duration."""

        if not self.frozen:
            return
        self._hold()
        if self._summary is not None:
            if waited:
                self._summary["waits"] += 1
                self._summary["wait_seconds"] += self._time() - started
            else:
                self._summary["tool_calls"] += 1
                self._summary["tool_seconds"] += elapsed
        self.env.time_manager.add_offset(elapsed)
        duration = getattr(self.env, "duration", None)
        if duration is not None and self.env.time_manager.time_passed() > duration:
            # ARE's loop enforces the cap only while running, so let it end the
            # scenario exactly as it would have without the freeze.
            self.cap_reached = True
            self.release()

    def _hold(self) -> None:
        if self.env.state == EnvironmentState.RUNNING:
            self.env.pause()
        elif (
            self.env.state == EnvironmentState.PAUSED
            and not self.env.time_manager.is_paused
        ):
            # ARE's wait_for_next_notification always resumes the time manager.
            self.env.time_manager.pause()

    def release(self) -> None:
        """Let the environment run again; idempotent, keeps the turn summary."""

        if not self.frozen:
            return
        self.frozen = False
        if self.env.state == EnvironmentState.PAUSED:
            self.env.resume_with_offset(0.0)
        else:
            # env.stop() during the turn: the event loop is still parked on the
            # pause event and must see it cleared to notice the stop.
            self.env.time_manager.resume()
            self.env.pause_event.clear()

    def unfreeze(self) -> dict[str, Any] | None:
        """End the turn: release the clock and return its summary."""

        self.release()
        summary, self._summary = self._summary, None
        if summary is not None:
            summary.update(end=self._time(), cap_reached=self.cap_reached)
        return summary


class ToolRequest(BaseModel):
    arguments: dict[str, Any] = Field(default_factory=dict)


class WaitRequest(ToolRequest):
    cursor: StrictInt
    consumer: str | None = None


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


app_tool_schema = app_tool_to_openai_schema


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    return str(value)


def _message_entry(sequence: int, message: Message) -> dict[str, Any]:
    return {
        "sequence": sequence,
        "type": message.message_type.value,
        "message": message.message,
        "simulated_timestamp": message.timestamp.astimezone(timezone.utc).isoformat(),
        "simulated_unix_timestamp": message.timestamp.timestamp(),
        "attachments": [_jsonable(item) for item in message.attachments],
    }


@dataclass
class ToolSession:
    session_id: str
    token: str
    scenario_id: str
    run_number: int | None
    notification_system: Any
    tools: dict[str, AppTool]
    schemas: list[dict[str, Any]]
    lock: threading.RLock = field(default_factory=threading.RLock)
    journal: list[dict[str, Any]] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    waits: list[dict[str, Any]] = field(default_factory=list)
    # A FrozenTurnClock (gyms/gaia2/are.py) when the proxy freezes turns.
    clock: Any | None = None

    def sanitize_error(self, tool_name: str, error: Exception) -> str:
        tool = self.tools.get(tool_name)
        return sanitize_app_tool_error(tool, error) if tool is not None else str(error)

    def sync_notifications(self) -> None:
        """Move ready native notifications into the persistent broker journal."""

        with self.lock:
            now = datetime.fromtimestamp(
                self.notification_system.get_current_time(), tz=timezone.utc
            )
            for message in self.notification_system.message_queue.get_by_timestamp(now):
                self.journal.append(_message_entry(len(self.journal) + 1, message))

    def _notifications_after(self, cursor: int) -> dict[str, Any]:
        entries = [entry for entry in self.journal if entry["sequence"] > cursor]
        latest_cursor = self.journal[-1]["sequence"] if self.journal else cursor
        return {
            "notifications": entries,
            "next_cursor": max(cursor, latest_cursor),
            "environment_stopped": any(
                entry["type"] == MessageType.ENVIRONMENT_STOP.value for entry in entries
            ),
        }

    def notifications_after(self, cursor: int) -> dict[str, Any]:
        with self.lock:
            self.sync_notifications()
            return self._notifications_after(cursor)

    def wait_for_notifications(
        self,
        cursor: int,
        arguments: dict[str, Any],
        *,
        consumer: str | None = None,
        requested_at: float | None = None,
    ) -> dict[str, Any]:
        """Return unread entries, advancing native simulated time at most once."""

        requested_at = time.time() if requested_at is None else requested_at
        with self.lock:
            # Wall-clock intervals of every wait show offline whether waits
            # from different workers overlapped.
            record: dict[str, Any] = {
                "consumer": consumer,
                "arguments": _jsonable(arguments),
                "requested_at": requested_at,
                "started_at": time.time(),
                "simulated_time_before": self.notification_system.get_current_time(),
                "native_wait_invoked": False,
            }
            self.waits.append(record)
            try:
                self.sync_notifications()
                result = self._notifications_after(cursor)
                if result["notifications"]:
                    return {**result, "native_wait_invoked": False}

                if WAIT_FOR_NOTIFICATION_TOOL not in self.tools:
                    raise KeyError(WAIT_FOR_NOTIFICATION_TOOL)

                # Keep the original bound AppTool invocation so ARE records the
                # genuine SystemApp wait event. RLock makes the nested invocation
                # safe while preserving one atomic drain/check/wait/drain operation.
                record["native_wait_invoked"] = True
                self.invoke(WAIT_FOR_NOTIFICATION_TOOL, arguments)
                self.sync_notifications()
                return {
                    **self._notifications_after(cursor),
                    "native_wait_invoked": True,
                }
            finally:
                record["finished_at"] = time.time()
                record["simulated_time_after"] = (
                    self.notification_system.get_current_time()
                )

    def invoke(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        with self.lock:
            if tool_name not in self.tools:
                raise KeyError(tool_name)
            started = time.monotonic()
            record: dict[str, Any] = {
                "tool": tool_name,
                "arguments": _jsonable(arguments),
                "started_at": time.time(),
            }
            if self.clock is not None:
                record["simulated_time"] = self.clock.before_tool()
            try:
                # Call the original bound AppTool exactly once.  Its decorators remain
                # the only code responsible for mutation and native ARE event logging.
                result = self.tools[tool_name](**arguments)
                record["result"] = _jsonable(result)
                record["ok"] = True
                return result
            except Exception as exc:
                record["ok"] = False
                record["error"] = (
                    f"{type(exc).__name__}: {self.sanitize_error(tool_name, exc)}"
                )
                raise
            finally:
                record["latency_seconds"] = time.monotonic() - started
                if self.clock is not None:
                    self.clock.after_tool(
                        record["latency_seconds"],
                        started=record["simulated_time"],
                        waited=tool_name == WAIT_FOR_NOTIFICATION_TOOL,
                    )
                self.trace.append(record)


class ToolServer:
    """One loopback HTTP broker shared by one benchmark process."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self._sessions: dict[str, ToolSession] = {}
        self._sessions_lock = threading.RLock()
        self._socket = socket.create_server((host, port))
        self._server = uvicorn.Server(
            uvicorn.Config(
                self._app(),
                log_config=None,
                access_log=False,
                lifespan="off",
                timeout_graceful_shutdown=5,
            )
        )
        self._thread = threading.Thread(
            target=self._server.run,
            kwargs={"sockets": [self._socket]},
            name="gaia2-tool-broker",
            daemon=True,
        )
        self._thread.start()
        deadline = time.monotonic() + 5
        while not self._server.started:
            if not self._thread.is_alive() or time.monotonic() >= deadline:
                self.close()
                raise RuntimeError("GAIA2 tool broker failed to start")
            time.sleep(0.01)

    @property
    def base_url(self) -> str:
        host, port = self._socket.getsockname()[:2]
        return f"http://{host}:{port}/v1"

    def register(self, scenario: Any, notification_system: Any) -> ToolSession:
        visible: dict[str, AppTool] = {}
        for tool in scenario.get_tools():
            public_name = tool._public_name or tool.name
            if public_name in HIDDEN_AUI_TOOLS or tool.name in HIDDEN_AUI_TOOLS:
                continue
            visible[public_name] = tool
        session_id = uuid.uuid4().hex
        session = ToolSession(
            session_id=session_id,
            token=secrets.token_urlsafe(32),
            scenario_id=scenario.scenario_id,
            run_number=getattr(scenario, "run_number", None),
            notification_system=notification_system,
            tools=visible,
            schemas=[app_tool_schema(tool) for tool in visible.values()],
        )
        with self._sessions_lock:
            self._sessions[session_id] = session
        return session

    def unregister(self, session_id: str) -> None:
        with self._sessions_lock:
            self._sessions.pop(session_id, None)

    def close(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)
        self._socket.close()
        if self._thread.is_alive():
            raise RuntimeError("GAIA2 tool broker failed to stop")

    def _app(self) -> FastAPI:
        app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)

        @app.exception_handler(HTTPException)
        async def http_error(request, error):
            return JSONResponse({"error": error.detail}, status_code=error.status_code)

        @app.exception_handler(RequestValidationError)
        async def validation_error(request, error):
            return JSONResponse(
                {"error": "; ".join(item["msg"] for item in error.errors())},
                status_code=400,
            )

        def session(
            session_id: str, authorization: Annotated[str, Header()] = ""
        ) -> ToolSession:
            with self._sessions_lock:
                result = self._sessions.get(session_id)
            if result is None:
                raise HTTPException(404, "unknown session")
            if not secrets.compare_digest(authorization, f"Bearer {result.token}"):
                raise HTTPException(401, "invalid token")
            return result

        @app.get("/health")
        def health():
            return {"ok": True}

        @app.get("/v1/sessions/{session_id}/tools")
        def tools(current: ToolSession = Depends(session)):
            return {"tools": current.schemas}

        @app.get("/v1/sessions/{session_id}/notifications")
        def notifications(cursor: int = 0, current: ToolSession = Depends(session)):
            return current.notifications_after(cursor)

        @app.post("/v1/sessions/{session_id}/notifications/wait")
        def wait(body: WaitRequest, current: ToolSession = Depends(session)):
            try:
                return current.wait_for_notifications(
                    body.cursor,
                    body.arguments,
                    consumer=body.consumer,
                )
            except KeyError:
                raise HTTPException(404, "wait-for-notification tool is unavailable")
            except (ValueError, TypeError) as error:
                raise HTTPException(
                    400, current.sanitize_error(WAIT_FOR_NOTIFICATION_TOOL, error)
                )
            except Exception as error:
                raise HTTPException(
                    422,
                    f"{type(error).__name__}: {current.sanitize_error(WAIT_FOR_NOTIFICATION_TOOL, error)}",
                )

        @app.post("/v1/sessions/{session_id}/tools/{tool_name}/invoke")
        def invoke(
            tool_name: str, body: ToolRequest, current: ToolSession = Depends(session)
        ):
            if tool_name not in current.tools:
                raise HTTPException(404, "unknown tool")
            try:
                return {"result": _jsonable(current.invoke(tool_name, body.arguments))}
            except (ValueError, TypeError) as error:
                raise HTTPException(400, current.sanitize_error(tool_name, error))
            except Exception as error:
                raise HTTPException(
                    422,
                    f"{type(error).__name__}: {current.sanitize_error(tool_name, error)}",
                )

        return app


TERMINAL_STATUSES = frozenset({"responded", "error", "timeout", "interrupted"})

ARE_TASK_TEMPLATE = "[TASK]: \n{content}\n"
ARE_ENVIRONMENT_NOTIFICATIONS_TEMPLATE = (
    "Environment notifications updates:\n***\n{content}\n***\n"
)
ARE_NOTIFICATION_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def _usage(messages: list[Any]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for message in messages:
        metadata = message.get("usage_metadata") or {}
        for key, value in metadata.items():
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
    return totals


def _visible_message_text(message: dict[str, Any]) -> str:
    """Extract user-visible text without leaking Responses API reasoning blocks."""

    if isinstance(message["content"], str):
        return message["content"]
    parts: list[str] = []
    for block in message["content"]:
        if isinstance(block, str):
            parts.append(block)
        elif not isinstance(block, Mapping):
            continue
        elif block.get("type") in {"text", "output_text"} and isinstance(
            block.get("text"), str
        ):
            parts.append(block["text"])
        elif block.get("type") == "refusal" and isinstance(block.get("refusal"), str):
            parts.append(block["refusal"])
    return "".join(parts)


def _public_context(context: dict[str, Any]) -> dict[str, Any]:
    public = {
        "tool_schemas": context["tool_schemas"],
        "broker_url": context["broker_url"],
        "session_token": "<redacted>",
        "policy": context["policy"],
        "scenario_id": context["scenario_id"],
        "run_number": context["run_number"],
        "notification_cursor": context["notification_cursor"],
    }
    worker_system_prompt = context.get("worker_system_prompt")
    if worker_system_prompt is not None:
        public["worker_system_prompt_sha256"] = hashlib.sha256(
            worker_system_prompt.encode("utf-8")
        ).hexdigest()
    return public


def _agent_summary(state: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """One entry per agent run, and the runs whose response the manager never received.

    Agents persist across runs (and turns), so the type lives on the agent,
    not on the run.
    """
    agents = state.get("agents") or {}
    summaries: list[dict[str, Any]] = []
    outstanding: list[str] = []
    for run_id, run in (state.get("agent_runs") or {}).items():
        agent = agents.get(run.get("agent_id")) or {}
        item = {
            "agent_run_id": run_id,
            "agent_id": run.get("agent_id"),
            "agent_type_id": agent.get("agent_type_id"),
            "status": run.get("status"),
            "prompt": run.get("prompt"),
            "response": run.get("response"),
            "error": run.get("error"),
            "tool_calls": run.get("tool_calls", []),
            "response_sequence_number": run.get("response_sequence_number"),
        }
        summaries.append(item)
        if (
            run.get("status") not in TERMINAL_STATUSES
            or run.get("response_sequence_number") is None
        ):
            outstanding.append(run_id)
    return summaries, outstanding


def _turn_messages(notifications: list[dict[str, Any]]) -> list[dict[str, str]]:
    """The manager's input for one turn: the user-role text ARE gives its own agent.

    ARE's agent runs `run(task=...)` on every wake-up, logging the new user
    messages as one task (`are_simulation_main.py:389-414, 482-486`), and injects
    environment notifications as a separate user-role update (`steps/are_simulation.py`).
    Nothing else is added, so the Decomposer and the native baseline read the same
    turn. Other notification types (e.g. the environment stop) are not rendered.
    """
    user_messages = [
        str(notification.get("message", ""))
        for notification in notifications
        if notification.get("type") == "USER_MESSAGE"
    ]
    # ARE hands attachments to its agent as images; the manager gets them as
    # JSON text. Execution and Search user messages carry none.
    attachments = [
        attachment
        for notification in notifications
        if notification.get("type") == "USER_MESSAGE"
        for attachment in notification.get("attachments") or []
    ]
    content = "\n".join(user_messages)
    if attachments:
        content += "\nAttachments: " + json.dumps(attachments, ensure_ascii=False)
    messages = [{"role": "user", "content": ARE_TASK_TEMPLATE.format(content=content)}]

    environment_notifications = [
        "["
        + datetime.fromisoformat(notification["simulated_timestamp"]).strftime(
            ARE_NOTIFICATION_TIME_FORMAT
        )
        + "] "
        + str(notification.get("message", ""))
        for notification in notifications
        if notification.get("type") == "ENVIRONMENT_NOTIFICATION"
    ]
    if environment_notifications:
        messages.append(
            {
                "role": "user",
                "content": ARE_ENVIRONMENT_NOTIFICATIONS_TEMPLATE.format(
                    content="\n".join(environment_notifications)
                ),
            }
        )
    return messages


def turn_result(
    state: dict[str, Any], context: dict[str, Any], elapsed: float
) -> dict[str, Any]:
    messages = state.get("messages") or []
    if not messages or messages[-1].get("type") != "ai":
        raise RuntimeError("Agent produced no final AI message")
    text = _visible_message_text(messages[-1])
    if not text.strip():
        raise RuntimeError("Agent produced no visible final text")
    agents, outstanding = _agent_summary(state)
    return {
        "final_text": text,
        "subagent_states": agents,
        "outstanding_subagents": outstanding,
        "usage": _usage(messages),
        "timing": {"elapsed_seconds": elapsed},
        "trace": {
            "manager_messages": [
                {"type": message["type"], "data": message} for message in messages
            ],
            "runtime_context": _public_context(context),
            "notification_cursor": context["notification_cursor"],
            "turn_number": 1,
        },
    }


class RemoteError(RuntimeError):
    pass


def _request(
    method: str,
    url: str,
    value: dict[str, Any] | None = None,
    *,
    token: str | None = None,
    timeout: float = 30,
) -> dict[str, Any]:
    data = None if value is None else json.dumps(value).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            result = json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RemoteError(f"{method} {url} returned HTTP {exc.code}: {body}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RemoteError(f"{method} {url} failed: {exc}") from exc
    if not isinstance(result, dict):
        raise RemoteError(f"{method} {url} returned a non-object response")
    return result


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                "<redacted>"
                if key.lower() in {"token", "secret", "password", "api_key"}
                or key.lower().endswith(("_token", "_secret", "_password", "_api_key"))
                else _redact(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


class AgentServerAgent(RunnableARESimulationAgent):
    def __init__(self, broker: Any, env: Any, config: dict[str, Any]) -> None:
        self.broker = broker
        self.env = env
        self.config = config
        self.server_url = config["agent_server_url"].rstrip("/")
        self.request_timeout = float(config.get("request_timeout_seconds", 3500))
        self.poll_interval = float(config.get("notification_poll_seconds", 0.1))
        self.sidecar_root = config.get("sidecar_root")
        self.gaia2_revision = config["gaia2_revision"]
        self.decomposer_revision = config["decomposer_revision"]
        self._thread_id: str | None = None
        self._run_id: str | None = None
        self._last_state: dict[str, Any] = {}
        self._session: Any | None = None
        self._cancel_lock = threading.Lock()
        self._cancelled = False
        self._finished = threading.Event()

    def stop(self) -> None:
        self._cancel_remote()

    def _cancel_remote(self) -> None:
        with self._cancel_lock:
            if self._cancelled:
                return
            self._cancelled = True
            if self._thread_id is None:
                return
            thread_url = f"{self.server_url}/threads/{self._thread_id}"
            if self._run_id is not None:
                try:
                    _request(
                        "POST",
                        f"{thread_url}/runs/{self._run_id}/cancel?wait=1",
                        timeout=10,
                    )
                except RemoteError:
                    pass
            try:
                state = (
                    _request("GET", f"{thread_url}/state", timeout=10)["values"] or {}
                )
                self._last_state = state
                agents = state.get("agents") or {}
                for run in (state.get("agent_runs") or {}).values():
                    if run.get("status") in TERMINAL_STATUSES:
                        continue
                    agent = agents[run["agent_id"]]
                    try:
                        _request(
                            "POST",
                            f"{self.server_url}/threads/{agent['thread_id']}/runs/{run['run_id']}/cancel?wait=1",
                            timeout=10,
                        )
                    except RemoteError:
                        pass
                for agent in agents.values():
                    try:
                        _request(
                            "DELETE",
                            f"{self.server_url}/threads/{agent['thread_id']}",
                            timeout=10,
                        )
                    except RemoteError:
                        pass
            except RemoteError:
                pass
            finally:
                try:
                    _request("DELETE", thread_url, timeout=10)
                except RemoteError:
                    pass

    def _run_turn(
        self, notifications: list[dict[str, Any]], context: dict[str, Any]
    ) -> dict[str, Any]:
        thread_url = f"{self.server_url}/threads/{self._thread_id}"
        # Serialize creation with cancellation so a late create cannot escape cleanup.
        with self._cancel_lock:
            if self._cancelled:
                raise RemoteError("Episode cancelled")
            run = _request(
                "POST",
                f"{thread_url}/runs",
                {
                    "assistant_id": self.config["assistant_id"],
                    "input": {"messages": _turn_messages(notifications)},
                    "context": context,
                    "config": {"recursion_limit": 10000},
                    "multitask_strategy": "reject",
                },
            )
            self._run_id = run["run_id"]
        deadline = time.monotonic() + self.request_timeout
        while run["status"] in {"pending", "running"}:
            if time.monotonic() >= deadline:
                raise TimeoutError("Agent run exceeded its request deadline")
            if self._cancelled:
                raise RemoteError("Episode cancelled")
            time.sleep(self.poll_interval)
            run = _request("GET", f"{thread_url}/runs/{self._run_id}")
        snapshot = _request("GET", f"{thread_url}/state")
        self._last_state = snapshot["values"] or {}
        if run["status"] != "success":
            raise RemoteError(f"Agent run {run['status']}: {snapshot.get('tasks', [])}")
        return snapshot["values"]

    def _unfreeze(self, clock: FrozenTurnClock) -> dict[str, Any] | None:
        with self._session.lock:
            return clock.unfreeze()

    def _release_clock(self, clock: FrozenTurnClock) -> None:
        with self._session.lock:
            clock.release()

    def _write_sidecar(self, scenario: Any, payload: dict[str, Any]) -> None:
        root = self.sidecar_root
        if root is None:
            output_dir = self.config.get("output_dir")
            if output_dir:
                root = str(Path(output_dir) / "decomposer_sidecars")
        if root is None:
            return
        path = Path(root)
        path.mkdir(parents=True, exist_ok=True)
        run = getattr(scenario, "run_number", None) or 0
        target = path / f"{scenario.scenario_id}__run{run}.json"
        temporary = target.with_suffix(f".tmp.{os.getpid()}")
        temporary.write_text(
            json.dumps(_redact(payload), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(temporary, target)

    def run_scenario(
        self,
        scenario: Any,
        notification_system: Any | None,
        initial_agent_logs: list[Any] | None = None,
    ) -> AgentExecutionResult:
        if notification_system is None:
            raise RuntimeError(
                "Agent Server integration requires an ARE notification system."
            )
        if reasons := frozen_turn_refusals(scenario):
            raise UnsupportedScenarioError(
                f"simulated_time=frozen_turn cannot run {scenario.scenario_id}: "
                + "; ".join(reasons)
            )
        self._session = self.broker.register(scenario, notification_system)
        # Generation is free, as for ARE's native agent: the clock is frozen for
        # each turn and moves only through tool calls (gyms/gaia2/are.py).
        clock = FrozenTurnClock(self.env)
        self._session.clock = clock
        aui = next(
            app for app in scenario.apps or [] if isinstance(app, AgentUserInterface)
        )
        aui.wait_for_user_response = False
        send_tool = next(
            tool
            for tool in scenario.get_tools()
            if tool.name == "AgentUserInterface__send_message_to_user"
        )
        context = {
            "tool_schemas": self._session.schemas,
            "broker_url": f"{self.broker.base_url}/sessions/{self._session.session_id}",
            "session_token": self._session.token,
            "policy": self.config.get("policy", "shared_serialized"),
            "scenario_id": scenario.scenario_id,
            "run_number": getattr(scenario, "run_number", None),
            "notification_cursor": 0,
        }
        worker_system_prompt = render_worker_system_prompt(scenario)
        context["worker_system_prompt"] = worker_system_prompt
        worker_system_prompt_sha256 = hashlib.sha256(
            worker_system_prompt.encode("utf-8")
        ).hexdigest()
        sidecar: dict[str, Any] = {
            "scenario_id": scenario.scenario_id,
            "run_number": getattr(scenario, "run_number", None),
            "configuration": _redact(self.config),
            "worker_system_prompt_sha256": worker_system_prompt_sha256,
            "simulated_time": "frozen_turn",
            "gaia2_revision": self.gaia2_revision,
            "decomposer_revision": self.decomposer_revision,
            "turns": [],
        }
        try:
            with self._cancel_lock:
                if self._cancelled or self.env.stop_event.is_set():
                    raise RemoteError("Episode cancelled")
                created = _request("POST", f"{self.server_url}/threads", {})
                self._thread_id = created["thread_id"]

            def watch_environment() -> None:
                while not self._finished.wait(self.poll_interval):
                    if self.env.stop_event.is_set():
                        # A stopped environment's event loop stays parked while
                        # the turn holds the clock frozen.
                        self._release_clock(clock)
                        self._cancel_remote()
                        return

            watcher = threading.Thread(
                target=watch_environment,
                name=f"decomposer-stop-{scenario.scenario_id}",
                daemon=True,
            )
            watcher.start()
            while True:
                journal = self._session.notifications_after(0)
                notifications = journal.get("notifications") or []
                if journal.get("environment_stopped") or self.env.stop_event.is_set():
                    return AgentExecutionResult(output="", metadata=sidecar)
                if notifications:
                    break
                time.sleep(self.poll_interval)

            context["notification_cursor"] = int(journal["next_cursor"])
            started = time.monotonic()
            clock_record: dict[str, Any] = {}
            response: dict[str, Any] = {}
            with self._session.lock:
                clock.freeze()
            try:
                try:
                    state = self._run_turn(notifications, context)
                    response = turn_result(state, context, time.monotonic() - started)
                except Exception as error:
                    self._cancel_remote()
                    messages = self._last_state.get("messages") or []
                    agents, outstanding = _agent_summary(self._last_state)
                    response = {
                        "failure": {
                            "error": str(error),
                            "traceback": traceback.format_exc(),
                        },
                        "subagent_states": agents,
                        "outstanding_subagents": outstanding,
                        "usage": _usage(messages),
                        "timing": {"elapsed_seconds": time.monotonic() - started},
                        "trace": {
                            "manager_messages": [
                                {"type": message["type"], "data": message}
                                for message in messages
                            ],
                            "runtime_context": _public_context(context),
                            "notification_cursor": context["notification_cursor"],
                            "turn_number": 1,
                        },
                    }
                    raise
                final_text = response.get("final_text")
                if not isinstance(final_text, str) or not final_text.strip():
                    self._cancel_remote()
                    raise RuntimeError("Agent returned an empty final response.")
                if response.get("outstanding_subagents"):
                    self._cancel_remote()
                    raise RuntimeError(
                        "Decomposer attempted to finish with uncollected agents."
                    )

                # This is the sole path by which Decomposer ends an ARE turn.
                # Under the session lock the final message is stamped after
                # every worker tool call, then the environment runs again.
                with self._session.lock:
                    clock.before_tool()
                    send_tool(content=final_text)
                    clock_record["simulated_clock"] = self._unfreeze(clock)
            finally:
                summary = self._unfreeze(clock)
                if summary is not None:
                    clock_record["simulated_clock"] = summary
                sidecar["turns"].append(
                    {
                        "turn_number": 1,
                        "notifications": notifications,
                        "manager": response,
                        "proxy_latency_seconds": time.monotonic() - started,
                        **clock_record,
                    }
                )
                sidecar["tool_calls"] = self._session.trace
                sidecar["waits"] = self._session.waits
                self._write_sidecar(scenario, sidecar)
            return AgentExecutionResult(output=final_text, metadata=sidecar)
        finally:
            self._finished.set()
            self._unfreeze(clock)
            self._cancel_remote()
            sidecar["tool_calls"] = self._session.trace
            sidecar["waits"] = self._session.waits
            self._write_sidecar(scenario, sidecar)
            self.broker.unregister(self._session.session_id)


class AgentConfigBuilder(AbstractAgentConfigBuilder):
    def build(self, agent_name: str):
        if agent_name != "agent_server":
            raise ValueError(
                f"Decomposer plugin does not provide agent {agent_name!r}."
            )
        return ARESimulationReactAgentConfig(
            agent_name=agent_name,
            base_agent_config=ARESimulationReactBaseAgentConfig(),
        )


class AgentBuilder(AbstractAgentBuilder):
    def __init__(self, config: dict[str, Any], broker: ToolServer) -> None:
        self.config = config
        self.broker = broker

    def list_agents(self) -> list[str]:
        return ["agent_server"]

    def build(self, agent_config, env=None, mock_responses=None):
        if agent_config.get_agent_name() != "agent_server":
            raise ValueError(
                f"Decomposer plugin does not provide {agent_config.get_agent_name()!r}."
            )
        if env is None:
            raise ValueError("Agent Server integration requires an Environment.")
        return AgentServerAgent(self.broker, env, self.config)


def create_plugin(config: dict[str, Any]):
    """Return ARE's existing config/agent builder extension points."""

    broker = ToolServer(
        host=str(config.get("broker_host", "127.0.0.1")),
        port=int(config.get("broker_port", 0)),
    )
    return AgentConfigBuilder(), AgentBuilder(config, broker)
