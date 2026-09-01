"""Authenticated in-process broker for bound ARE tools and notifications.

This module intentionally uses only the Python standard library and ARE. It is
imported by the benchmark process; LangChain and Decomposer live in separate
service processes.
"""

from __future__ import annotations

import json
import secrets
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from are.simulation.notification_system import Message, MessageType
from are.simulation.tool_utils import (
    AppTool,
    app_tool_to_openai_schema,
    sanitize_app_tool_error,
)

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
class BrokerSession:
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
        self, cursor: int, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Return unread entries, advancing native simulated time at most once."""

        with self.lock:
            self.sync_notifications()
            result = self._notifications_after(cursor)
            if result["notifications"]:
                return {**result, "native_wait_invoked": False}

            if WAIT_FOR_NOTIFICATION_TOOL not in self.tools:
                raise KeyError(WAIT_FOR_NOTIFICATION_TOOL)

            # Keep the original bound AppTool invocation so ARE records the genuine
            # SystemApp wait event. RLock makes the nested invocation safe while
            # preserving one atomic drain/check/wait/drain operation.
            self.invoke(WAIT_FOR_NOTIFICATION_TOOL, arguments)
            self.sync_notifications()
            return {
                **self._notifications_after(cursor),
                "native_wait_invoked": True,
            }

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
                self.trace.append(record)


class ToolStateBroker:
    """One loopback HTTP broker shared by one benchmark process."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self._sessions: dict[str, BrokerSession] = {}
        self._sessions_lock = threading.RLock()
        self._server = ThreadingHTTPServer((host, port), self._handler_type())
        self._server.broker = self  # type: ignore[attr-defined]
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="gaia2-tool-broker", daemon=True
        )
        self._thread.start()

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"

    def register(self, scenario: Any, notification_system: Any) -> BrokerSession:
        visible: dict[str, AppTool] = {}
        for tool in scenario.get_tools():
            public_name = tool._public_name or tool.name
            if public_name in HIDDEN_AUI_TOOLS or tool.name in HIDDEN_AUI_TOOLS:
                continue
            visible[public_name] = tool
        session_id = uuid.uuid4().hex
        session = BrokerSession(
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
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def _handler_type(self):
        class Handler(BaseHTTPRequestHandler):
            server_version = "Gaia2Broker/1"

            def log_message(self, format: str, *args: Any) -> None:
                return

            @property
            def broker(self) -> "ToolStateBroker":
                return self.server.broker  # type: ignore[attr-defined]

            def _write(self, status: int, value: Any) -> None:
                data = json.dumps(_jsonable(value), ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _session(self, session_id: str) -> BrokerSession | None:
                with self.broker._sessions_lock:
                    session = self.broker._sessions.get(session_id)
                if session is None:
                    self._write(HTTPStatus.NOT_FOUND, {"error": "unknown session"})
                    return None
                supplied = self.headers.get("Authorization", "")
                if not secrets.compare_digest(supplied, f"Bearer {session.token}"):
                    self._write(HTTPStatus.UNAUTHORIZED, {"error": "invalid token"})
                    return None
                return session

            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path == "/health":
                    self._write(HTTPStatus.OK, {"ok": True})
                    return
                parts = parsed.path.strip("/").split("/")
                if len(parts) < 4 or parts[:2] != ["v1", "sessions"]:
                    self._write(HTTPStatus.NOT_FOUND, {"error": "not found"})
                    return
                session = self._session(parts[2])
                if session is None:
                    return
                if parts[3] == "tools" and len(parts) == 4:
                    self._write(HTTPStatus.OK, {"tools": session.schemas})
                    return
                if parts[3] == "notifications" and len(parts) == 4:
                    query = parse_qs(parsed.query)
                    try:
                        cursor = int(query.get("cursor", ["0"])[0])
                    except ValueError:
                        self._write(HTTPStatus.BAD_REQUEST, {"error": "invalid cursor"})
                        return
                    self._write(HTTPStatus.OK, session.notifications_after(cursor))
                    return
                self._write(HTTPStatus.NOT_FOUND, {"error": "not found"})

            def do_POST(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                parts = parsed.path.strip("/").split("/")
                if (
                    len(parts) == 5
                    and parts[:2] == ["v1", "sessions"]
                    and parts[3:] == ["notifications", "wait"]
                ):
                    session = self._session(parts[2])
                    if session is None:
                        return
                    try:
                        length = int(self.headers.get("Content-Length", "0"))
                        value = json.loads(self.rfile.read(length) or b"{}")
                        cursor = value.get("cursor")
                        arguments = value.get("arguments", {})
                        if not isinstance(cursor, int) or isinstance(cursor, bool):
                            raise ValueError("cursor must be an integer")
                        if not isinstance(arguments, dict):
                            raise ValueError("arguments must be an object")
                        result = session.wait_for_notifications(cursor, arguments)
                    except KeyError:
                        self._write(
                            HTTPStatus.NOT_FOUND,
                            {"error": "wait-for-notification tool is unavailable"},
                        )
                        return
                    except (ValueError, TypeError, json.JSONDecodeError) as exc:
                        self._write(
                            HTTPStatus.BAD_REQUEST,
                            {
                                "error": session.sanitize_error(
                                    WAIT_FOR_NOTIFICATION_TOOL, exc
                                )
                            },
                        )
                        return
                    except Exception as exc:
                        self._write(
                            HTTPStatus.UNPROCESSABLE_ENTITY,
                            {
                                "error": (
                                    f"{type(exc).__name__}: "
                                    f"{session.sanitize_error(WAIT_FOR_NOTIFICATION_TOOL, exc)}"
                                )
                            },
                        )
                        return
                    self._write(HTTPStatus.OK, result)
                    return
                if (
                    len(parts) != 6
                    or parts[:2] != ["v1", "sessions"]
                    or parts[3] != "tools"
                    or parts[5] != "invoke"
                ):
                    self._write(HTTPStatus.NOT_FOUND, {"error": "not found"})
                    return
                session = self._session(parts[2])
                if session is None:
                    return
                tool_name = parts[4]
                if tool_name not in session.tools:
                    self._write(HTTPStatus.NOT_FOUND, {"error": "unknown tool"})
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    value = json.loads(self.rfile.read(length) or b"{}")
                    arguments = value.get("arguments", {})
                    if not isinstance(arguments, dict):
                        raise ValueError("arguments must be an object")
                    result = session.invoke(tool_name, arguments)
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    self._write(
                        HTTPStatus.BAD_REQUEST,
                        {"error": session.sanitize_error(tool_name, exc)},
                    )
                    return
                except Exception as exc:
                    self._write(
                        HTTPStatus.UNPROCESSABLE_ENTITY,
                        {
                            "error": (
                                f"{type(exc).__name__}: "
                                f"{session.sanitize_error(tool_name, exc)}"
                            )
                        },
                    )
                    return
                self._write(HTTPStatus.OK, {"result": result})

        return Handler
