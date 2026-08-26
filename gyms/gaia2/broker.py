"""Authenticated in-process broker for bound ARE tools and notifications.

This module intentionally uses only the Python standard library and ARE. It is
imported by the benchmark process; LangChain and Decomposer live in separate
service processes.
"""

from __future__ import annotations

import inspect
import json
import secrets
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import NoneType, UnionType
from typing import Any, Literal, Union, get_args, get_origin
from urllib.parse import parse_qs, urlparse

from are.simulation.notification_system import Message, MessageType
from are.simulation.tool_utils import AppTool

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


def _legacy_type(type_name: str) -> Any:
    """Resolve the small legacy type vocabulary used by hand-built AppTools."""

    scalar_types = {
        "Any": Any,
        "str": str,
        "string": str,
        "int": int,
        "integer": int,
        "float": float,
        "number": float,
        "bool": bool,
        "boolean": bool,
        "None": NoneType,
    }
    if type_name in scalar_types:
        return scalar_types[type_name]
    if " | " in type_name:
        members = [_legacy_type(item) for item in type_name.split(" | ")]
        result = members[0]
        for member in members[1:]:
            result = result | member
        return result
    if type_name.startswith("list[") and type_name.endswith("]"):
        return list[_legacy_type(type_name[5:-1])]
    if type_name.startswith("dict[") and type_name.endswith("]"):
        key_name, separator, value_name = type_name[5:-1].partition(", ")
        if not separator:
            raise TypeError(f"Unsupported ARE argument type {type_name!r}")
        return dict[_legacy_type(key_name), _legacy_type(value_name)]
    if type_name == "dict":
        return dict[str, Any]
    raise TypeError(f"Unsupported ARE argument type {type_name!r}")


def _json_schema_for_type(type_obj: Any) -> dict[str, Any]:
    """Convert one runtime Python annotation into strict JSON Schema."""

    if type_obj is Any:
        return {}
    if type_obj in {None, NoneType}:
        return {"type": "null"}
    primitive_types = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
    }
    if type_obj in primitive_types:
        return {"type": primitive_types[type_obj]}
    if type_obj is list:
        return {"type": "array", "items": {}}
    if type_obj is dict:
        return {"type": "object", "additionalProperties": {}}

    origin = get_origin(type_obj)
    arguments = get_args(type_obj)
    if origin in {Union, UnionType}:
        return {"anyOf": [_json_schema_for_type(item) for item in arguments]}
    if origin is list:
        item_type = arguments[0] if arguments else Any
        return {"type": "array", "items": _json_schema_for_type(item_type)}
    if origin is dict:
        key_type, value_type = arguments if arguments else (str, Any)
        if key_type not in {str, Any}:
            raise TypeError(f"JSON object keys must be strings, received {key_type!r}")
        return {
            "type": "object",
            "additionalProperties": _json_schema_for_type(value_type),
        }
    if origin is Literal:
        values = list(arguments)
        schema: dict[str, Any] = {"enum": _jsonable(values)}
        non_null = [item for item in values if item is not None]
        value_types = {type(item) for item in non_null}
        if len(value_types) == 1:
            schema.update(_json_schema_for_type(value_types.pop()))
        return schema
    raise TypeError(f"Unsupported ARE argument annotation {type_obj!r}")


def app_tool_schema(tool: AppTool) -> dict[str, Any]:
    """Build a standards-compliant public schema from an ARE AppTool."""

    parameter_kinds: dict[str, inspect._ParameterKind] = {}
    if tool.function is not None:
        parameter_kinds = {
            name: parameter.kind
            for name, parameter in inspect.signature(tool.function).parameters.items()
        }

    properties: dict[str, Any] = {}
    required: list[str] = []
    for argument in tool.args:
        if parameter_kinds.get(argument.name) in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        type_obj = argument.type_obj
        if type_obj in {None, "Any"}:
            type_obj = _legacy_type(argument.arg_type)
        try:
            property_schema = _json_schema_for_type(type_obj)
        except TypeError as error:
            public_name = tool._public_name or tool.name
            raise TypeError(
                f"Cannot expose Gaia2 tool {public_name!r} argument "
                f"{argument.name!r}: {error}"
            ) from error
        if argument.description:
            property_schema["description"] = argument.description
        if argument.has_default:
            property_schema["default"] = _jsonable(argument.default)
        else:
            required.append(argument.name)
        properties[argument.name] = property_schema

    parameters: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        parameters["required"] = required
    return {
        "type": "function",
        "function": {
            "name": tool._public_name or tool.name,
            "description": tool._public_description or tool.function_description or "",
            "parameters": parameters,
        },
    }


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
                record["error"] = f"{type(exc).__name__}: {exc}"
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
                        self._write(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                        return
                    except Exception as exc:
                        self._write(
                            HTTPStatus.UNPROCESSABLE_ENTITY,
                            {"error": f"{type(exc).__name__}: {exc}"},
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
                    self._write(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    return
                except Exception as exc:
                    self._write(
                        HTTPStatus.UNPROCESSABLE_ENTITY,
                        {"error": f"{type(exc).__name__}: {exc}"},
                    )
                    return
                self._write(HTTPStatus.OK, {"result": result})

        return Handler
