from __future__ import annotations

import pytest

pytest.importorskip("are")

import json
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any

from are.simulation.apps.app import App
from are.simulation.environment import Environment, EnvironmentConfig
from are.simulation.notification_system import (
    Message,
    MessageType,
    VerboseNotificationSystem,
)
from are.simulation.time_manager import TimeManager
from are.simulation.tool_utils import AppTool, AppToolArg, OperationType, app_tool
from are.simulation.types import event_registered

from gyms.gaia2.broker import ToolStateBroker, app_tool_schema


class Scenario:
    scenario_id = "scenario-a"
    run_number = 1

    def __init__(self, tools):
        self._tools = tools

    def get_tools(self):
        return self._tools


def tool(name, function, *, default=False):
    return AppTool(
        class_name=name.split("__")[0],
        app_name=name.split("__")[0],
        name=name,
        function_description="Do a thing.",
        args=[
            AppToolArg(
                name="value",
                arg_type="int",
                description="A value.",
                has_default=default,
                default=3 if default else None,
                type_obj=int,
            )
        ],
        function=function,
    )


def request(method, url, token, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read())


def notification_system():
    result = VerboseNotificationSystem()
    manager = TimeManager()
    manager.reset(start_time=1000)
    result.initialize(manager)
    return result


def test_schema_conversion_and_hidden_filtering():
    visible = tool("Calendar__write", lambda value: value)
    hidden = tool("AgentUserInterface__send_message_to_user", lambda value: value)
    schema = app_tool_schema(visible)
    assert schema["function"]["parameters"] == {
        "type": "object",
        "properties": {
            "value": {
                "type": "integer",
                "description": "A value.",
            }
        },
        "additionalProperties": False,
        "required": ["value"],
    }

    broker = ToolStateBroker()
    try:
        session = broker.register(Scenario([visible, hidden]), notification_system())
        assert [s["function"]["name"] for s in session.schemas] == ["Calendar__write"]
    finally:
        broker.close()


def test_schema_uses_runtime_types_defaults_and_omits_variadic_kwargs():
    class TypedTools(App):
        @app_tool()
        def send(
            self,
            subject: str,
            recipients: list[str] | None = None,
            min_price: int | float | None = None,
            enabled: bool = False,
            **kwargs: dict[str, Any],
        ) -> str:
            """Send a typed value.

            :param subject: Message subject.
            :param recipients: Optional recipients.
            :param min_price: Optional minimum price.
            :param enabled: Whether the operation is enabled.
            :param kwargs: Additional unsupported options.
            :return: The subject.
            """

            return subject

    app = TypedTools()
    environment = Environment(EnvironmentConfig(duration=1))
    environment.register_apps([app])
    schema = app_tool_schema(app.get_tools()[0])["function"]["parameters"]

    assert schema["required"] == ["subject"]
    assert "kwargs" not in schema["properties"]
    assert schema["properties"]["recipients"] == {
        "anyOf": [
            {"type": "array", "items": {"type": "string"}},
            {"type": "null"},
        ],
        "description": "Optional recipients.",
        "default": None,
    }
    assert schema["properties"]["min_price"]["anyOf"] == [
        {"type": "integer"},
        {"type": "number"},
        {"type": "null"},
    ]
    assert schema["properties"]["enabled"]["default"] is False


def test_token_tool_rejection_and_exactly_once_invocation():
    calls = []
    visible = tool("Calendar__write", lambda value: calls.append(value) or value + 1)
    broker = ToolStateBroker()
    try:
        session = broker.register(Scenario([visible]), notification_system())
        url = f"{broker.base_url}/sessions/{session.session_id}/tools/Calendar__write/invoke"
        assert request("POST", url, session.token, {"arguments": {"value": 4}}) == {
            "result": 5
        }
        assert calls == [4]
        with pytest.raises(urllib.error.HTTPError) as invalid_token:
            request("POST", url, "wrong", {"arguments": {"value": 4}})
        assert invalid_token.value.code == 401
        with pytest.raises(urllib.error.HTTPError) as invalid_tool:
            request(
                "POST",
                f"{broker.base_url}/sessions/{session.session_id}/tools/Nope/invoke",
                session.token,
                {"arguments": {}},
            )
        assert invalid_tool.value.code == 404
        assert calls == [4]
    finally:
        broker.close()


def test_per_session_serialization_and_isolation():
    current = 0
    maximum = 0
    lock = threading.Lock()

    def slow(value):
        nonlocal current, maximum
        with lock:
            current += 1
            maximum = max(maximum, current)
        time.sleep(0.03)
        with lock:
            current -= 1
        return value

    broker = ToolStateBroker()
    try:
        first = broker.register(
            Scenario([tool("App__slow", slow)]), notification_system()
        )
        second_scenario = Scenario([tool("App__only_second", lambda value: value)])
        second_scenario.scenario_id = "scenario-b"
        second = broker.register(second_scenario, notification_system())
        url = f"{broker.base_url}/sessions/{first.session_id}/tools/App__slow/invoke"
        threads = [
            threading.Thread(
                target=request,
                args=("POST", url, first.token, {"arguments": {"value": index}}),
            )
            for index in range(3)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert maximum == 1
        assert "App__only_second" not in first.tools
        assert "App__slow" not in second.tools
    finally:
        broker.close()


def test_parallel_sessions_do_not_share_lock():
    barrier = threading.Barrier(2)

    def parallel(value):
        barrier.wait(timeout=1)
        return value

    broker = ToolStateBroker()
    try:
        first = broker.register(
            Scenario([tool("App__parallel", parallel)]), notification_system()
        )
        second_scenario = Scenario([tool("App__parallel", parallel)])
        second_scenario.scenario_id = "scenario-b"
        second = broker.register(second_scenario, notification_system())
        threads = [
            threading.Thread(
                target=request,
                args=(
                    "POST",
                    f"{broker.base_url}/sessions/{session.session_id}/tools/App__parallel/invoke",
                    session.token,
                    {"arguments": {"value": index}},
                ),
            )
            for index, session in enumerate((first, second))
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert all(not thread.is_alive() for thread in threads)
    finally:
        broker.close()


def test_notification_journal_is_non_destructive_and_has_attachments():
    notifications = notification_system()
    notifications.message_queue.put(
        Message(
            message_type=MessageType.USER_MESSAGE,
            message="hello",
            timestamp=datetime.fromtimestamp(1000, tz=timezone.utc),
        )
    )
    broker = ToolStateBroker()
    try:
        session = broker.register(Scenario([]), notifications)
        first = session.notifications_after(0)
        second = session.notifications_after(0)
        assert first == second
        assert first["notifications"][0]["sequence"] == 1
        assert first["notifications"][0]["attachments"] == []
        assert len(notifications.message_queue.list_view()) == 1
        assert session.notifications_after(1)["notifications"] == []
    finally:
        broker.close()


def test_broker_invocation_creates_one_native_are_event():
    class Counter(App):
        def __init__(self):
            super().__init__()
            self.value = 0

        @app_tool()
        @event_registered(operation_type=OperationType.WRITE)
        def increment(self, amount: int) -> int:
            """Increment the counter.

            :param amount: Amount to add.
            :return: The updated value.
            """
            self.value += amount
            return self.value

    app = Counter()
    environment = Environment(EnvironmentConfig(duration=1))
    environment.register_apps([app])
    scenario = Scenario(app.get_tools())
    broker = ToolStateBroker()
    try:
        session = broker.register(scenario, environment.notification_system)
        assert environment.get_event_log_size() == 0
        assert session.invoke("Counter__increment", {"amount": 2}) == 2
        assert environment.get_event_log_size() == 1
        assert len(session.trace) == 1
    finally:
        broker.close()
