from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("are")

from are.simulation.apps.agent_user_interface import AgentUserInterface
from are.simulation.apps.app import App
from are.simulation.apps.system import SystemApp
from are.simulation.environment import Environment, EnvironmentConfig
from are.simulation.notification_system import MessageType
from are.simulation.tool_utils import OperationType, app_tool
from are.simulation.types import (
    CapabilityTag,
    EnvironmentState,
    Event,
    EventTimeComparator,
    EventType,
    OracleEvent,
    event_registered,
)

from gyms.gaia2.broker import ToolStateBroker
from gyms.gaia2.simulated_time import (
    TOOL_CALL_STEP_SECONDS,
    FrozenTurnClock,
    frozen_turn_refusals,
)

GAIA2_REPO = Path(__file__).parents[2] / "external" / "gaia2"


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


class Scenario:
    scenario_id = "scenario-a"
    run_number = 1

    def __init__(self, tools):
        self._tools = tools

    def get_tools(self):
        return self._tools


@pytest.fixture
def world():
    counter, system = Counter(), SystemApp()
    env = Environment(EnvironmentConfig(start_time=1000, duration=600, verbose=False))
    env.register_apps([counter, system])
    env.start()
    broker = ToolStateBroker()
    session = broker.register(
        Scenario([*counter.get_tools(), *system.get_tools()]), env.notification_system
    )
    session.clock = FrozenTurnClock(env)
    try:
        yield env, session
    finally:
        broker.close()
        with session.lock:
            session.clock.release()
        env.stop()
        env.thread.join(timeout=5)


def freeze(session):
    with session.lock:
        session.clock.freeze()


def agent_events(env):
    return sorted(
        (e for e in env.event_log.list_view() if e.event_type == EventType.AGENT),
        key=lambda e: e.event_time,
    )


def test_frozen_clock_ignores_wall_time(world):
    env, session = world
    freeze(session)
    before = env.time_manager.time()

    time.sleep(1.2)

    assert env.state == EnvironmentState.PAUSED
    assert env.time_manager.time() == before


def test_concurrent_tool_calls_get_unique_times_in_call_order(world):
    env, session = world
    freeze(session)
    start = env.time_manager.time()

    def call():
        for _ in range(5):
            session.invoke("Counter__increment", {"amount": 1})

    threads = [threading.Thread(target=call) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    events = agent_events(env)
    times = [event.event_time for event in events]
    assert all(later > earlier for earlier, later in zip(times, times[1:]))
    # The counter's value records the order in which the lock admitted calls.
    assert [event.metadata.return_value for event in events] == list(range(1, 21))
    assert [record["simulated_time"] for record in session.trace] == times
    tool_seconds = sum(record["latency_seconds"] for record in session.trace)
    assert env.time_manager.time() - start == pytest.approx(
        20 * TOOL_CALL_STEP_SECONDS + tool_seconds, abs=1e-6
    )


def test_wait_jumps_the_clock_and_leaves_it_frozen(world):
    env, session = world
    freeze(session)
    before = env.time_manager.time()

    result = session.wait_for_notifications(0, {"timeout": 30}, consumer="worker-1")

    after = env.time_manager.time()
    assert result["native_wait_invoked"] is True
    assert 30 <= after - before < 30.1
    assert env.state == EnvironmentState.PAUSED
    assert env.time_manager.is_paused
    time.sleep(1.2)
    assert env.time_manager.time() == after
    (wait,) = session.waits
    assert wait["consumer"] == "worker-1"
    assert wait["native_wait_invoked"] is True
    assert wait["requested_at"] <= wait["started_at"] <= wait["finished_at"]
    assert wait["simulated_time_after"] - wait["simulated_time_before"] >= 30
    with session.lock:
        summary = session.clock.unfreeze()
    assert summary["waits"] == 1
    assert 30 <= summary["wait_seconds"] < 30.1


def test_wait_past_the_cap_lets_are_end_the_scenario(world):
    env, session = world
    freeze(session)

    session.wait_for_notifications(0, {"timeout": 900})

    assert session.clock.cap_reached
    assert not session.clock.frozen
    env.thread.join(timeout=5)
    assert not env.thread.is_alive()
    journal = session.notifications_after(0)["notifications"]
    assert MessageType.ENVIRONMENT_STOP.value in {entry["type"] for entry in journal}


def test_releasing_after_a_stop_lets_the_parked_loop_exit(world):
    env, session = world
    freeze(session)
    time.sleep(1.2)  # the event loop parks on the pause event

    env.stop()
    time.sleep(0.5)
    # ARE's pause loop does not look at stop_event.
    assert env.thread.is_alive()

    with session.lock:
        session.clock.release()
    env.thread.join(timeout=5)
    assert not env.thread.is_alive()


def test_unfreeze_restores_the_running_clock(world):
    env, session = world
    freeze(session)
    with session.lock:
        session.invoke("Counter__increment", {"amount": 1})
        summary = session.clock.unfreeze()

    assert env.state == EnvironmentState.RUNNING
    assert summary["frozen"] is True
    assert summary["tool_calls"] == 1
    assert summary["cap_reached"] is False
    assert 0 < summary["end"] - summary["start"] < 0.1
    before = env.time_manager.time()
    time.sleep(0.3)
    assert env.time_manager.time() > before + 0.2
    with session.lock:
        assert session.clock.unfreeze() is None


def _scenario(*extra_events, oracle_delay=1.0, comparator=None, **attributes):
    aui = AgentUserInterface()
    task = Event.from_function(
        aui.send_message_to_agent, event_type=EventType.USER, content="Do it."
    )
    task.event_relative_time = 0
    oracle = OracleEvent.from_event(
        Event.from_function(aui.send_message_to_user, content="Done.")
    )
    oracle.event_relative_time = oracle_delay
    oracle.event_time_comparator = comparator
    oracle.dependencies = [task]
    values = {"nb_turns": 1, "tags": (), "env_events_config": None}
    values.update(attributes)
    return SimpleNamespace(events=[task, oracle, *extra_events], **values)


def test_guard_admits_a_static_single_turn_scenario():
    assert frozen_turn_refusals(_scenario()) == []


@pytest.mark.parametrize(
    ("scenario", "reason"),
    [
        (lambda: _scenario(Event.from_function(Counter().increment, amount=1)), "ENV event"),
        (
            lambda: _scenario(
                Event.from_function(
                    AgentUserInterface().send_message_to_agent,
                    event_type=EventType.USER,
                    content="More.",
                )
            ),
            "USER event",
        ),
        (lambda: _scenario(nb_turns=2), "2 turns"),
        (lambda: _scenario(tags=(CapabilityTag.Time,)), "Time capability tag"),
        (lambda: _scenario(env_events_config=object()), "noise"),
        (lambda: _scenario(oracle_delay=30), "time-checked oracle"),
        (
            lambda: _scenario(comparator=EventTimeComparator.LESS_THAN),
            "time-checked oracle",
        ),
    ],
)
def test_guard_refuses_scenarios_with_events_during_a_turn(scenario, reason):
    reasons = frozen_turn_refusals(scenario())

    assert len(reasons) == 1
    assert reason in reasons[0]


def test_ares_litellm_engine_still_reports_no_generation_time():
    # The simple agent's clock stands still during generation only because ARE's
    # engine never reports `completion_duration` (base_agent.py resumes the
    # environment with `metadata.get("completion_duration", 0)`). If this changes,
    # the two agents no longer share the clock rule gyms/gaia2/simulated_time.py
    # assumes.
    engine = GAIA2_REPO / "are/simulation/agents/llm/litellm/litellm_engine.py"
    agent = GAIA2_REPO / "are/simulation/agents/default_agent/base_agent.py"

    assert "completion_duration" not in engine.read_text(encoding="utf-8")
    assert 'metadata.get("completion_duration", 0)' in agent.read_text(
        encoding="utf-8"
    )
