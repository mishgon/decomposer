"""ARE's simulated clock during a Decomposer turn.

ARE's native agent pauses the environment around each model call and resumes it
with the measured generation time (`base_agent.py`, `simulated_generation_time_mode
="measured"`), which ARE's LiteLLM engine reports as zero. Its clock therefore moves
only while tools run and during waits. The Decomposer's manager and workers generate
concurrently in other processes, so the proxy freezes the clock for the whole turn
and the broker advances it only through tool calls: 1 ms before each call, so every
event gets a unique timestamp in call order, and the call's measured duration after
it. `SystemApp__wait_for_notification` jumps the clock as in ARE and is re-frozen
afterwards. Concurrent waits add up instead of overlapping; the broker logs every
wait so the effect can be measured.

All methods run under the broker session lock. The guard admits only scenarios in
which nothing is scheduled to happen during a turn (GAIA2 Execution and Search).
"""

from __future__ import annotations

from typing import Any

from are.simulation.types import (
    CapabilityTag,
    EnvironmentState,
    EventType,
    OracleEvent,
)

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
            and getattr(event, "function_name", lambda: None)() == "send_message_to_agent"
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
