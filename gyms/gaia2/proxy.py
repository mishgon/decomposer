"""Python-3.11-compatible ARE proxy for the remote Decomposer service."""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from are.simulation.agents.agent_execution_result import AgentExecutionResult
from are.simulation.agents.are_simulation_agent import RunnableARESimulationAgent
from are.simulation.apps.agent_user_interface import AgentUserInterface


class RemoteError(RuntimeError):
    pass


class RemoteModelOverflowError(RemoteError):
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
            result = json.loads(response.read())
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


class DecomposerProxyAgent(RunnableARESimulationAgent):
    def __init__(self, broker: Any, env: Any, config: dict[str, Any]) -> None:
        self.broker = broker
        self.env = env
        self.config = config
        self.service_url = config.get("service_url", "http://127.0.0.1:8124").rstrip(
            "/"
        )
        self.request_timeout = float(config.get("request_timeout_seconds", 3500))
        self.poll_interval = float(config.get("notification_poll_seconds", 0.1))
        # Diagnostic escape hatch for measuring premature manager finals.
        # Production evaluations keep the strict default.
        self.allow_uncollected_final = bool(
            config.get("allow_uncollected_final", False)
        )
        self.sidecar_root = config.get("sidecar_root")
        self.gaia2_revision = config.get("gaia2_revision") or os.environ.get(
            "GAIA2_GIT_SHA"
        )
        self.decomposer_revision = config.get("decomposer_revision") or os.environ.get(
            "DECOMPOSER_GIT_SHA"
        )
        self._episode_id: str | None = None
        self._session: Any | None = None
        self._cancel_lock = threading.Lock()
        self._cancelled = False
        self._finished = threading.Event()

    def stop(self) -> None:
        self._cancel_remote()

    def _uncollected_final_is_blocking(self, response: dict[str, Any]) -> bool:
        return bool(response.get("outstanding_subagents")) and not (
            self.allow_uncollected_final
        )

    def _cancel_remote(self) -> None:
        with self._cancel_lock:
            if self._cancelled or self._episode_id is None:
                return
            self._cancelled = True
            try:
                _request(
                    "DELETE",
                    f"{self.service_url}/v1/episodes/{self._episode_id}",
                    timeout=min(10.0, self.request_timeout),
                )
            except RemoteError:
                pass

    def _broker_get(self, suffix: str, timeout: float = 30) -> dict[str, Any]:
        assert self._session is not None
        return _request(
            "GET",
            f"{self.broker.base_url}/sessions/{self._session.session_id}/{suffix}",
            token=self._session.token,
            timeout=timeout,
        )

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
            raise RuntimeError("Decomposer proxy requires an ARE notification system.")
        self._session = self.broker.register(scenario, notification_system)
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
        created = _request(
            "POST",
            f"{self.service_url}/v1/episodes",
            {"context": context},
            timeout=min(30.0, self.request_timeout),
        )
        self._episode_id = str(created["episode_id"])
        cursor = 0
        sidecar: dict[str, Any] = {
            "scenario_id": scenario.scenario_id,
            "run_number": getattr(scenario, "run_number", None),
            "configuration": _redact(self.config),
            "gaia2_revision": self.gaia2_revision,
            "decomposer_revision": self.decomposer_revision,
            "turns": [],
        }
        max_turns = scenario.nb_turns if scenario.nb_turns is not None else 1
        last_text = ""
        try:

            def watch_environment() -> None:
                while not self._finished.wait(self.poll_interval):
                    if self.env.stop_event.is_set():
                        self._cancel_remote()
                        return

            watcher = threading.Thread(
                target=watch_environment,
                name=f"decomposer-stop-{scenario.scenario_id}",
                daemon=True,
            )
            watcher.start()
            for turn_number in range(1, max_turns + 1):
                while True:
                    journal = self._broker_get(
                        "notifications?"
                        + urllib.parse.urlencode(
                            {"cursor": cursor, "consumer": "manager"}
                        )
                    )
                    notifications = journal.get("notifications") or []
                    if (
                        journal.get("environment_stopped")
                        or self.env.stop_event.is_set()
                    ):
                        self._cancel_remote()
                        return AgentExecutionResult(output=last_text, metadata=sidecar)
                    if notifications:
                        break
                    time.sleep(self.poll_interval)

                cursor = int(journal["next_cursor"])
                started = time.monotonic()
                try:
                    response = _request(
                        "POST",
                        f"{self.service_url}/v1/episodes/{self._episode_id}/turn",
                        {
                            "notifications": notifications,
                            "notification_cursor": cursor,
                            "turn_number": turn_number,
                        },
                        timeout=self.request_timeout,
                    )
                    if failure := response.get("failure"):
                        sidecar["turns"].append(
                            {
                                "turn_number": turn_number,
                                "notifications": notifications,
                                "manager": response,
                                "proxy_latency_seconds": time.monotonic() - started,
                            }
                        )
                        self._write_sidecar(scenario, sidecar)
                        raise RemoteModelOverflowError(
                            "Decomposer manager model overflow: "
                            + json.dumps(failure, ensure_ascii=False)
                        )
                except Exception:
                    self._cancel_remote()
                    raise
                last_text = response.get("final_text")
                if not isinstance(last_text, str) or not last_text.strip():
                    self._cancel_remote()
                    raise RuntimeError("Decomposer returned an empty final response.")
                if self._uncollected_final_is_blocking(response):
                    self._cancel_remote()
                    raise RuntimeError(
                        "Decomposer attempted to finish with uncollected subagents."
                    )

                # This is the sole path by which Decomposer ends an ARE turn.
                send_tool(content=last_text)
                sidecar["turns"].append(
                    {
                        "turn_number": turn_number,
                        "notifications": notifications,
                        "manager": response,
                        "proxy_latency_seconds": time.monotonic() - started,
                    }
                )
                sidecar["tool_calls"] = self._session.trace
                self._write_sidecar(scenario, sidecar)
            return AgentExecutionResult(output=last_text, metadata=sidecar)
        finally:
            self._finished.set()
            self._cancel_remote()
            sidecar["tool_calls"] = self._session.trace
            self._write_sidecar(scenario, sidecar)
            self.broker.unregister(self._session.session_id)
