"""The tau2 resources server scores its own call log, not the Decomposer's report.

Needs NeMo-Gym and tau2 together, so it skips under the project venv. Run it with the
tau2 gym component venv that `gym env start` builds:

    TAU2_DATA_DIR=$PWD/external/tau2_gym/data \\
      ~/decomposer_artifacts_new/venvs/tau2-gym-components/<hash>/resources_servers/tau2_gym/.venv/bin/python \\
      -m pytest -q tests/tau2_gym/test_resources_server.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GYM_COMPONENTS = REPO_ROOT / "gyms" / "tau2_gym" / "gym_components"
os.environ.setdefault("TAU2_DATA_DIR", str(REPO_ROOT / "external" / "tau2_gym" / "data"))
os.environ.setdefault("LOGURU_LEVEL", "ERROR")
os.environ.setdefault("NEMO_GYM_EXTRA_ROOTS", str(GYM_COMPONENTS))
pytest.importorskip("tau2")
pytest.importorskip("nemo_gym")
sys.path.insert(0, str(GYM_COMPONENTS))

from fastapi.testclient import TestClient  # noqa: E402
from nemo_gym.server_utils import ServerClient  # noqa: E402
from resources_servers.tau2_gym import tau2_bridge as bridge  # noqa: E402
from resources_servers.tau2_gym.app import (  # noqa: E402
    Tau2GymResourcesServer,
    Tau2GymResourcesServerConfig,
)

pytestmark = pytest.mark.integration

DOMAIN, TASK_ID = "addon_provisioning", "hw0_ambi_10"


def _client() -> tuple[Tau2GymResourcesServer, TestClient]:
    config = Tau2GymResourcesServerConfig(host="127.0.0.1", port=0, entrypoint="", name="tau2_gym")
    server = Tau2GymResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    return server, TestClient(server.setup_webserver())


def _verify_body(final_text: str, reported: list[dict]) -> dict:
    message = {
        "id": "msg_final",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": final_text, "annotations": []}],
    }
    return {
        "domain": DOMAIN,
        "task_id": TASK_ID,
        "responses_create_params": {"input": "task"},
        "response": {
            "id": "resp",
            "created_at": 0.0,
            "model": "manager",
            "object": "response",
            "output": [*reported, message],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        },
    }


def _gold() -> list[tuple[str, dict]]:
    task = bridge.load_task(DOMAIN, TASK_ID)
    return [(action.name, dict(action.arguments)) for action in task.evaluation_criteria.actions or []]


def test_verify_scores_every_executed_call_even_if_none_is_reported() -> None:
    server, client = _client()
    assert client.post("/seed_session", json={"domain": DOMAIN, "task_id": TASK_ID}).status_code == 200
    for name, arguments in _gold():
        assert client.post(f"/{name}", json=arguments).status_code == 200
    # A subagent that died after this call: executed on the server, never reported.
    assert client.post("/create_order", json={}).status_code == 200

    response = client.post("/verify", json=_verify_body("Could you clarify which one you mean?", []))

    assert response.status_code == 200
    body = response.json()
    assert body["reward"] == 0.0
    breakdown = body["breakdown"]
    assert breakdown["scoring"] == "server_log_v1"
    assert breakdown["logged_calls"] == len(_gold()) + 1
    assert breakdown["reported_calls"] == 0
    assert breakdown["unreported_calls"] == len(_gold()) + 1
    assert breakdown["forbidden_calls"] == 1
    # The session is freed once scored.
    assert server.session_id_to_calls == {} and server.session_id_to_environment == {}


def test_sessions_are_isolated_and_reported_calls_are_matched() -> None:
    server, first = _client()
    second = TestClient(first.app)
    for client in (first, second):
        assert client.post("/seed_session", json={"domain": DOMAIN, "task_id": TASK_ID}).status_code == 200
    gold = _gold()
    for name, arguments in gold:
        first.post(f"/{name}", json=arguments)
    second.post("/create_order", json={})

    reported = [
        {"type": "function_call", "name": name, "arguments": json.dumps(arguments),
         "call_id": f"c{index}", "id": f"fc{index}", "status": "completed"}
        for index, (name, arguments) in enumerate(gold)
    ]
    breakdown = first.post("/verify", json=_verify_body("Done.", reported)).json()["breakdown"]

    assert breakdown["logged_calls"] == len(gold)
    assert breakdown["unreported_calls"] == 0
    assert breakdown["reported_not_executed_calls"] == 0
    assert len(server.session_id_to_calls) == 1  # the second session is untouched


def test_verify_without_a_seeded_session_is_an_error() -> None:
    _, client = _client()
    assert client.post("/verify", json=_verify_body("Done.", [])).status_code == 400
