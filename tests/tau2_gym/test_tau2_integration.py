"""Export and scoring against the real tau2 checkout.

tau2 lives in its own venv, so these skip under the project venv. Run them with:

    TAU2_DATA_DIR=$PWD/external/tau2_gym/data \
      ~/decomposer_artifacts_new/venvs/tau2/bin/python -m pytest -q tests/tau2_gym/test_tau2_integration.py

A stratified sample (one task per domain) is scored by default; set
TAU2_FULL_POOL_SCORING=1 to score every task of both pools (about five minutes).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("TAU2_DATA_DIR", str(REPO_ROOT / "external" / "tau2_gym" / "data"))
os.environ.setdefault("LOGURU_LEVEL", "ERROR")
pytest.importorskip("tau2")
sys.path.insert(0, str(REPO_ROOT / "gyms" / "tau2_gym" / "gym_components" / "resources_servers" / "tau2_gym"))
sys.path.insert(0, str(REPO_ROOT / "gyms" / "tau2_gym"))

import tau2_bridge as bridge  # noqa: E402
import tau2_export  # noqa: E402
from tau2.data_model.message import ToolCall, ToolMessage  # noqa: E402
from tau2.training.reward import task_notes_meta  # noqa: E402

from gyms.tau2_gym.task_pools import load_pool  # noqa: E402

pytestmark = pytest.mark.integration


def _sample(pool: str) -> list[tuple[str, str]]:
    tasks, _ = load_pool(pool)
    if os.environ.get("TAU2_FULL_POOL_SCORING") == "1":
        return tasks
    return tau2_export.stratified_subset(tasks, 1)


def _gold_calls(domain: str, task_id: str) -> list[ToolCall]:
    task = bridge.load_task(domain, task_id)
    return [
        ToolCall(id=str(index), name=action.name, arguments=action.arguments, requestor=action.requestor)
        for index, action in enumerate(task.evaluation_criteria.actions or [])
    ]


def _executed(domain: str, task_id: str, calls: list[ToolCall]) -> list[tuple[ToolCall, ToolMessage]]:
    """What the resources server logs: each call executed on the live environment, with its result."""
    environment = bridge.seed_environment(domain, task_id)
    return [(call, environment.get_response(call)) for call in calls]


def _spec(domain: str, task_id: str) -> dict:
    return bridge.tau2_reward._answer_specs_for(domain).get(task_id) or {}


def _gold_answer(domain: str, task_id: str) -> str:
    spec = _spec(domain, task_id)
    if spec.get("type") == "ask":
        clarify = task_notes_meta(bridge.load_task(domain, task_id)).get("clarify") or "which one you mean"
        return f"Before I change anything: could you clarify {clarify}?"
    if spec:
        return f"Done. The answer is {spec['value']}."
    return "Done."


def test_stratified_subset_is_deterministic_and_covers_every_domain() -> None:
    tasks, _ = load_pool("decomposer_train_v2")
    first = tau2_export.stratified_subset(tasks, 2)
    assert first == tau2_export.stratified_subset(list(tasks), 2)
    assert len(first) == 2 * len({domain for domain, _ in tasks})
    assert first == [key for key in tasks if key in set(first)]


def test_export_rows_satisfy_the_gym_and_sft_contracts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "rows.jsonl"
    tau2_export.main(["--pool", "decomposer_eval_v1", "--tasks-per-domain", "1", "--output", str(output)])
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    rows = [json.loads(line) for line in output.read_text().splitlines()]

    assert summary["rows"] == len(rows) == len(summary["domains"])
    assert summary["skipped"] == {}
    for row in rows:
        assert row["agent_ref"] == {"type": "responses_api_agents", "name": "decomposer"}
        # Required on every materialized input by sft/adapters/nemo_gym.py.
        assert row["category"] == row["domain"]
        assert row["environment_name"] == "tau2_gym"
        system, user = row["responses_create_params"]["input"]
        assert system["role"] == "system" and system["content"]
        assert user["role"] == "user" and user["content"]
        assert row["responses_create_params"]["tools"]
        assert all(tool["type"] == "function" for tool in row["responses_create_params"]["tools"])


@pytest.mark.parametrize("pool", ["decomposer_train_v2", "decomposer_eval_v1"])
def test_gold_replay_passes_and_controls_fail(pool: str) -> None:
    failures: list[tuple] = []
    for domain, task_id in _sample(pool):
        gold = _gold_calls(domain, task_id)
        reward, breakdown = bridge.score_trajectory(domain, task_id, gold, _gold_answer(domain, task_id))
        if reward != 1.0:
            failures.append(("gold", domain, task_id, breakdown["terms"], breakdown["details"]))
        if bridge.score_trajectory(domain, task_id, [], None)[0] != 0.0:
            failures.append(("empty", domain, task_id))
        if _spec(domain, task_id).get("type") not in (None, "ask"):
            reward, _ = bridge.score_trajectory(domain, task_id, gold, "Done. The answer is 987654321.")
            if reward != 0.0:
                failures.append(("wrong_answer", domain, task_id))
    assert not failures, failures[:10]


def test_structural_score_is_kept_for_comparability() -> None:
    domain, task_id = "gym_memberships", "hw0_filt_0"
    _, breakdown = bridge.score_trajectory(domain, task_id, _gold_calls(domain, task_id), _gold_answer(domain, task_id))
    assert breakdown["structural"] == 1.0
    assert set(breakdown["structural_breakdown"]) == {"DB", "ENV_ASSERTION", "ACTION", "ACTION_fraction"}
    assert breakdown["binding_terms"] >= 3


def test_forbidden_call_breaks_restraint() -> None:
    """Acting on the ambiguous referent fails the task even when everything else is right."""
    domain, task_id = "addon_provisioning", "hw0_ambi_10"
    assert "forbidden=create_order" in bridge.load_task(domain, task_id).description.notes
    gold = _gold_calls(domain, task_id)
    answer = _gold_answer(domain, task_id)
    assert bridge.score_trajectory(domain, task_id, gold, answer)[0] == 1.0

    guess = ToolCall(id="guess", name="create_order", arguments={}, requestor="assistant")
    reward, breakdown = bridge.score_trajectory(domain, task_id, [*gold, guess], answer)
    assert reward == 0.0
    assert breakdown["terms"]["restraint"] is False
    assert breakdown["forbidden_calls"] == 1


@pytest.mark.parametrize("pool", ["decomposer_train_v2", "decomposer_eval_v1"])
def test_logged_gold_calls_pass_and_controls_fail(pool: str) -> None:
    failures: list[tuple] = []
    for domain, task_id in _sample(pool):
        gold = _gold_calls(domain, task_id)
        reward, breakdown = bridge.score_logged_calls(
            domain, task_id, _executed(domain, task_id, gold), _gold_answer(domain, task_id)
        )
        if (reward != 1.0 or breakdown["scoring"] != "server_log_v1"
                or breakdown["logged_calls"] != len(gold) or not breakdown["log_replays"]):
            failures.append(("gold", domain, task_id, breakdown["terms"], breakdown["details"]))
        if bridge.score_logged_calls(domain, task_id, [], None)[0] != 0.0:
            failures.append(("empty", domain, task_id))
    assert not failures, failures[:10]


def test_calls_the_decomposer_never_reported_are_still_scored() -> None:
    """A dead subagent's forbidden write fails the task although its history was lost."""
    domain, task_id = "addon_provisioning", "hw0_ambi_10"
    gold = _gold_calls(domain, task_id)
    answer = _gold_answer(domain, task_id)
    guess = ToolCall(id="guess", name="create_order", arguments={}, requestor="assistant")

    # What the Decomposer could report: only the gold calls. The old path passes it.
    assert bridge.score_trajectory(domain, task_id, gold, answer)[0] == 1.0
    # What the environment executed: the dead subagent's guess as well.
    reward, breakdown = bridge.score_logged_calls(
        domain, task_id, _executed(domain, task_id, [*gold, guess]), answer
    )
    assert reward == 0.0
    assert breakdown["terms"]["restraint"] is False
    assert breakdown["forbidden_calls"] == 1


def test_a_log_whose_results_do_not_reproduce_fails_closed() -> None:
    domain, task_id = "gym_memberships", "hw0_filt_0"
    executed = _executed(domain, task_id, _gold_calls(domain, task_id))

    assert bridge.score_logged_calls(domain, task_id, list(executed), _gold_answer(domain, task_id))[0] == 1.0
    call, result = executed[0]
    executed[0] = (call, result.model_copy(update={"content": "tampered"}))

    reward, breakdown = bridge.score_logged_calls(domain, task_id, executed, _gold_answer(domain, task_id))

    assert reward == 0.0
    assert breakdown["log_replays"] is False
    assert breakdown["log_replay_error"].startswith("call 0 ")
