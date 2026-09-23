"""tau2 access layer shared by the resources server and the dataset builder.

Everything that imports ``tau2`` lives here so the FastAPI app stays thin and the
same task/tool/scoring logic is used by ``gyms/tau2_gym/prepare.py`` (through
``tau2_export.py``) and by ``verify``.

Scoring note: the Decomposer hands the verifier a *flattened* trajectory -- every
subagent ``function_call`` in report order plus the final assistant message, and no
``function_call_output`` items (``decomposer_agent/app.py:287-311``). tau2's
``Environment.set_state`` cannot consume that: it requires a ``ToolMessage`` after
every tool call and raises when the replayed output differs from the recorded one
(``environment.py:362-410``). So the predicted database is rebuilt by replaying the
predicted calls into a fresh environment, symmetrically with the gold replay that
``EnvironmentEvaluator`` already does. That is the same shape as
``workplace_assistant``'s ``is_correct``.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from tau2.data_model.message import AssistantMessage, ToolCall
from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment
from tau2.evaluator.evaluator_action import ActionEvaluator
from tau2.training.world import task_world, world_env_constructor
from tau2.utils.utils import DATA_DIR

DEFAULT_LANGUAGE = "en"


@lru_cache(maxsize=None)
def _tasks_by_id(domain: str) -> dict[str, Task]:
    """Load a domain's ``tasks_hard.json`` once, keyed by task id."""
    from tau2.run import load_tasks

    return {task.id: task for task in load_tasks(domain)}


@lru_cache(maxsize=None)
def _raw_tasks_by_id(domain: str) -> dict[str, dict[str, Any]]:
    """Raw task JSON, keyed by id.

    ``world`` and ``answer_spec`` are not fields of ``Task`` and are dropped by
    ``Task.model_validate``; tau2 itself re-reads them from disk
    (``training/world.py:29``, ``training/reward.py:747``). We do the same.
    """
    path = DATA_DIR / "tau2" / "domains" / domain / "tasks_hard.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {entry["id"]: entry for entry in raw}


def load_task(domain: str, task_id: str) -> Task:
    try:
        return _tasks_by_id(domain)[task_id]
    except KeyError:
        raise KeyError(f"Unknown tau2 task {task_id!r} in domain {domain!r}") from None


def task_ids(domain: str) -> list[str]:
    return list(_tasks_by_id(domain))


def world_of(domain: str, task_id: str) -> str | None:
    """World index for a task, from ``description.notes`` (``world=<k>``).

    ``None`` for legacy domains that predate the worlds/ layout (fleet_dispatch,
    insurance_claims, subscription_billing, warehouse_orders). tau2 treats that as
    "unpinned" and falls back to the domain's default ``db.json``
    (``utils.resolve_world_db_path``), so it must be passed through rather than
    coerced -- ``int(None)`` is what previously made those four domains unusable.
    """
    return task_world(load_task(domain, task_id))


def build_environment(
    domain: str, world: str | None, *, language: str = DEFAULT_LANGUAGE
) -> Environment:
    """A fresh environment pinned to ``world``.

    ``world_env_constructor`` sets ``TAU2_WORLD`` under a global lock, so this is
    safe to call from the server's worker threads.
    """
    return world_env_constructor(domain, world)(language=language)


def seed_environment(domain: str, task_id: str, *, language: str = DEFAULT_LANGUAGE) -> Environment:
    """A fresh environment with the task's ``initial_state`` applied."""
    task = load_task(domain, task_id)
    env = build_environment(domain, world_of(domain, task_id), language=language)

    initial = task.initial_state
    env.set_state(
        initialization_data=initial.initialization_data if initial else None,
        initialization_actions=initial.initialization_actions if initial else None,
        message_history=initial.message_history if initial and initial.message_history else [],
    )
    return env


def flat_tool_schemas(
    domain: str, world: str | None = None, *, language: str = DEFAULT_LANGUAGE
) -> list[dict[str, Any]]:
    """Domain tools as Responses-API *flat* function schemas.

    ``Tool.openai_schema`` (``tool.py:177``) emits Chat-Completions nesting; a Gym
    dataset row needs the flat form, which ``NeMoGymSubagentMiddleware`` re-nests
    for the subagent (``decomposer_agent/subagents/graph.py:108-124``).
    """
    env = build_environment(domain, world, language=language)
    schemas = []
    for tool in env.get_tools():
        nested = tool.openai_schema["function"]
        schemas.append(
            {
                "type": "function",
                "name": nested["name"],
                "description": nested.get("description") or "",
                "parameters": nested.get("parameters") or {"type": "object", "properties": {}},
                "strict": False,
            }
        )
    return schemas


def domain_policy(domain: str, *, language: str = DEFAULT_LANGUAGE) -> str:
    return build_environment(domain, None, language=language).get_policy()


def _replay(env: Environment, tool_calls: list[ToolCall]) -> None:
    """Execute calls in order, tolerating failures.

    A call that raises leaves the database untouched, which is exactly how a wrong
    prediction should score. ``EnvironmentEvaluator`` swallows gold-replay errors
    the same way (``evaluator_env.py:95-101``).
    """
    for call in tool_calls:
        try:
            env.get_response(call)
        except Exception:  # noqa: BLE001 - a failed call simply does not mutate the db
            continue


def score_trajectory(
    domain: str,
    task_id: str,
    predicted_tool_calls: list[ToolCall],
    *,
    language: str = DEFAULT_LANGUAGE,
) -> tuple[float, dict[str, Any]]:
    """Structural reward: DB x ACTION x ENV_ASSERTION, each binarised.

    Mirrors ``tau2.training.reward.compute_reward_from_evaluators`` but rebuilds the
    predicted database by replay instead of ``set_state`` -- see the module docstring.
    The deliberately omitted terms (FORMAT / TOOL_VALIDITY / SIGNAL from
    ``reward.py:1093``) describe a single flat agent's own text output; under the
    Decomposer they would measure subagent formatting, not the manager.
    """
    task = load_task(domain, task_id)
    breakdown: dict[str, Any] = {"num_predicted_tool_calls": len(predicted_tool_calls)}

    criteria = task.evaluation_criteria
    if criteria is None:
        return 1.0, {**breakdown, "note": "no evaluation criteria"}

    world = world_of(domain, task_id)
    initial = task.initial_state
    init_data = initial.initialization_data if initial else None
    init_actions = initial.initialization_actions if initial else None
    init_history = initial.message_history if initial and initial.message_history else []

    # Predicted environment: fresh, initial state applied, predicted calls replayed.
    predicted_env = build_environment(domain, world, language=language)
    predicted_env.set_state(init_data, init_actions, [])
    _replay(predicted_env, predicted_tool_calls)

    # Gold environment: fresh, initial state applied, gold actions replayed.
    gold_env = build_environment(domain, world, language=language)
    gold_env.set_state(init_data, init_actions, init_history)
    for action in criteria.actions or []:
        try:
            gold_env.make_tool_call(
                tool_name=action.name, requestor=action.requestor, **action.arguments
            )
        except Exception:  # noqa: BLE001 - matches evaluator_env.py:95-101
            continue

    db_match = predicted_env.get_db_hash() == gold_env.get_db_hash()
    user_db_match = predicted_env.get_user_db_hash() == gold_env.get_user_db_hash()
    db_score = 1.0 if (db_match and user_db_match) else 0.0
    breakdown["DB"] = db_score

    env_assertion_score = 1.0
    for assertion in criteria.env_assertions or []:
        if not predicted_env.run_env_assertion(assertion, raise_assertion_error=False):
            env_assertion_score = 0.0
    breakdown["ENV_ASSERTION"] = env_assertion_score

    # ActionEvaluator reads tool calls straight off assistant messages
    # (evaluator_action.py:82-88), so the flattened view is enough.
    action_info = ActionEvaluator.calculate_reward(
        task=task,
        full_trajectory=[AssistantMessage(role="assistant", content=None, tool_calls=predicted_tool_calls)],
    )
    action_score = 1.0 if action_info.reward == 1.0 else 0.0
    breakdown["ACTION"] = action_score
    breakdown["ACTION_fraction"] = action_info.reward

    reward = db_score * env_assertion_score * action_score
    return reward, breakdown
