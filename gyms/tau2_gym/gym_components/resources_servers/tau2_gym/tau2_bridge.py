"""tau2 access layer shared by the resources server and the dataset builder.

Everything that imports ``tau2`` lives here so the FastAPI app stays thin and the
same task/tool/scoring logic is used by ``tau2_export.py`` and by ``verify``.

Scoring note: the Decomposer hands the verifier a *flattened* trajectory -- every
subagent ``function_call`` in report order plus the manager's final assistant
message, and no ``function_call_output`` items (``decomposer_agent/app.py:287-311``).
``score_trajectory`` turns that into the trajectory of a *virtual flat agent*: the
calls are replayed into a freshly seeded environment and each recorded result is
appended as a ``ToolMessage``, then the manager's final report becomes the agent's
final message. That trajectory is exactly what tau2's own
``tau2.training.reward.compute_reward`` consumes (its DB and env-assertion terms
replay it through ``Environment.set_state``, which needs a result after every call),
so the Decomposer is graded by the same predicate as tau2's flat agents, minus the
terms that only describe a flat agent's own text.
"""

from __future__ import annotations

import os

# tau2's training runs arm the restraint term with REWARD_FORBIDDEN_PENALTY=1.0
# (training/scripts/run_4b_*.sh). reward.py reads it at import, so it must be set
# before the first tau2.training import in this process.
os.environ.setdefault("REWARD_FORBIDDEN_PENALTY", "1.0")

import json  # noqa: E402
import uuid  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from functools import lru_cache  # noqa: E402
from typing import Any  # noqa: E402

from tau2.data_model.message import AssistantMessage, Message, ToolCall, UserMessage  # noqa: E402
from tau2.data_model.simulation import SimulationRun, TerminationReason  # noqa: E402
from tau2.data_model.tasks import Task  # noqa: E402
from tau2.environment.environment import Environment  # noqa: E402
from tau2.evaluator.evaluator_action import ActionEvaluator  # noqa: E402
from tau2.training import reward as tau2_reward  # noqa: E402
from tau2.training.world import task_world, world_env_constructor  # noqa: E402
from tau2.utils.utils import DATA_DIR  # noqa: E402

DEFAULT_LANGUAGE = "en"

# Terms of tau2's binary predicate (reward.py:1467-1504) that grade the Decomposer.
# tool_validity, signal and closed describe a flat agent's own tool names and text
# channel; under the Decomposer they would measure subagent formatting and the
# harness, not the manager, so they are reported but not binding.
DECOMPOSER_TERMS = ("action", "param", "db", "answer", "env_assertion", "restraint", "side_effects")
NON_BINDING_TERMS = ("tool_validity", "signal", "closed")

if tau2_reward.FORBIDDEN_PENALTY <= 0:
    raise RuntimeError(
        "REWARD_FORBIDDEN_PENALTY is 0, so tau2 would not grade restraint. Something "
        "imported tau2.training.reward before tau2_bridge; unset the variable or set it > 0."
    )


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


def first_message(task: Task) -> str | None:
    """The single user turn, or ``None`` when the task has none.

    The Decomposer never talks to a user simulator: the scenario's ``first_message``
    is the whole request, as under tau2's ScriptedUser.
    """
    instructions = task.user_scenario.instructions
    if isinstance(instructions, str):
        return instructions or None
    return getattr(instructions, "first_message", None) or None


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


def virtual_trajectory(
    domain: str,
    task_id: str,
    predicted_tool_calls: list[ToolCall],
    final_message: str | None,
    *,
    language: str = DEFAULT_LANGUAGE,
) -> tuple[list[Message], set[str]]:
    """The flat-agent trajectory equivalent to a Decomposer rollout, and the tool names.

    One call per assistant message, each followed by the result a freshly seeded
    environment returns for it. Failing calls come back as error ``ToolMessage``s
    (``Environment.get_response``) and leave the database untouched, exactly as they
    did when the subagent made them. tau2 worlds are deterministic, so tau2's own
    replay of this trajectory reproduces every recorded result.
    """
    task = load_task(domain, task_id)
    env = seed_environment(domain, task_id, language=language)
    tool_names = {tool.openai_schema["function"]["name"] for tool in env.get_tools()}

    messages: list[Message] = [UserMessage(role="user", content=first_message(task) or "")]
    for call in predicted_tool_calls:
        messages.append(AssistantMessage(role="assistant", content=None, tool_calls=[call]))
        messages.append(env.get_response(call))
    if final_message:
        messages.append(AssistantMessage(role="assistant", content=final_message))
    return messages, tool_names


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


def structural_score(
    domain: str,
    task_id: str,
    predicted_tool_calls: list[ToolCall],
    *,
    language: str = DEFAULT_LANGUAGE,
) -> tuple[float, dict[str, Any]]:
    """The first tau2-gym reward: DB x ACTION x ENV_ASSERTION, each binarised.

    Kept as a diagnostic so rollouts scored before the full predicate stay
    comparable. It ignores the answer, parameters, restraint and side effects.
    """
    task = load_task(domain, task_id)
    breakdown: dict[str, Any] = {}
    criteria = task.evaluation_criteria
    if criteria is None:
        return 0.0, {"note": "no evaluation criteria"}

    world = world_of(domain, task_id)
    initial = task.initial_state
    init_data = initial.initialization_data if initial else None
    init_actions = initial.initialization_actions if initial else None
    init_history = initial.message_history if initial and initial.message_history else []

    predicted_env = build_environment(domain, world, language=language)
    predicted_env.set_state(init_data, init_actions, [])
    _replay(predicted_env, predicted_tool_calls)

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
    breakdown["DB"] = 1.0 if (db_match and user_db_match) else 0.0

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
    breakdown["ACTION"] = 1.0 if action_info.reward == 1.0 else 0.0
    breakdown["ACTION_fraction"] = action_info.reward
    return breakdown["DB"] * breakdown["ENV_ASSERTION"] * breakdown["ACTION"], breakdown


def score_trajectory(
    domain: str,
    task_id: str,
    predicted_tool_calls: list[ToolCall],
    final_message: str | None,
    *,
    language: str = DEFAULT_LANGUAGE,
) -> tuple[float, dict[str, Any]]:
    """Binary reward: the conjunction of the applicable ``DECOMPOSER_TERMS``.

    Terms come from tau2's ``compute_reward`` with its tri-state semantics: ``None``
    means the term does not apply to this task (no answer_spec, no declared
    prohibition, read-only gold) and does not bind.
    """
    task = load_task(domain, task_id)
    breakdown: dict[str, Any] = {"num_predicted_tool_calls": len(predicted_tool_calls)}
    if task.evaluation_criteria is None:
        return 0.0, {**breakdown, "note": "no evaluation criteria"}

    messages, tool_names = virtual_trajectory(
        domain, task_id, predicted_tool_calls, final_message, language=language
    )
    now = datetime.now(UTC).isoformat()
    simulation = SimulationRun(
        id=uuid.uuid4().hex,
        task_id=task_id,
        start_time=now,
        end_time=now,
        duration=0.0,
        termination_reason=TerminationReason.AGENT_STOP,
        messages=messages,
    )
    _, info = tau2_reward.compute_reward(
        simulation, task, domain, language=language, valid_tool_names=tool_names
    )

    terms = {name: info.terms.get(name) for name in DECOMPOSER_TERMS}
    binding = {name: value for name, value in terms.items() if value is not None}
    reward = 1.0 if binding and all(binding.values()) else 0.0

    structural, structural_breakdown = structural_score(
        domain, task_id, predicted_tool_calls, language=language
    )
    breakdown.update(
        {
            "terms": terms,
            "binding_terms": len(binding),
            "non_binding_terms": {name: info.terms.get(name) for name in NON_BINDING_TERMS},
            "tau2_binary_success": info.binary_success,
            "scores": {
                "action": info.action_score,
                "param": info.param_score,
                "db": info.db_score,
                "answer": info.answer_score,
                "env_assertion": info.env_assertion_score,
            },
            "forbidden_calls": info.forbidden_calls,
            "side_effect_writes": info.side_effect_writes,
            "has_final_message": bool(final_message),
            "details": json.loads(json.dumps(info.details, default=str)),
            "structural": structural,
            "structural_breakdown": structural_breakdown,
        }
    )
    return reward, breakdown
