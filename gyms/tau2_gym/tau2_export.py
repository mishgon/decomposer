"""Emit NeMo-Gym dataset rows for tau2 tasks.

Runs inside the tau2 venv (tau2 is not installed in the project venv, and its
dependency set is heavy enough that it should not be). ``gyms/tau2_gym/prepare.py``
drives this as a subprocess, mirroring how workplace's ``prepare.py`` re-invokes
itself with the Gym interpreter (``_run_upstream_preparer``, prepare.py:379-406).

Row shape matches ``resources_servers/workplace_assistant/data/example.jsonl``:
``id`` + ``responses_create_params{input, tools, ...}`` + the extra columns that
``seed_session`` and ``verify`` need. The same dict is forwarded to the agent, the
seed-session endpoint and the verifier, so every consumer's fields must be present.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "gym_components" / "resources_servers" / "tau2_gym"))

import tau2_bridge as bridge  # noqa: E402

AGENT_REF = {"type": "responses_api_agents", "name": "decomposer"}


def first_message(task) -> str:
    """The single user turn.

    GAIA2-hard domains run with ScriptedUser (no user-simulator LLM), and the
    scenario's ``first_message`` is the whole request.
    """
    instructions = task.user_scenario.instructions
    if isinstance(instructions, str):
        return instructions
    message = getattr(instructions, "first_message", None)
    if not message:
        raise ValueError(f"Task {task.id} has no first_message; it is not a single-turn task")
    return message


def build_rows(domains: list[str], tasks_per_domain: int | None, language: str) -> list[dict]:
    rows: list[dict] = []
    for domain in domains:
        policy = bridge.domain_policy(domain, language=language)
        ids = bridge.task_ids(domain)
        if tasks_per_domain is not None:
            ids = ids[:tasks_per_domain]

        schemas_by_world: dict[int, list[dict]] = {}
        for task_id in ids:
            task = bridge.load_task(domain, task_id)
            world = bridge.world_of(domain, task_id)
            if world not in schemas_by_world:
                schemas_by_world[world] = bridge.flat_tool_schemas(domain, world, language=language)

            rows.append(
                {
                    "id": len(rows),
                    "agent_ref": AGENT_REF,
                    "domain": domain,
                    "task_id": task_id,
                    "world": world,
                    "responses_create_params": {
                        "input": [
                            {"role": "system", "content": policy},
                            {"role": "user", "content": first_message(task)},
                        ],
                        "tools": schemas_by_world[world],
                        "parallel_tool_calls": True,
                    },
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains", required=True, help="comma-separated tau2 domain names")
    parser.add_argument("--tasks-per-domain", type=int, default=None)
    parser.add_argument("--language", default=bridge.DEFAULT_LANGUAGE)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    rows = build_rows(domains, args.tasks_per_domain, args.language)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(args.output)

    summary = {"rows": len(rows), "domains": domains, "output": str(args.output)}
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
