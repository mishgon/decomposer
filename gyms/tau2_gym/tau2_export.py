"""Emit NeMo-Gym dataset rows for a tau2 task pool.

Runs inside the tau2 venv (tau2 is not installed in the project venv, and its
dependency set is heavy enough that it should not be). ``run.py`` drives this as a
subprocess, mirroring how workplace's ``prepare.py`` re-invokes itself with the Gym
interpreter (``_run_upstream_preparer``, prepare.py:379-406).

Row shape matches ``resources_servers/workplace_assistant/data/example.jsonl``:
``id`` + ``responses_create_params{input, tools, ...}`` + the extra columns that
``seed_session`` and ``verify`` need, plus ``category`` / ``environment_name``, which
the ``nemo_gym`` SFT adapter requires on every materialized input. The same dict is
forwarded to the agent, the seed-session endpoint and the verifier, so every
consumer's fields must be present.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
from pathlib import Path

GYM_DIR = Path(__file__).resolve().parent
REPO_ROOT = GYM_DIR.parents[1]
sys.path.insert(0, str(GYM_DIR / "gym_components" / "resources_servers" / "tau2_gym"))
sys.path.insert(0, str(REPO_ROOT))

import tau2_bridge as bridge  # noqa: E402

from gyms.tau2_gym.task_pools import TaskKey, load_pool, resolve_pool  # noqa: E402

AGENT_REF = {"type": "responses_api_agents", "name": "decomposer"}
ENVIRONMENT_NAME = "tau2_gym"


def stratified_subset(tasks: list[TaskKey], per_domain: int) -> list[TaskKey]:
    """``per_domain`` tasks from every domain, chosen by a stable hash.

    Pool ids sort by world and task family (``hw0_ambi_*`` before ``hw1_*``), so
    taking the first ids would skew a smoke run towards one world and one family.
    """
    by_domain: dict[str, list[str]] = collections.defaultdict(list)
    for domain, task_id in tasks:
        by_domain[domain].append(task_id)
    chosen: set[TaskKey] = set()
    for domain, ids in by_domain.items():
        ranked = sorted(ids, key=lambda task_id: hashlib.sha256(f"{domain}/{task_id}".encode()).hexdigest())
        chosen.update((domain, task_id) for task_id in ranked[:per_domain])
    return [key for key in tasks if key in chosen]


def build_rows(tasks: list[TaskKey], language: str) -> tuple[list[dict], dict[str, int]]:
    rows: list[dict] = []
    skipped: collections.Counter[str] = collections.Counter()
    policies: dict[str, str] = {}
    schemas: dict[tuple[str, str | None], list[dict]] = {}
    for domain, task_id in tasks:
        task = bridge.load_task(domain, task_id)
        message = bridge.first_message(task)
        if message is None:
            skipped["no_first_message"] += 1
            continue
        world = bridge.world_of(domain, task_id)
        if domain not in policies:
            policies[domain] = bridge.domain_policy(domain, language=language)
        if (domain, world) not in schemas:
            schemas[(domain, world)] = bridge.flat_tool_schemas(domain, world, language=language)

        rows.append(
            {
                "id": len(rows),
                "agent_ref": AGENT_REF,
                "category": domain,
                "environment_name": ENVIRONMENT_NAME,
                "domain": domain,
                "task_id": task_id,
                "world": world,
                "responses_create_params": {
                    "input": [
                        {"role": "system", "content": policies[domain]},
                        {"role": "user", "content": message},
                    ],
                    "tools": schemas[(domain, world)],
                    "parallel_tool_calls": True,
                },
            }
        )
    return rows, dict(skipped)


def read_tasks_file(path: Path, pool_tasks: list[TaskKey]) -> list[TaskKey]:
    """An explicit task subset (JSON list of [domain, task_id]) in pool order.

    Every task must belong to the pool, so a subset can never reach outside the pool's
    held-out hygiene.
    """
    wanted = {(str(domain), str(task_id)) for domain, task_id in json.loads(path.read_text(encoding="utf-8"))}
    outside = sorted(wanted - set(pool_tasks))
    if outside:
        raise SystemExit(f"{len(outside)} tasks in {path} are not in the pool, e.g. {outside[:3]}")
    return [key for key in pool_tasks if key in wanted]


def subset_sha256(tasks: list[TaskKey]) -> str:
    return hashlib.sha256(json.dumps(sorted(tasks)).encode()).hexdigest()[:12]


def dataset_filename(
    pool_name: str, pool_sha: str, tasks_per_domain: int | None, subset_sha: str | None = None
) -> str:
    suffix = f"-k{tasks_per_domain}" if tasks_per_domain is not None else ""
    if subset_sha is not None:
        suffix += f"-s{subset_sha}"
    return f"{pool_name}-{pool_sha}{suffix}.decomposer.jsonl"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True, help="task pool name or path to its .json")
    parser.add_argument(
        "--tasks-per-domain",
        type=int,
        default=None,
        help="deterministic stratified subsample, for smoke runs and quick evals",
    )
    parser.add_argument("--tasks-file", type=Path, default=None,
                        help="explicit task subset of the pool: JSON list of [domain, task_id]")
    parser.add_argument("--language", default=bridge.DEFAULT_LANGUAGE)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    tasks, meta = load_pool(args.pool)
    if args.tasks_file is not None:
        tasks = read_tasks_file(args.tasks_file, tasks)
    if args.tasks_per_domain is not None:
        if args.tasks_per_domain < 1:
            raise SystemExit("--tasks-per-domain must be at least 1")
        tasks = stratified_subset(tasks, args.tasks_per_domain)
    rows, skipped = build_rows(tasks, args.language)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Per-process temp name: concurrent runs on one pool export the same file, and a
    # shared temp path let one run's replace() steal the other's.
    tmp = args.output.with_suffix(args.output.suffix + f".{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(args.output)

    summary = {
        "rows": len(rows),
        "pool": meta["name"],
        "pool_sha256": meta["sha256"],
        "pool_path": str(resolve_pool(args.pool)),
        "tasks_per_domain": args.tasks_per_domain,
        "tasks_file": str(args.tasks_file) if args.tasks_file else None,
        "domains": sorted({row["domain"] for row in rows}),
        "skipped": skipped,
        "output": str(args.output),
    }
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
