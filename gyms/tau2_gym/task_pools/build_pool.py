#!/usr/bin/env python3
"""Build versioned Decomposer task pools from tau2's training splits.

Every pool comes from a recipe in ``RECIPES``, so it can be rebuilt from git. To grow
the pool, add a recipe under a new name and never edit a built one: experiments and
SFT sources refer to pools by name and sha256.

Tasks are validated against ``data/tau2/domains/<domain>/tasks_hard.json`` as raw
JSON, so building a pool does not need the tau2 venv. Two recipe options need it:
``domain_filter="scripted"`` (pool v1), which asks tau2's env manager how each domain
is simulated, and ``order_gate`` (broad pool v1), which replays gold actions.

Bucket assignments from tau2's calibration are deliberately not inherited: they were
cut on a solo Qwen3.5-4B, and the Decomposer is a different system. What a pool
inherits is the task list and the held-out hygiene.

    python gyms/tau2_gym/task_pools/build_pool.py --list
    python gyms/tau2_gym/task_pools/build_pool.py decomposer_train_v2
"""

from __future__ import annotations

import argparse
import collections
import datetime
import glob
import json
import subprocess
import sys
import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from gyms.tau2_gym.task_pools import POOLS_DIR, TaskKey, pool_sha256  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TAU2_ROOT = REPO_ROOT / "external" / "tau2_gym"
SPLITS_DIR = Path("data") / "training_splits"
DOMAINS_DIR = Path("data") / "tau2" / "domains"


@dataclass(frozen=True)
class PoolRecipe:
    name: str
    role: Literal["train", "eval"]
    purpose: str
    # Split files under data/training_splits whose tasks are unioned.
    sources: tuple[str, ...]
    # Split files whose (domain, task) pairs are removed.
    exclude: tuple[str, ...] = ()
    # Split files whose `_meta.reserved_domains_never_train` domains are removed whole.
    exclude_reserved_domains: tuple[str, ...] = ()
    # Calibration reports (glob relative to the tau2 root); their p=0 tasks are added.
    # Canon splits drop dead tasks by construction, but those were dead for a solo
    # Qwen3.5-4B, not necessarily for the Decomposer.
    dead_reports: str | None = None
    # "scripted" keeps only the domains tau2's env manager simulates with ScriptedUser.
    # The Decomposer only ever sends the task's first_message, and tau2's own training
    # forces ScriptedUser on every domain, so v2 and later use "all".
    domain_filter: Literal["all", "scripted"] = "all"
    # Every task in every domain's tasks_hard.json is a source too.
    all_tasks_hard: bool = False
    # Name suffixes of rebuilt copies of a domain (e.g. "_dsh") that share its
    # reservation: a reserved domain's copies are removed whole as well.
    exclude_reserved_variants: tuple[str, ...] = ()
    # The Decomposer sends one user turn, so a scripted follow-up never arrives.
    drop_followup: bool = False
    # Tasks whose answer is a clarifying question (the SFT selection drops them).
    drop_ask: bool = False
    # Drop filter/superlative tasks whose DB check depends on the order of their
    # independent writes (IDs minted from row counts): tau2's DB term replays the
    # gold in listed order, so a correct parallel run would fail it.
    order_gate: bool = False


_CANON = ("canon_full_sft.json", "canon_full_grpo.json", "canon_full_dpo.json")
_HELD_OUT = ("HELDOUT_v2.json", "heldout_exec_v5.json", "cycle_heldout.json", "QUARANTINE.json")
_DEAD_REPORTS = "progress/gaia2/unified/full/full_report_*.json"
_ASK_TEMPLATES = ("ambiguity", "pure_ambiguity")
# Templates whose gold writes are independent of each other (one write per row).
_INDEPENDENT_WRITE_TEMPLATES = ("filter_cardinality", "superlative_chain")

RECIPES: dict[str, PoolRecipe] = {
    recipe.name: recipe
    for recipe in (
        PoolRecipe(
            name="decomposer_pool_v1",
            role="train",
            purpose="First Decomposer pool: canon tasks on the ScriptedUser domains only.",
            sources=_CANON,
            exclude=_HELD_OUT,
            dead_reports=_DEAD_REPORTS,
            domain_filter="scripted",
        ),
        PoolRecipe(
            name="decomposer_train_v2",
            role="train",
            purpose="Canon tasks on every canon domain: teacher traces for SFT and OPD rollouts.",
            sources=_CANON,
            exclude=_HELD_OUT,
            exclude_reserved_domains=("HELDOUT_v2.json",),
            dead_reports=_DEAD_REPORTS,
        ),
        PoolRecipe(
            name="decomposer_broad_v1",
            role="train",
            purpose=(
                "Every runnable tasks_hard.json task outside the held-out sets, without "
                "ask tasks or DB-order-sensitive ones: broad teacher traces."
            ),
            sources=(),
            all_tasks_hard=True,
            exclude=_HELD_OUT,
            exclude_reserved_domains=("HELDOUT_v2.json",),
            exclude_reserved_variants=("_dsh",),
            drop_followup=True,
            drop_ask=True,
            order_gate=True,
        ),
        PoolRecipe(
            name="decomposer_eval_v1",
            role="eval",
            purpose="HELDOUT_v2: 18 domains reserved from all tau2 training.",
            sources=("HELDOUT_v2.json",),
            exclude=("QUARANTINE.json",),
        ),
    )
}


def read_split(tau2_root: Path, name: str) -> tuple[set[TaskKey], dict[str, Any]]:
    raw = json.loads((tau2_root / SPLITS_DIR / name).read_text(encoding="utf-8"))
    meta = raw.pop("_meta", None) or {}
    keys = {
        (domain, str(task_id))
        for domain, ids in raw.items()
        if isinstance(ids, list)
        for task_id in ids
    }
    return keys, meta


def dead_tasks(tau2_root: Path, pattern: str) -> set[TaskKey]:
    """Tasks whose latest calibration report (in sorted file order) has zero passes."""
    pass_counts: dict[TaskKey, int] = {}
    for path in sorted(glob.glob(str(tau2_root / pattern))):
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        default_trials = report.get("trials", 8)
        for result in report.get("results", []):
            domain = result.get("domain")
            task_id = result.get("task_id") or result.get("task")
            if domain is None or task_id is None:
                continue
            passes = result.get("n_pass", result.get("passes"))
            if passes is None and result.get("pass_rate") is not None:
                passes = round(result["pass_rate"] * result.get("trials", default_trials))
            if passes is not None:
                pass_counts[(domain, str(task_id))] = passes
    return {key for key, passes in pass_counts.items() if passes == 0}


def first_message(task: dict[str, Any]) -> str | None:
    """The single user turn the Decomposer receives, as tau2_export reads it."""
    instructions = (task.get("user_scenario") or {}).get("instructions")
    if isinstance(instructions, str):
        return instructions or None
    if isinstance(instructions, dict):
        return instructions.get("first_message") or None
    return None


def scripted_domains(tau2_root: Path) -> set[str]:
    for entry in (tau2_root / "training", tau2_root / "src"):
        if str(entry) not in sys.path:
            sys.path.insert(0, str(entry))
    import tau2_env_manager
    from tau2.registry import registry

    return {
        domain
        for domain in registry.get_domains()
        if tau2_env_manager._user_type_for_domain(domain) == "scripted"
    }


def template_of(task: dict[str, Any]) -> str | None:
    purpose = str((task.get("description") or {}).get("purpose") or "")
    return purpose[1:].split("]", 1)[0] if purpose.startswith("[") else None


def is_ask_task(task: dict[str, Any]) -> bool:
    return template_of(task) in _ASK_TEMPLATES or (task.get("answer_spec") or {}).get("type") == "ask"


def needs_order_gate(task: dict[str, Any]) -> bool:
    criteria = task.get("evaluation_criteria") or {}
    return (
        template_of(task) in _INDEPENDENT_WRITE_TEMPLATES
        and "DB" in (criteria.get("reward_basis") or [])
        and len(criteria.get("actions") or []) >= 2
    )


# (domain, raw task) -> "ok" | "db_order_sensitive" | "gold_replay_error"
OrderCheck = Callable[[str, dict[str, Any]], str]


def tau2_order_check(tau2_root: Path) -> OrderCheck:
    """Replays a task's gold in listed and reversed order on fresh worlds (tau2 venv).

    Failing calls are skipped, as tau2's own gold replay for the DB term does; a task
    whose world or gold cannot be set up at all is a replay error.
    """
    if str(tau2_root / "src") not in sys.path:
        sys.path.insert(0, str(tau2_root / "src"))
    os.environ.setdefault("TAU2_DATA_DIR", str(tau2_root / "data"))
    from tau2.data_model.tasks import Task
    from tau2.registry import registry

    def final_hash(domain: str, task: dict[str, Any], actions: list[dict[str, Any]]) -> str:
        if task.get("world") is None:
            os.environ.pop("TAU2_WORLD", None)
        else:
            os.environ["TAU2_WORLD"] = str(task["world"])
        environment = registry.get_env_constructor(domain)(language="en")
        initial = Task.model_validate(task).initial_state
        environment.set_state(
            initial.initialization_data if initial else None,
            initial.initialization_actions if initial else None,
            [],
        )
        for action in actions:
            try:
                environment.make_tool_call(
                    tool_name=action["name"], requestor=action.get("requestor", "assistant"), **action["arguments"]
                )
            except Exception:  # noqa: BLE001 - matches tau2's gold replay
                continue
        return environment.get_db_hash()

    def check(domain: str, task: dict[str, Any]) -> str:
        actions = list((task.get("evaluation_criteria") or {}).get("actions") or [])
        try:
            same = final_hash(domain, task, actions) == final_hash(domain, task, actions[::-1])
        except Exception:  # noqa: BLE001
            return "gold_replay_error"
        return "ok" if same else "db_order_sensitive"

    return check


def _all_tasks_hard(tau2_root: Path) -> set[TaskKey]:
    keys: set[TaskKey] = set()
    for path in sorted((tau2_root / DOMAINS_DIR).glob("*/tasks_hard.json")):
        for task in json.loads(path.read_text(encoding="utf-8")):
            if isinstance(task, dict) and "id" in task:
                keys.add((path.parent.name, str(task["id"])))
    return keys


def build_pool(
    recipe: PoolRecipe, tau2_root: Path, *, order_check: OrderCheck | None = None
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """The pool and the counts that explain it. Pure apart from reading ``tau2_root``
    (and replaying gold actions through tau2 when the recipe has an order gate)."""
    union: set[TaskKey] = set()
    for name in recipe.sources:
        union |= read_split(tau2_root, name)[0]
    if recipe.all_tasks_hard:
        union |= _all_tasks_hard(tau2_root)
    dead = dead_tasks(tau2_root, recipe.dead_reports) if recipe.dead_reports else set()

    excluded: set[TaskKey] = set()
    for name in recipe.exclude:
        excluded |= read_split(tau2_root, name)[0]
    reserved: set[str] = set()
    for name in recipe.exclude_reserved_domains:
        reserved |= set(read_split(tau2_root, name)[1].get("reserved_domains_never_train", []))
    reserved |= {domain + suffix for domain in reserved for suffix in recipe.exclude_reserved_variants}
    allowed = scripted_domains(tau2_root) if recipe.domain_filter == "scripted" else None
    if recipe.order_gate and order_check is None:
        order_check = tau2_order_check(tau2_root)
    gated = 0

    dropped: collections.Counter[str] = collections.Counter()
    domain_tasks: dict[str, dict[str, dict[str, Any]] | None] = {}
    pool: set[TaskKey] = set()
    for key in sorted(union | dead):
        domain, task_id = key
        if key in excluded:
            dropped["excluded_split"] += 1
            continue
        if domain in reserved:
            dropped["reserved_domain"] += 1
            continue
        if allowed is not None and domain not in allowed:
            dropped["not_scripted"] += 1
            continue
        if domain not in domain_tasks:
            path = tau2_root / DOMAINS_DIR / domain / "tasks_hard.json"
            domain_tasks[domain] = (
                {str(task["id"]): task for task in json.loads(path.read_text(encoding="utf-8"))}
                if path.is_file()
                else None
            )
        tasks = domain_tasks[domain]
        if tasks is None:
            dropped["no_tasks_hard_file"] += 1
            continue
        task = tasks.get(task_id)
        if task is None:
            dropped["not_in_tasks_hard"] += 1
            continue
        if first_message(task) is None:
            dropped["no_first_message"] += 1
            continue
        if recipe.drop_followup and (task.get("evaluation_criteria") or {}).get("followup_triggers"):
            dropped["needs_followup"] += 1
            continue
        if recipe.drop_ask and is_ask_task(task):
            dropped["ask_task"] += 1
            continue
        if recipe.order_gate and needs_order_gate(task):
            assert order_check is not None
            gated += 1
            verdict = order_check(domain, task)
            if verdict != "ok":
                dropped[verdict] += 1
                continue
        pool.add(key)

    by_domain: dict[str, list[str]] = collections.defaultdict(list)
    for domain, task_id in sorted(pool):
        by_domain[domain].append(task_id)
    counts = {
        "sources_union": len(union),
        "dead_from_calibration": len(dead),
        "dead_outside_sources": len(dead - union),
        "dropped": dict(sorted(dropped.items())),
        "pool_tasks": len(pool),
        "pool_domains": len(by_domain),
        "of_which_dead": len(pool & dead),
    }
    if recipe.order_gate:
        counts["order_gate_checked"] = gated
    return dict(by_domain), counts


def _built_pools(pools_dir: Path) -> Iterable[tuple[dict[str, Any], set[TaskKey]]]:
    for meta_path in sorted(pools_dir.glob("*.meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        pool_path = meta_path.with_name(meta_path.name.removesuffix(".meta.json") + ".json")
        raw = json.loads(pool_path.read_text(encoding="utf-8"))
        yield meta, {(domain, str(task_id)) for domain, ids in raw.items() for task_id in ids}


def check_disjoint(recipe: PoolRecipe, pool: dict[str, list[str]], pools_dir: Path) -> dict[str, int]:
    """Refuse any train/eval task overlap; report shared domains for visibility."""
    keys = {(domain, task_id) for domain, ids in pool.items() for task_id in ids}
    shared_domains: dict[str, int] = {}
    for meta, other in _built_pools(pools_dir):
        # Pool v1 predates roles; it is a training pool.
        if meta["name"] == recipe.name or meta.get("role", "train") == recipe.role:
            continue
        overlap = keys & other
        if overlap:
            raise SystemExit(
                f"{recipe.name} ({recipe.role}) shares {len(overlap)} tasks with "
                f"{meta['name']} ({meta.get('role', 'train')}), e.g. {sorted(overlap)[:5]}"
            )
        shared_domains[meta["name"]] = len({d for d, _ in keys} & {d for d, _ in other})
    return shared_domains


def _git_head(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else None


def write_pool(
    recipe: PoolRecipe,
    pool: dict[str, list[str]],
    counts: dict[str, Any],
    *,
    tau2_root: Path,
    pools_dir: Path,
    force: bool,
) -> str:
    """Write the pool and its meta; an identical rebuild is a no-op."""
    path = pools_dir / f"{recipe.name}.json"
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) == pool:
            return "unchanged"
        if not force:
            raise SystemExit(
                f"{path} exists with different tasks. Pools are immutable: add a new "
                f"recipe, or pass --force to overwrite deliberately."
            )
    shared_domains = check_disjoint(recipe, pool, pools_dir)

    path.write_text(json.dumps(pool, indent=1) + "\n", encoding="utf-8")
    meta = {
        "name": recipe.name,
        "role": recipe.role,
        "purpose": recipe.purpose,
        "built": datetime.date.today().isoformat(),
        "built_by": "gyms/tau2_gym/task_pools/build_pool.py",
        "recipe": asdict(recipe),
        "tau2_commit": _git_head(tau2_root),
        "counts": counts,
        "shared_domains_with": shared_domains,
        "sha256": pool_sha256(path),
    }
    path.with_name(f"{recipe.name}.meta.json").write_text(
        json.dumps(meta, indent=1) + "\n", encoding="utf-8"
    )
    return "written"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help="recipe names to build")
    parser.add_argument("--list", action="store_true", help="list recipes and exit")
    parser.add_argument("--tau2-root", type=Path, default=DEFAULT_TAU2_ROOT)
    parser.add_argument("--pools-dir", type=Path, default=POOLS_DIR)
    parser.add_argument("--force", action="store_true", help="overwrite a pool whose tasks changed")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list or not args.names:
        for recipe in RECIPES.values():
            print(f"{recipe.name:24} {recipe.role:5}  {recipe.purpose}")
        return 0
    unknown = [name for name in args.names if name not in RECIPES]
    if unknown:
        raise SystemExit(f"Unknown pool recipes: {unknown}; see --list")
    for name in args.names:
        recipe = RECIPES[name]
        pool, counts = build_pool(recipe, args.tau2_root)
        status = write_pool(
            recipe, pool, counts, tau2_root=args.tau2_root, pools_dir=args.pools_dir, force=args.force
        )
        print(json.dumps({"name": name, "status": status, **counts}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
