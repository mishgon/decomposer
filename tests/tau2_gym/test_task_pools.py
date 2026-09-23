from __future__ import annotations

import json
from pathlib import Path

import pytest

from gyms.tau2_gym.task_pools import POOLS_DIR, load_pool, pool_sha256
from gyms.tau2_gym.task_pools.build_pool import (
    PoolRecipe,
    build_pool,
    check_disjoint,
    dead_tasks,
    write_pool,
)


def _task(task_id: str, *, first_message: str | None = "Please do it.") -> dict:
    instructions = {"first_message": first_message} if first_message is not None else {}
    return {"id": task_id, "user_scenario": {"instructions": instructions}}


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def tau2_root(tmp_path: Path) -> Path:
    root = tmp_path / "tau2"
    splits = root / "data" / "training_splits"
    _write(splits / "canon.json", {"_meta": {"v": 1}, "shop": ["a", "b", "c", "quiet"], "bank": ["x"]})
    _write(splits / "extra.json", {"bank": ["y"], "vault": ["v1"], "ghost": ["g1"]})
    _write(splits / "heldout.json", {"_meta": {"reserved_domains_never_train": ["vault"]}, "shop": ["b"]})
    domains = root / "data" / "tau2" / "domains"
    _write(domains / "shop" / "tasks_hard.json", [_task("a"), _task("b"), _task("c"), _task("quiet", first_message=None), _task("d")])
    _write(domains / "bank" / "tasks_hard.json", [_task("x"), _task("y")])
    _write(domains / "vault" / "tasks_hard.json", [_task("v1")])
    # `shop/d` is dead in the latest report; `bank/x` was dead but later passed.
    reports = root / "progress" / "reports"
    _write(reports / "r_0.json", {"trials": 8, "results": [{"domain": "bank", "task": "x", "n_pass": 0}]})
    _write(reports / "r_1.json", {"results": [
        {"domain": "bank", "task_id": "x", "n_pass": 3},
        {"domain": "shop", "task": "d", "pass_rate": 0.0, "trials": 4},
    ]})
    return root


def _recipe(**overrides) -> PoolRecipe:
    values = {
        "name": "pool_t",
        "role": "train",
        "purpose": "test",
        "sources": ("canon.json", "extra.json"),
        "exclude": ("heldout.json",),
        "exclude_reserved_domains": ("heldout.json",),
        "dead_reports": "progress/reports/r_*.json",
    }
    return PoolRecipe(**(values | overrides))


def test_dead_tasks_use_the_latest_report(tau2_root: Path) -> None:
    assert dead_tasks(tau2_root, "progress/reports/r_*.json") == {("shop", "d")}


def test_build_pool_applies_exclusions_and_validates_tasks(tau2_root: Path) -> None:
    pool, counts = build_pool(_recipe(), tau2_root)

    assert pool == {"bank": ["x", "y"], "shop": ["a", "c", "d"]}
    assert counts["dropped"] == {
        "excluded_split": 1,  # shop/b is held out
        "no_first_message": 1,  # shop/quiet cannot be sent as one user turn
        "no_tasks_hard_file": 1,  # ghost/g1
        "reserved_domain": 1,  # vault is reserved from training entirely
    }
    assert counts["of_which_dead"] == 1
    assert counts["dead_outside_sources"] == 1


def test_write_pool_is_immutable_and_rebuilds_are_noops(tau2_root: Path, tmp_path: Path) -> None:
    pools_dir = tmp_path / "pools"
    pools_dir.mkdir()
    recipe = _recipe()
    pool, counts = build_pool(recipe, tau2_root)

    assert write_pool(recipe, pool, counts, tau2_root=tau2_root, pools_dir=pools_dir, force=False) == "written"
    tasks, meta = load_pool(pools_dir / "pool_t.json")
    assert tasks == [("bank", "x"), ("bank", "y"), ("shop", "a"), ("shop", "c"), ("shop", "d")]
    assert meta["sha256"] == pool_sha256(pools_dir / "pool_t.json")
    assert meta["recipe"]["sources"] == ["canon.json", "extra.json"]

    assert write_pool(recipe, pool, counts, tau2_root=tau2_root, pools_dir=pools_dir, force=False) == "unchanged"
    with pytest.raises(SystemExit, match="immutable"):
        write_pool(recipe, {"bank": ["x"]}, counts, tau2_root=tau2_root, pools_dir=pools_dir, force=False)


def test_edited_pool_is_rejected(tau2_root: Path, tmp_path: Path) -> None:
    pools_dir = tmp_path / "pools"
    pools_dir.mkdir()
    recipe = _recipe()
    pool, counts = build_pool(recipe, tau2_root)
    write_pool(recipe, pool, counts, tau2_root=tau2_root, pools_dir=pools_dir, force=False)
    (pools_dir / "pool_t.json").write_text(json.dumps({"bank": ["x"]}), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match its meta"):
        load_pool(pools_dir / "pool_t.json")


def test_train_and_eval_pools_may_not_share_tasks(tau2_root: Path, tmp_path: Path) -> None:
    pools_dir = tmp_path / "pools"
    pools_dir.mkdir()
    train = _recipe()
    pool, counts = build_pool(train, tau2_root)
    write_pool(train, pool, counts, tau2_root=tau2_root, pools_dir=pools_dir, force=False)

    leaking = _recipe(name="eval_t", role="eval", sources=("canon.json",), exclude=(), exclude_reserved_domains=())
    leaking_pool, _ = build_pool(leaking, tau2_root)
    with pytest.raises(SystemExit, match="shares"):
        check_disjoint(leaking, leaking_pool, pools_dir)

    clean = _recipe(name="eval_clean", role="eval", sources=("heldout.json",), exclude=(),
                    exclude_reserved_domains=(), dead_reports=None)
    clean_pool, _ = build_pool(clean, tau2_root)
    assert clean_pool == {"shop": ["b"]}
    assert check_disjoint(clean, clean_pool, pools_dir) == {"pool_t": 1}


@pytest.mark.parametrize("name", ["decomposer_pool_v1", "decomposer_train_v2", "decomposer_eval_v1"])
def test_committed_pools_match_their_meta(name: str) -> None:
    tasks, meta = load_pool(name)
    assert meta["name"] == name
    assert meta["counts"]["pool_tasks"] == len(tasks)
    assert len(set(tasks)) == len(tasks)


def test_committed_train_and_eval_pools_are_disjoint_by_domain() -> None:
    train, _ = load_pool("decomposer_train_v2")
    evaluation, eval_meta = load_pool("decomposer_eval_v1")
    assert eval_meta["role"] == "eval"
    # HELDOUT_v2 reserves whole domains, so even the worlds and tools differ.
    assert not {domain for domain, _ in train} & {domain for domain, _ in evaluation}


def test_pools_dir_holds_only_built_pools() -> None:
    for path in POOLS_DIR.glob("*.json"):
        if path.name.endswith(".meta.json"):
            continue
        assert path.with_name(path.stem + ".meta.json").is_file(), path
