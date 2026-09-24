from __future__ import annotations

from pathlib import Path

import yaml

from gyms.tau2_gym.task_pools import load_pool
from opd import loop

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "opd/tau2_gym/configs/tau2_qwen35_4b.yaml"


def test_task_sampling_is_deterministic_fresh_per_round_and_inside_the_pool() -> None:
    first = loop.sample_tasks("decomposer_train_v2", count=48, seed=0, round_index=0)
    assert first == loop.sample_tasks("decomposer_train_v2", count=48, seed=0, round_index=0)
    second = loop.sample_tasks("decomposer_train_v2", count=48, seed=0, round_index=1)
    assert len(first) == len(set(first)) == 48
    assert first != second
    pool, _ = load_pool("decomposer_train_v2")
    assert set(first) <= set(pool)


def test_rounds_chain_checkpoints(tmp_path: Path) -> None:
    config = loop.load_config(CONFIG)
    config["run"]["root"] = str(tmp_path)
    assert loop.round_checkpoint(config, 0) == Path(config["run"]["init_checkpoint"])
    assert loop.round_checkpoint(config, 3) == tmp_path / config["run"]["name"] / "round_002" / "train" / "final"


def test_commands_wire_the_round_files(tmp_path: Path) -> None:
    config = loop.load_config(CONFIG)
    paths = loop.RoundPaths(tmp_path / "round_000")
    checkpoint = Path("/ckpt")
    rollout = loop.rollout_command(config, paths, checkpoint)
    assert rollout[rollout.index("--experiment") + 1] == "opd_rollout"
    assert rollout[rollout.index("--tasks-file") + 1] == str(paths.tasks)
    assert rollout[rollout.index("--manager-checkpoint") + 1] == "/ckpt"

    train = loop.train_command(config, paths, checkpoint, 2)
    assert "--nproc-per-node=3" in train
    assert train[train.index("--samples") + 1] == str(paths.scored)
    train_config = yaml.safe_load((REPO_ROOT / config["train"]["config"]).read_text())
    assert train_config["run"]["expected_world_size"] == len(config["train"]["gpus"].split(","))

    evaluation = loop.eval_command(config, paths)
    assert evaluation[evaluation.index("--manager-checkpoint") + 1] == str(paths.checkpoint)
    assert evaluation[evaluation.index("--pool") + 1] == "decomposer_eval_v1"


def test_train_environment_puts_overlays_first(tmp_path: Path, monkeypatch) -> None:
    config = loop.load_config(CONFIG)
    overlay = tmp_path / "triton"
    overlay.mkdir()
    config["train"]["pythonpath"] = [str(overlay)]
    monkeypatch.setenv("PYTHONPATH", "/elsewhere")
    env = loop.train_environment(config)
    assert env["CUDA_VISIBLE_DEVICES"] == config["train"]["gpus"]
    assert env["PYTHONPATH"].split(":") == [str(overlay), str(REPO_ROOT), "/elsewhere"]

    config["train"]["pythonpath"] = [str(tmp_path / "missing")]
    import pytest

    with pytest.raises(SystemExit, match="does not exist"):
        loop.train_environment(config)
