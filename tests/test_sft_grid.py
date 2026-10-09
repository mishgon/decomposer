from __future__ import annotations

import json
from pathlib import Path

import yaml

from sft.grid import busy_gpus, cell_config, cell_name, pending_cells, report

BASE = {
    "model": {"name_or_path": "/models/base"},
    "data": {"train_file": "/data/train.jsonl"},
    "training": {
        "output_dir": "/runs/base",
        "run_name": "base-run",
        "learning_rate": 1.0e-4,
        "global_batch_size": 8,
        "max_length": 32768,
    },
    "lora": {"r": 32, "alpha": 64, "dropout": 0.05},
    "clearml": {"enabled": True, "task": "Base task", "tags": ["sft", "32k"]},
    "run": {"expected_world_size": 4},
}


def test_cell_config_changes_only_the_grid_keys(tmp_path: Path) -> None:
    cell = tmp_path / cell_name(5e-5, 32)
    config = cell_config(
        BASE, learning_rate=5e-5, global_batch_size=32, cell_dir=cell, grid_name="grid-a"
    )
    assert cell.name == "lr5e-05-gb32"
    assert config["training"] == {
        "output_dir": str(cell / "run"),
        "run_name": "base-run-lr5e-05-gb32",
        "learning_rate": 5e-5,
        "global_batch_size": 32,
        "max_length": 32768,
    }
    assert config["clearml"]["task"] == "Base task lr 5e-05 gb 32"
    assert config["clearml"]["tags"] == ["sft", "32k", "grid-a", "lr-5e-05", "gb-32"]
    for section in ("model", "data", "lora", "run"):
        assert config[section] == BASE[section]
    assert BASE["training"]["learning_rate"] == 1.0e-4


def test_cell_names_keep_every_learning_rate_apart() -> None:
    names = [cell_name(rate, 16) for rate in (1e-4, 1.5e-4, 2e-4, 2.5e-4, 3e-4)]
    assert names[:2] == ["lr0.0001-gb16", "lr0.00015-gb16"]
    assert len(set(names)) == 5


def test_cells_with_a_run_directory_are_skipped(tmp_path: Path) -> None:
    done, reused, pending = (tmp_path / name for name in ("a", "b", "c"))
    (done / "run" / "final").mkdir(parents=True)
    reused.mkdir()
    (reused / "run").symlink_to(done / "run")
    pending.mkdir()
    assert pending_cells([done, reused, pending]) == [pending]


def test_busy_gpus_uses_the_launcher_threshold() -> None:
    output = "0, 0\n1, 1024\n4, 1500\n5, 23000\n"
    assert busy_gpus(output, [0, 1, 4]) == [4]
    assert busy_gpus(output, [0, 1]) == []


def _write_cell(root: Path, learning_rate: float, batch: int, losses: dict[str, float] | None) -> None:
    cell = root / cell_name(learning_rate, batch)
    cell.mkdir()
    config = cell_config(
        BASE, learning_rate=learning_rate, global_batch_size=batch, cell_dir=cell, grid_name="g"
    )
    (cell / "config.yaml").write_text(yaml.safe_dump(config))
    if losses is None:
        (cell / "exit_code").write_text("1\n")
        return
    run = cell / "run"
    history = []
    for step, scale in ((100, 1.0), (200, 0.9)):
        for name, loss in losses.items():
            history.append({"epoch": step / 100, f"eval_{name}_loss": loss * scale, "step": step})
    for step in (100, 200):
        (run / f"checkpoint-{step}").mkdir(parents=True)
    (run / "checkpoint-200" / "trainer_state.json").write_text(json.dumps({"log_history": history}))
    (run / "training_summary.json").write_text(
        json.dumps(
            {
                "training_state": {
                    "best_model_checkpoint": str(run / "checkpoint-200"),
                    "completed_epochs": 2.0,
                    "early_stopped": False,
                }
            }
        )
    )


def test_report_takes_per_gym_losses_at_the_best_checkpoint(tmp_path: Path) -> None:
    _write_cell(tmp_path, 1e-4, 8, {"all": 0.5, "tau2_gym": 0.6})
    _write_cell(tmp_path, 2e-4, 8, None)
    rows = {row["cell"]: row for row in report(tmp_path)}

    done = rows["lr0.0001-gb8"]
    assert done["status"] == "done"
    assert done["best_epoch"] == 2.0
    assert done["eval_all_loss"] == 0.45
    assert abs(done["eval_tau2_gym_loss"] - 0.54) < 1e-12
    assert rows["lr0.0002-gb8"]["status"] == "failed (exit 1)"
    assert json.loads((tmp_path / "summary.json").read_text())[0]["cell"] == "lr0.0001-gb8"
