"""Learning-rate x global-batch grid over one training config, two runs at a time.

    python -m sft.grid --config <base yaml> --learning-rates 5e-5 1e-4 2e-4 \
        --global-batch-sizes 8 16 32 --gpu-groups 0,1,2,3 4,5,6,7 --output-root <dir>
    python -m sft.grid --report <dir>

Each cell lives in ``<root>/lr<lr>-gb<batch>/`` with its derived ``config.yaml``,
``train.log`` and the run's ``run/`` output. A cell whose ``run/`` exists is
skipped, so a grid resumes where it stopped and can reuse an earlier run through
a ``run`` symlink. Runs go through ``sft/train_qwen35_h200.sh``, one GPU group at
a time per worker; a worker waits until its GPUs are free and never stops other
processes.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

JsonObject = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = REPO_ROOT / "sft" / "train_qwen35_h200.sh"
# The launcher refuses GPUs holding more memory than this.
FREE_GPU_MIB = 1024
GPU_POLL_SECONDS = 60
# Launch attempts for a cell that fails before creating its run directory, such
# as a GPU taken between the free check and the launch.
LAUNCH_ATTEMPTS = 3


def cell_name(learning_rate: float, global_batch_size: int) -> str:
    return f"lr{learning_rate:g}-gb{global_batch_size}"


def cell_config(
    base: Mapping[str, Any],
    *,
    learning_rate: float,
    global_batch_size: int,
    cell_dir: Path,
    grid_name: str,
) -> JsonObject:
    """The base config with only the grid's learning rate, batch and run identity changed."""
    config = copy.deepcopy(dict(base))
    training = config["training"]
    training["learning_rate"] = learning_rate
    training["global_batch_size"] = global_batch_size
    training["output_dir"] = str(cell_dir / "run")
    training["run_name"] = f"{training['run_name']}-{cell_dir.name}"
    clearml = config["clearml"]
    clearml["task"] = f"{clearml['task']} lr {learning_rate:g} gb {global_batch_size}"
    clearml["tags"] = [
        *clearml.get("tags", []),
        grid_name,
        f"lr-{learning_rate:g}",
        f"gb-{global_batch_size}",
    ]
    return config


def pending_cells(cells: Sequence[Path]) -> list[Path]:
    return [cell for cell in cells if not (cell / "run").exists()]


def busy_gpus(nvidia_smi_csv: str, gpus: Sequence[int]) -> list[int]:
    """GPUs from ``gpus`` that hold memory, given ``index, memory.used`` CSV rows."""
    used = {}
    for line in nvidia_smi_csv.strip().splitlines():
        index, memory = (field.strip() for field in line.split(","))
        used[int(index)] = int(memory)
    return [gpu for gpu in gpus if used.get(gpu, 0) > FREE_GPU_MIB]


def _log(message: str) -> None:
    print(f"{datetime.now():%Y-%m-%dT%H:%M:%S} {message}", flush=True)


def _wait_for_gpus(group: str) -> None:
    gpus = [int(gpu) for gpu in group.split(",")]
    while True:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        if not busy_gpus(output, gpus):
            return
        time.sleep(GPU_POLL_SECONDS)


def _worker(group: str, queue: list[Path], lock: threading.Lock) -> None:
    while True:
        with lock:
            if not queue:
                return
            cell = queue.pop(0)
        for attempt in range(1, LAUNCH_ATTEMPTS + 1):
            _wait_for_gpus(group)
            _log(f"start {cell.name} on GPUs {group} (attempt {attempt})")
            with (cell / "train.log").open("a", encoding="utf-8") as log:
                code = subprocess.run(
                    [str(LAUNCHER), str(cell / "config.yaml"), group],
                    cwd=REPO_ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                ).returncode
            (cell / "exit_code").write_text(f"{code}\n", encoding="utf-8")
            _log(f"end {cell.name} exit {code}")
            if code == 0 or (cell / "run").exists():
                break
            time.sleep(GPU_POLL_SECONDS)


def run_grid(
    *,
    config_path: Path,
    learning_rates: Sequence[float],
    global_batch_sizes: Sequence[int],
    gpu_groups: Sequence[str],
    root: Path,
) -> None:
    base = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root.mkdir(parents=True, exist_ok=True)
    cells = []
    for learning_rate, global_batch_size in itertools.product(
        learning_rates, global_batch_sizes
    ):
        cell = root / cell_name(learning_rate, global_batch_size)
        cell.mkdir(exist_ok=True)
        config = cell_config(
            base,
            learning_rate=learning_rate,
            global_batch_size=global_batch_size,
            cell_dir=cell,
            grid_name=root.name,
        )
        (cell / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        cells.append(cell)
    queue = pending_cells(cells)
    _log(f"{len(queue)} of {len(cells)} cells to run: {[cell.name for cell in queue]}")
    lock = threading.Lock()
    workers = [
        threading.Thread(target=_worker, args=(group, queue, lock)) for group in gpu_groups
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    _log("grid finished")


def _best_step(training_state: Mapping[str, Any]) -> int | None:
    checkpoint = training_state.get("best_model_checkpoint")
    if not checkpoint:
        return None
    return int(Path(checkpoint).name.removeprefix("checkpoint-"))


def cell_report(cell: Path) -> JsonObject:
    config = yaml.safe_load((cell / "config.yaml").read_text(encoding="utf-8"))
    row: JsonObject = {
        "cell": cell.name,
        "learning_rate": config["training"]["learning_rate"],
        "global_batch_size": config["training"]["global_batch_size"],
    }
    run = cell / "run"
    summary_path = run / "training_summary.json"
    if not summary_path.is_file():
        exit_code = cell / "exit_code"
        row["status"] = (
            f"failed (exit {exit_code.read_text().strip()})"
            if exit_code.is_file() and exit_code.read_text().strip() != "0"
            else "not finished"
        )
        return row
    state = json.loads(summary_path.read_text(encoding="utf-8"))["training_state"]
    row["status"] = "early stopped" if state["early_stopped"] else "done"
    row["epochs"] = state["completed_epochs"]
    best_step = _best_step(state)
    checkpoints = sorted(
        run.glob("checkpoint-*"), key=lambda path: int(path.name.removeprefix("checkpoint-"))
    )
    if not checkpoints:
        return row
    history = json.loads((checkpoints[-1] / "trainer_state.json").read_text(encoding="utf-8"))[
        "log_history"
    ]
    for entry in history:
        if entry.get("step") != best_step:
            continue
        row["best_epoch"] = entry.get("epoch")
        row.update(
            {key: value for key, value in entry.items() if key.startswith("eval_") and key.endswith("_loss")}
        )
    return row


def report(root: Path) -> list[JsonObject]:
    rows = [
        cell_report(cell)
        for cell in sorted(root.iterdir())
        if cell.is_dir() and (cell / "config.yaml").is_file()
    ]
    (root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    losses = sorted({key for row in rows for key in row if key.startswith("eval_")})
    header = ["cell", "status", "epochs", "best_epoch", *losses]
    print("\t".join(header))
    for row in sorted(rows, key=lambda row: row.get("eval_all_loss", float("inf"))):
        print(
            "\t".join(
                f"{row[key]:.4f}" if isinstance(row.get(key), float) and key.startswith("eval_")
                else str(row.get(key, "-"))
                for key in header
            )
        )
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", type=Path, help="print and save the table of a grid root")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--learning-rates", type=float, nargs="+")
    parser.add_argument("--global-batch-sizes", type=int, nargs="+")
    parser.add_argument("--gpu-groups", nargs="+", help="e.g. 0,1,2,3 4,5,6,7")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    if args.report is not None:
        report(args.report)
        return
    missing = [
        name
        for name in ("config", "learning_rates", "global_batch_sizes", "gpu_groups", "output_root")
        if getattr(args, name) is None
    ]
    if missing:
        parser.error("missing: " + ", ".join(f"--{name.replace('_', '-')}" for name in missing))
    run_grid(
        config_path=args.config,
        learning_rates=args.learning_rates,
        global_batch_sizes=args.global_batch_sizes,
        gpu_groups=args.gpu_groups,
        root=args.output_root,
    )


if __name__ == "__main__":
    main()
