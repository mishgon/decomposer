"""Iterative on-policy distillation of the Decomposer manager on the tau2 gym.

Each round, from checkpoint_r:
  1. sample `tasks_per_round` tasks from the pool (deterministic in seed and round);
  2. roll out with gyms/tau2_gym/run.py --experiment opd_rollout, which serves
     checkpoint_r in vLLM and records token ids and behaviour log-probs;
  3. build samples (training/opd/samples.py);
  4. score them with the teacher (training/opd/teacher.py);
  5. train one pass and export checkpoint_{r+1} (training/opd/train.py, torchrun);
  6. every `eval.every` rounds, evaluate checkpoint_{r+1} on the held-out pool.

Every step writes a marker, so an interrupted loop resumes where it stopped. The
steps only communicate through files in `<root>/<name>/round_NNN/`; an online
trainer can later replace this driver without changing them.

    python -m training.opd.loop --config training/opd/configs/tau2_qwen35_4b.yaml --dry
    python -m training.opd.loop --config training/opd/configs/tau2_qwen35_4b.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from gyms.tau2_gym.run import checkpoint_fingerprint
from gyms.tau2_gym.task_pools import TaskKey, load_pool
from training.opd.samples import build_samples, read_rollouts, write_samples
from training.opd.teacher import TeacherEndpoint, score_file

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
TAU2_RUN = REPO_ROOT / "gyms" / "tau2_gym" / "run.py"


@dataclass(frozen=True)
class RoundPaths:
    root: Path

    @property
    def tasks(self) -> Path:
        return self.root / "tasks.json"

    @property
    def rollouts(self) -> Path:
        return self.root / "rollouts"

    @property
    def samples(self) -> Path:
        return self.root / "samples.jsonl"

    @property
    def scored(self) -> Path:
        return self.root / "scored.jsonl"

    @property
    def train(self) -> Path:
        return self.root / "train"

    @property
    def checkpoint(self) -> Path:
        return self.train / "final"

    @property
    def evaluation(self) -> Path:
        return self.root / "eval"

    @property
    def summary(self) -> Path:
        return self.root / "round.json"


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for section in ("run", "rollout", "teacher", "samples", "train"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"{path}: missing section {section!r}")
    return config


def run_dir(config: dict[str, Any]) -> Path:
    return Path(config["run"]["root"]) / config["run"]["name"]


def sample_tasks(pool: str, *, count: int, seed: int, round_index: int) -> list[TaskKey]:
    """`count` pool tasks ranked by a hash of (seed, round, task): fresh tasks every round."""
    tasks, _ = load_pool(pool)
    if count > len(tasks):
        raise ValueError(f"tasks_per_round={count} exceeds the {len(tasks)} tasks of {pool}")

    def rank(key: TaskKey) -> str:
        return hashlib.sha256(f"{seed}/{round_index}/{key[0]}/{key[1]}".encode()).hexdigest()

    chosen = set(sorted(tasks, key=rank)[:count])
    return [key for key in tasks if key in chosen]


def round_checkpoint(config: dict[str, Any], round_index: int) -> Path:
    if round_index == 0:
        return Path(config["run"]["init_checkpoint"])
    return RoundPaths(run_dir(config) / f"round_{round_index - 1:03d}").checkpoint


def rollout_command(config: dict[str, Any], paths: RoundPaths, checkpoint: Path) -> list[str]:
    rollout = config["rollout"]
    command = [
        str(PROJECT_PYTHON), str(TAU2_RUN),
        "--experiment", rollout.get("experiment", "opd_rollout"),
        "--pool", rollout["pool"],
        "--tasks-file", str(paths.tasks),
        "--num-repeats", str(rollout["repeats"]),
        "--manager-checkpoint", str(checkpoint),
        "--manager-gpu", str(rollout["manager_gpu"]),
        "--subagent-backend", rollout.get("subagent_backend", "llm_proxy"),
        "--output-dir", str(paths.rollouts),
        "--port-offset", str(rollout.get("port_offset", 0)),
    ]
    if rollout.get("concurrency"):
        command += ["--concurrency", str(rollout["concurrency"])]
    return command


def train_command(config: dict[str, Any], paths: RoundPaths, checkpoint: Path, round_index: int) -> list[str]:
    train = config["train"]
    gpus = [gpu for gpu in str(train["gpus"]).split(",") if gpu]
    return [
        str(REPO_ROOT / ".venv" / "bin" / "torchrun"),
        f"--nproc-per-node={len(gpus)}",
        f"--master-port={train.get('master_port', 29517)}",
        "-m", "training.opd.train",
        "--config", str(REPO_ROOT / train["config"]),
        "--init", str(checkpoint),
        "--samples", str(paths.scored),
        "--output-dir", str(paths.train),
        "--seed", str(int(config["run"].get("seed", 0)) * 1000 + round_index),
    ]


def train_environment(config: dict[str, Any]) -> dict[str, str]:
    """GPUs and PYTHONPATH overlays for the torchrun step.

    Qwen3.5's gated-delta backward needs Triton >= 3.7.1 on Hopper (fla refuses older
    versions, which give wrong gradients). The project venv pins 3.6, so the SFT jobs
    put a Triton 3.7.1 overlay first on PYTHONPATH; `train.pythonpath` does the same.
    """
    train = config["train"]
    overlays = [str(Path(entry)) for entry in train.get("pythonpath") or []]
    for entry in overlays:
        if not Path(entry).is_dir():
            raise SystemExit(f"train.pythonpath entry does not exist: {entry}")
    existing = os.environ.get("PYTHONPATH")
    return {
        "CUDA_VISIBLE_DEVICES": str(train["gpus"]),
        "PYTHONPATH": os.pathsep.join([*overlays, str(REPO_ROOT), *([existing] if existing else [])]),
    }


def eval_command(config: dict[str, Any], paths: RoundPaths) -> list[str]:
    evaluation = config["eval"]
    return [
        str(PROJECT_PYTHON), str(TAU2_RUN),
        "--experiment", evaluation.get("experiment", "qwen35_4b_student_checkpoint"),
        "--pool", evaluation["pool"],
        "--tasks-per-domain", str(evaluation["tasks_per_domain"]),
        "--num-repeats", str(evaluation.get("repeats", 1)),
        "--manager-checkpoint", str(paths.checkpoint),
        "--manager-gpu", str(config["rollout"]["manager_gpu"]),
        "--subagent-backend", config["rollout"].get("subagent_backend", "llm_proxy"),
        "--output-dir", str(paths.evaluation),
        "--port-offset", str(config["rollout"].get("port_offset", 0)),
    ]


def _run(command: list[str], *, log: Path, env: dict[str, str] | None = None) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"[opd] $ {' '.join(command)}", flush=True)
    with log.open("a", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=REPO_ROOT, env={**os.environ, **(env or {})},
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise SystemExit(f"step failed (exit {result.returncode}); see {log}")


def _atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def run_round(config: dict[str, Any], round_index: int) -> dict[str, Any]:
    paths = RoundPaths(run_dir(config) / f"round_{round_index:03d}")
    paths.root.mkdir(parents=True, exist_ok=True)
    checkpoint = round_checkpoint(config, round_index)
    if not (checkpoint / "config.json").is_file():
        raise SystemExit(f"round {round_index}: checkpoint {checkpoint} is not an exported model")
    logs = paths.root / "logs"
    rollout = config["rollout"]

    if not paths.tasks.is_file():
        tasks = sample_tasks(rollout["pool"], count=int(rollout["tasks_per_round"]),
                             seed=int(config["run"].get("seed", 0)), round_index=round_index)
        _atomic_json(paths.tasks, [list(key) for key in tasks])

    if not (paths.rollouts / ".eval_done.json").is_file():
        if paths.rollouts.exists():
            # An interrupted rollout: rerun it whole rather than mixing checkpoints.
            shutil.rmtree(paths.rollouts)
        _run(rollout_command(config, paths, checkpoint), log=logs / "rollout.log")
    status = json.loads((paths.rollouts / "run_status.json").read_text(encoding="utf-8"))
    if status["manager_checkpoint_fingerprint"] != checkpoint_fingerprint(checkpoint):
        raise SystemExit(f"round {round_index}: rollouts were not produced by {checkpoint}")

    samples_summary_path = paths.root / "samples_summary.json"
    if not samples_summary_path.is_file():
        samples, summary = build_samples(
            read_rollouts([paths.rollouts / "rollouts.jsonl"]),
            max_length=int(config["samples"].get("max_length", 65536)),
            include_failed=bool(config["samples"].get("include_failed", True)),
        )
        write_samples(samples, paths.samples)
        _atomic_json(samples_summary_path, summary)
    samples_summary = json.loads(samples_summary_path.read_text(encoding="utf-8"))

    teacher_summary_path = paths.root / "teacher_summary.json"
    if not teacher_summary_path.is_file():
        teacher = config["teacher"]
        endpoint = TeacherEndpoint.from_env(
            model=teacher["model"],
            base_url_env=teacher.get("base_url_env", "LLM_PROXY_URL"),
            api_key_env=teacher.get("api_key_env", "LLM_PROXY_MASTER_KEY"),
            verify_tls=bool(teacher.get("verify_tls", False)),
        )
        summary = score_file(endpoint, paths.samples, paths.scored, concurrency=int(teacher.get("concurrency", 8)))
        if summary["failed"]:
            raise SystemExit(f"round {round_index}: {summary['failed']} samples failed teacher scoring: {summary['failures'][:3]}")
        _atomic_json(teacher_summary_path, summary)

    if not (paths.train / "opd_summary.json").is_file():
        if paths.train.exists():
            shutil.rmtree(paths.train)
        _run(train_command(config, paths, checkpoint, round_index), log=logs / "train.log",
             env=train_environment(config))
    train_summary = json.loads((paths.train / "opd_summary.json").read_text(encoding="utf-8"))

    evaluation = config.get("eval") or {}
    eval_result = None
    every = int(evaluation.get("every", 0) or 0)
    if every and (round_index + 1) % every == 0:
        if not (paths.evaluation / ".eval_done.json").is_file():
            if paths.evaluation.exists():
                shutil.rmtree(paths.evaluation)
            _run(eval_command(config, paths), log=logs / "eval.log")
        eval_result = json.loads((paths.evaluation / "run_status.json").read_text(encoding="utf-8"))["result"]

    summary = {
        "round": round_index,
        "checkpoint_in": str(checkpoint),
        "checkpoint_out": str(paths.checkpoint),
        "rollout_result": status.get("result"),
        "samples": samples_summary,
        "train": train_summary.get("round_metrics"),
        "eval": eval_result,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(paths.summary, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--rounds", type=int, default=None, help="override run.rounds")
    parser.add_argument("--dry", action="store_true", help="print the next round's commands and exit")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    rounds = args.rounds if args.rounds is not None else int(config["run"]["rounds"])
    directory = run_dir(config)

    start = 0
    while (directory / f"round_{start:03d}" / "round.json").is_file():
        start += 1
    if args.dry:
        paths = RoundPaths(directory / f"round_{start:03d}")
        checkpoint = round_checkpoint(config, start)
        print(json.dumps({
            "run_dir": str(directory), "next_round": start, "rounds": rounds, "checkpoint": str(checkpoint),
            "rollout": rollout_command(config, paths, checkpoint),
            "train": train_command(config, paths, checkpoint, start),
            **({"eval": eval_command(config, paths)} if config.get("eval") else {}),
        }, indent=2))
        return 0

    directory.mkdir(parents=True, exist_ok=True)
    frozen = directory / "config.yaml"
    if frozen.is_file():
        if yaml.safe_load(frozen.read_text(encoding="utf-8")) != config:
            raise SystemExit(f"{args.config} differs from the config this run started with ({frozen})")
    else:
        shutil.copy2(args.config, frozen)
    for round_index in range(start, rounds):
        summary = run_round(config, round_index)
        print(f"[opd] round {round_index}: {json.dumps({k: summary[k] for k in ('rollout_result', 'train', 'eval')})}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
