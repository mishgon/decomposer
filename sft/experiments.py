"""Experiments-as-code for Decomposer supervised fine-tuning jobs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

INSTANCE_TYPES_BY_NUM_GPUS: dict[int, str] = {
    1: "a100plus.1gpu.80vG.12C.244G",
    2: "a100plus.2gpu.80vG.24C.488G",
    4: "a100plus.4gpu.80vG.48C.976G",
}


@dataclass(frozen=True)
class ExperimentConfig:
    """One immutable SFT run; ``name`` is its artifact and dedup identity."""

    name: str
    config_path: str
    description: str
    num_gpus: int = 2
    use_liger_kernel: bool = False
    pytorch_cuda_alloc_conf: str | None = None
    runtime_profile: str | None = None
    benchmark: bool = False


def sft_experiments() -> list[ExperimentConfig]:
    # The configs of earlier runs were removed with their releases; register a
    # config here to run it as an MLSpace job.
    return []


def collect_experiments(filter_name: str | None = None) -> list[ExperimentConfig]:
    experiments = sft_experiments()
    names = [experiment.name for experiment in experiments]
    if len(names) != len(set(names)):
        raise ValueError("SFT experiment names must be unique.")
    if filter_name:
        experiments = [
            experiment for experiment in experiments if filter_name in experiment.name
        ]
    return experiments


def build_train_command(
    experiment: ExperimentConfig,
    *,
    workdir: str | Path,
    output_dir: str | Path,
    resume_from_checkpoint: str | Path | None = None,
) -> list[str]:
    workdir = Path(workdir)
    command = [
        "torchrun",
        "--standalone",
        f"--nproc-per-node={experiment.num_gpus}",
        "-m",
        "sft.train",
        "--config",
        str(workdir / experiment.config_path),
        "--output-dir",
        str(output_dir),
        "--clearml",
    ]
    command.append(
        "--use-liger-kernel" if experiment.use_liger_kernel else "--no-use-liger-kernel"
    )
    if experiment.benchmark:
        command.append("--benchmark")
    if resume_from_checkpoint is not None:
        command.extend(["--resume-from-checkpoint", str(Path(resume_from_checkpoint))])
    return command


def has_training_artifacts(output_dir: str | Path) -> bool:
    output_dir = Path(output_dir)
    final_dir = output_dir / "final"
    weights_exist = (final_dir / "model.safetensors").is_file() or (
        final_dir / "model.safetensors.index.json"
    ).is_file()
    return (output_dir / "training_summary.json").is_file() and weights_exist


def has_experiment_artifacts(
    experiment: ExperimentConfig, output_dir: str | Path
) -> bool:
    output_dir = Path(output_dir)
    if experiment.benchmark:
        return (output_dir / "benchmark_summary.json").is_file()
    return has_training_artifacts(output_dir)
