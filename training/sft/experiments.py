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


def sft_experiments() -> list[ExperimentConfig]:
    return [
        ExperimentConfig(
            name="gemma4-e2b-smoke-native",
            config_path="training/sft/configs/gemma4_e2b_smoke.yaml",
            description="Gemma-4 E2B SFT smoke (native kernels)",
        ),
        ExperimentConfig(
            name="gemma4-e2b-smoke-liger",
            config_path="training/sft/configs/gemma4_e2b_smoke.yaml",
            description="Gemma-4 E2B SFT smoke (Liger kernels)",
            use_liger_kernel=True,
        ),
        ExperimentConfig(
            name="gemma4-e2b-nonthinking-4gpu-liger-workplace-26b-v3",
            config_path=(
                "training/sft/configs/"
                "gemma4_e2b_nonthinking_4gpu_liger_workplace_26b_v3.yaml"
            ),
            description=(
                "Gemma-4 E2B Decomposer SFT "
                "(non-thinking, Workplace 26B-source v3, 4 GPU, "
                "Liger fused CE, global batch 8)"
            ),
            num_gpus=4,
            use_liger_kernel=True,
            pytorch_cuda_alloc_conf="expandable_segments:True",
        ),
        ExperimentConfig(
            name="gemma4-e4b-nonthinking-4gpu-liger-workplace-26b-v3",
            config_path=(
                "training/sft/configs/"
                "gemma4_e4b_nonthinking_4gpu_liger_workplace_26b_v3.yaml"
            ),
            description=(
                "Gemma-4 E4B Decomposer SFT "
                "(non-thinking, Workplace 26B-source v3, 4 GPU, "
                "Liger fused CE, global batch 4)"
            ),
            num_gpus=4,
            use_liger_kernel=True,
            pytorch_cuda_alloc_conf="expandable_segments:True",
        ),
    ]


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
        "training.sft.train",
        "--config",
        str(workdir / experiment.config_path),
        "--output-dir",
        str(output_dir),
        "--clearml",
    ]
    command.append(
        "--use-liger-kernel" if experiment.use_liger_kernel else "--no-use-liger-kernel"
    )
    if resume_from_checkpoint is not None:
        command.extend(["--resume-from-checkpoint", str(Path(resume_from_checkpoint))])
    return command


def has_training_artifacts(output_dir: str | Path) -> bool:
    output_dir = Path(output_dir)
    return (output_dir / "training_summary.json").is_file() and (
        output_dir / "final" / "model.safetensors"
    ).is_file()
