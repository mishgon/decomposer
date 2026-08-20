"""Experiments-as-code and artifact layout for Gaia2 execution evaluation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

ARTIFACTS_ROOT = Path("/mnt/shared_ru.ml.SZ-5_000264/sukhorukov/decomposer_artifacts")
PROJECT_ROOT = Path("/mnt/shared_ru.ml.SZ-5_000264/sukhorukov/decomposer_sft")
PROJECT_VENV = PROJECT_ROOT / ".venv"
DEFAULT_GAIA2_REPO = PROJECT_ROOT / "external" / "gaia2"

DATASET_ID = "meta-agents-research-environments/gaia2"
DATASET_REVISION = "78ea3bdbdeec2bdcd6afa5420915d8a22f23ed99"
FILESYSTEM_DATASET_ID = "meta-agents-research-environments/gaia2_filesystem"
FILESYSTEM_DATASET_REVISION = "132e26376f5e963bb59f64bcccdd02188cb08dee"
GAIA2_REVISION = "3bee736488864e028231755ce2ee32a7065e8648"
SPLIT = "validation"
DOMAIN = "execution"
SCENARIO_COUNT = 160

DATA_ROOT = ARTIFACTS_ROOT / "evaluation" / "data" / "gaia2"
RESULTS_ROOT = ARTIFACTS_ROOT / "evaluation" / "gaia2" / "results"
PREPARATION_MANIFEST_ROOT = DATA_ROOT / "manifests" / SPLIT / DOMAIN
DECOMPOSER_STAGING_ROOT = ARTIFACTS_ROOT / "code" / "decomposer"
GAIA2_STAGING_ROOT = ARTIFACTS_ROOT / "code" / "gaia2"
GAIA2_VENV_ROOT = ARTIFACTS_ROOT / "venvs" / "gaia2"
UV_CACHE = ARTIFACTS_ROOT / "cache" / "uv"
UV_BIN = ARTIFACTS_ROOT / "tools" / "uv"
HF_HOME = Path("/mnt/shared_ru.ml.SZ-5_000264/.cache/huggingface")

BASE_IMAGE = "cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.7.0:0.0.41"
INSTANCE_TYPES_BY_NUM_GPUS = {
    1: "a100plus.1gpu.80vG.12C.182G",
    2: "a100plus.2gpu.80vG.24C.364G",
}

JUDGE_MODEL = "openai/nvidia/Llama-3.3-70B-Instruct-FP8"
SCENARIO_TIMEOUT_SECONDS = 3600

GEMMA4_E4B_BASE = (
    HF_HOME
    / "hub"
    / "models--google--gemma-4-E4B-it"
    / "snapshots"
    / "ee0ef6023621cff504d758262d4e04895a5af4a2"
)
GEMMA4_E4B_SFT = (
    ARTIFACTS_ROOT
    / "training"
    / "sft"
    / "jobs"
    / "gemma4-e4b-nonthinking-deepseek-e4b-v1-8k-full-4gpu"
    / "final-vllm"
)


def gaia2_lock_hash(gaia2_root: Path) -> str:
    return hashlib.sha256((gaia2_root / "uv.lock").read_bytes()).hexdigest()[:16]


def gaia2_venv(gaia2_root: Path) -> Path:
    return GAIA2_VENV_ROOT / gaia2_lock_hash(gaia2_root)


def dataset_revision_root() -> Path:
    return DATA_ROOT / DATASET_REVISION


def dataset_root() -> Path:
    """Root passed to ARE with ``--config execution``."""

    return dataset_revision_root() / SPLIT


def scenario_dir() -> Path:
    return dataset_root() / DOMAIN


def dataset_manifest() -> Path:
    return dataset_revision_root() / "dataset_manifest.json"


def filesystem_revision_root() -> Path:
    return DATA_ROOT / "filesystem" / FILESYSTEM_DATASET_REVISION


def filesystem_dir() -> Path:
    return filesystem_revision_root() / "demo_filesystem"


def filesystem_manifest() -> Path:
    return filesystem_revision_root() / "filesystem_manifest.json"


@dataclass(frozen=True)
class DecomposerExperiment:
    name: str
    manager_checkpoint: Path
    worker_checkpoint: Path
    num_gpus: int = 2
    manager_served_name: str = "decomposer/gemma4-e4b-sft-deepseek-e4b-v1-8k"
    worker_served_name: str = "google/gemma-4-E4B-it"
    manager_port: int = 8020
    worker_port: int = 8021
    service_port: int = 8124
    subagent_port: int = 2024
    max_model_len: int = 65536
    max_num_seqs: int = 16
    max_completion_tokens: int = 4096
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    gpu_memory_utilization: float = 0.90
    concurrency: int = 4
    manager_thinking: bool = False
    worker_thinking: bool = True
    kind: Literal["decomposer"] = field(init=False, default="decomposer")


@dataclass(frozen=True)
class SimpleExperiment:
    name: str
    checkpoint: Path
    num_gpus: int = 1
    served_name: str = "google/gemma-4-E4B-it"
    port: int = 8100
    max_model_len: int = 65536
    max_num_seqs: int = 16
    max_completion_tokens: int = 4096
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    gpu_memory_utilization: float = 0.90
    concurrency: int = 4
    thinking: bool = True
    kind: Literal["simple"] = field(init=False, default="simple")


Experiment = DecomposerExperiment | SimpleExperiment

DECOMPOSER_EXPERIMENT = DecomposerExperiment(
    name=("gemma4-e4b-sft-deepseek-e4b-v1-8k-non-thinking-gemma4-e4b-thinking"),
    manager_checkpoint=GEMMA4_E4B_SFT,
    worker_checkpoint=GEMMA4_E4B_BASE,
)
SIMPLE_EXPERIMENT = SimpleExperiment(
    name="gemma4-e4b-it-thinking",
    checkpoint=GEMMA4_E4B_BASE,
)
ALL_EXPERIMENTS: tuple[Experiment, ...] = (
    DECOMPOSER_EXPERIMENT,
    SIMPLE_EXPERIMENT,
)
EXPERIMENTS = {experiment.name: experiment for experiment in ALL_EXPERIMENTS}
if len(EXPERIMENTS) != len(ALL_EXPERIMENTS):
    raise ValueError("Gaia2 experiment names must be unique")


def get_experiment(name: str) -> Experiment:
    try:
        return EXPERIMENTS[name]
    except KeyError as error:
        expected = ", ".join(EXPERIMENTS)
        raise ValueError(
            f"Unknown Gaia2 experiment {name!r}; expected: {expected}"
        ) from error


def collect_experiments(
    names: tuple[str, ...] = (),
    filters: tuple[str, ...] = (),
    *,
    default_all: bool = True,
) -> list[Experiment]:
    unknown = [name for name in names if name not in EXPERIMENTS]
    if unknown:
        raise ValueError(f"Unknown Gaia2 experiments: {unknown}")
    if not names and not filters:
        return list(ALL_EXPERIMENTS) if default_all else []
    selected = set(names)
    selected.update(
        experiment.name
        for experiment in ALL_EXPERIMENTS
        if any(pattern in experiment.name for pattern in filters)
    )
    return [experiment for experiment in ALL_EXPERIMENTS if experiment.name in selected]


def preparation_manifest(experiment: Experiment) -> Path:
    return PREPARATION_MANIFEST_ROOT / f"{experiment.name}.json"


def run_name(experiment: Experiment, num_repeats: int) -> str:
    if num_repeats < 1:
        raise ValueError("num_repeats must be at least 1")
    return experiment.name if num_repeats == 1 else f"{experiment.name}-n{num_repeats}"


def output_dir(
    experiment: Experiment,
    num_repeats: int,
    limit: int | None = None,
) -> Path:
    base = RESULTS_ROOT / SPLIT / DOMAIN / run_name(experiment, num_repeats)
    return base if limit is None else base / f"smoke_{limit}"


def completion_marker(
    experiment: Experiment,
    num_repeats: int,
    limit: int | None = None,
) -> Path:
    return output_dir(experiment, num_repeats, limit) / ".eval_done.json"


def job_description(
    experiment: Experiment,
    num_repeats: int,
    limit: int | None = None,
) -> str:
    identity = run_name(experiment, num_repeats)
    if limit is not None:
        identity += f"-smoke-{limit}"
    return f"gaia2-{SPLIT}-{DOMAIN} {experiment.kind}-agent {identity}"
