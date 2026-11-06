"""The stock-model MLSpace smoke experiment and its durable storage."""
from dataclasses import dataclass
from pathlib import Path

HOME = Path('/mnt/shared_ru.ml.SZ-5_000264/matrosov')
ARTIFACTS_ROOT = HOME / 'decomposer-synth-artifacts'
STAGING_ROOT = ARTIFACTS_ROOT / 'code'
SANITY_ROOT = ARTIFACTS_ROOT / 'sanity'
MODEL = Path('/home/jovyan/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/'
             '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a')
INSTANCE_TYPES_BY_NUM_GPUS = {1: 'a100plus.1gpu.80vG.12C.244G',
                            2: 'a100plus.2gpu.80vG.24C.488G'}
BASE_IMAGE = 'cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.7.0:0.0.41'


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    rollout_steps: int = 512
    num_gpus: int = 1


def collect_experiments():
    return [ExperimentConfig('stock-qwen4b-synth-workplace20-twoh100-cpus-20261102', num_gpus=2)]
