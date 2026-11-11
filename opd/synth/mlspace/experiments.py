"""The stock-model MLSpace smoke experiment and its durable storage."""
from dataclasses import dataclass
INSTANCE_TYPES_BY_NUM_GPUS = {1: 'a100plus.1gpu.80vG.12C.244G',
                            2: 'a100plus.2gpu.80vG.24C.488G'}
BASE_IMAGE = 'cr.ai.cloud.ru/aicloud-base-images/py3.12-torch2.7.0:0.0.41'


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    rollout_steps: int = 512
    num_gpus: int = 1
    training_gpus: int = 1


def collect_experiments():
    return [ExperimentConfig('stock-qwen4b-synth-workplace20-t1-20261111', num_gpus=2)]
