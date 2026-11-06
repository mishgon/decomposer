"""WideSeek uses the same allocation, model and storage roots as final synth OPD."""
from opd.synth.mlspace.experiments import (
    HOME, ARTIFACTS_ROOT, STAGING_ROOT, SANITY_ROOT, MODEL, BASE_IMAGE,
    INSTANCE_TYPES_BY_NUM_GPUS, ExperimentConfig,
)

SOURCE = ARTIFACTS_ROOT / 'wideseek/data/width/tasks.jsonl'


def collect_experiments():
    return [ExperimentConfig('stock-qwen4b-wideseek-workplace20-twoh100-20261102', num_gpus=2)]
