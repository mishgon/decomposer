"""WideSeek experiments use the shared OPD allocation and runtime settings."""
from opd.synth.mlspace.experiments import ExperimentConfig


def collect_experiments():
    return [ExperimentConfig('stock-qwen4b-wideseek-workplace20', num_gpus=2)]
