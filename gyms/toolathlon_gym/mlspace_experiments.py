from __future__ import annotations

from dataclasses import dataclass


INSTANCE_TYPES_BY_NUM_GPUS = {
    1: "a100plus.1gpu.80vG.12C.244G",
    8: "a100plus.8gpu.80vG.96C.1952G",
}


@dataclass(frozen=True)
class InferenceExperiment:
    name: str
    num_gpus: int
    remote_port_start: int
    description: str


EXPERIMENTS = (
    InferenceExperiment(
        name="toolathlon-gemma-pilot",
        num_gpus=1,
        remote_port_start=18099,
        description="Toolathlon Gemma remote inference connectivity pilot",
    ),
    InferenceExperiment(
        name="toolathlon-gemma-pool-a",
        num_gpus=8,
        remote_port_start=18100,
        description="Toolathlon Gemma inference pool A (8xH100)",
    ),
    InferenceExperiment(
        name="toolathlon-gemma-pool-b",
        num_gpus=8,
        remote_port_start=18108,
        description="Toolathlon Gemma inference pool B (8xH100)",
    ),
)


def collect_experiments(*, pilot: bool) -> list[InferenceExperiment]:
    selected = [
        experiment
        for experiment in EXPERIMENTS
        if (experiment.num_gpus == 1) is pilot
    ]
    names = [experiment.name for experiment in selected]
    if len(names) != len(set(names)):
        raise ValueError("MLSpace experiment names must be unique")
    return selected
