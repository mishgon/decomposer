from __future__ import annotations

from dataclasses import dataclass


INSTANCE_TYPES_BY_NUM_GPUS = {
    1: "a100plus.1gpu.80vG.12C.244G",
}


@dataclass(frozen=True)
class InferenceExperiment:
    name: str
    num_gpus: int
    remote_port_start: int
    description: str
    pilot: bool = False


EXPERIMENTS = (
    InferenceExperiment(
        name="toolathlon-bench-pilot",
        num_gpus=1,
        remote_port_start=18199,
        description="Toolathlon bench Qwen remote inference connectivity pilot",
        pilot=True,
    ),
    *(
        InferenceExperiment(
            name=f"toolathlon-bench-worker-{index:02d}",
            num_gpus=1,
            remote_port_start=18200 + index,
            description=f"Toolathlon bench Qwen inference worker {index:02d} (1xH100)",
        )
        for index in range(16)
    ),
)


def collect_experiments(*, pilot: bool) -> list[InferenceExperiment]:
    selected = [
        experiment
        for experiment in EXPERIMENTS
        if experiment.pilot is pilot
    ]
    names = [experiment.name for experiment in selected]
    if len(names) != len(set(names)):
        raise ValueError("MLSpace experiment names must be unique")
    return selected
