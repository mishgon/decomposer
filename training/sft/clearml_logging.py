from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from typing import Any

import torch
from transformers.integrations import ClearMLCallback


def validate_weight_norm_interval(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            "clearml.weight_norm_interval_steps must be a positive integer."
        )
    return value


def _metric_phase(logs: Mapping[str, Any]) -> str:
    if any(key.startswith("eval_") for key in logs):
        return "eval"
    if any(key.startswith("test_") for key in logs):
        return "test"
    return "train"


def clearml_scalars(logs: Mapping[str, Any]) -> list[tuple[str, float]]:
    """Map one Trainer log event to independent ClearML plot titles."""
    phase = _metric_phase(logs)
    phase_prefix = f"{phase}_"
    scalars: list[tuple[str, float]] = []
    for key, value in logs.items():
        if isinstance(value, bool) or not isinstance(value, Real):
            continue
        metric = key.removeprefix(phase_prefix)
        scalars.append((f"{phase}/{metric}", float(value)))
    return scalars


@torch.no_grad()
def global_weight_l2_norm(model: torch.nn.Module) -> float:
    parameters = [
        parameter.detach()
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    norm = torch.nn.utils.get_total_norm(parameters, norm_type=2.0)
    return float(norm.item())


class SeparatePlotsClearMLCallback(ClearMLCallback):
    """Preserve Transformers' ClearML lifecycle while separating scalar plots."""

    def __init__(self, *, weight_norm_interval_steps: int) -> None:
        super().__init__()
        self.weight_norm_interval_steps = validate_weight_norm_interval(
            weight_norm_interval_steps
        )
        self._last_weight_norm_step: int | None = None

    def _should_report_weight_norm(
        self,
        *,
        step: int,
        logs: Mapping[str, Any],
    ) -> bool:
        if self._last_weight_norm_step == step:
            return False
        if "train_loss" in logs:
            return True
        return "loss" in logs and (
            step == 1 or step % self.weight_norm_interval_steps == 0
        )

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        model: torch.nn.Module | None = None,
        processing_class: Any = None,
        logs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        metrics = dict(logs or {})
        step = int(state.global_step)
        if self._should_report_weight_norm(step=step, logs=metrics):
            if model is None:
                raise RuntimeError(
                    "Trainer did not provide a model for weight norm logging."
                )
            metrics["weight_norm"] = global_weight_l2_norm(model)
            self._last_weight_norm_step = step

        if not self._initialized:
            self.setup(args, state, model, processing_class, **kwargs)
        if not state.is_world_process_zero or self._clearml_task is None:
            return

        clearml_logger = self._clearml_task.get_logger()
        for title, value in clearml_scalars(metrics):
            clearml_logger.report_scalar(
                title=title,
                series="value",
                value=value,
                iteration=step,
            )
