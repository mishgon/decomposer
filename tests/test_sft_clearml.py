from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from training.sft import clearml_logging
from training.sft.clearml_logging import (
    SeparatePlotsClearMLCallback,
    clearml_scalars,
    global_weight_l2_norm,
    validate_weight_norm_interval,
)
from training.sft.train import _build_trainer_callbacks


class _RecordingLogger:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def report_scalar(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


class _FakeTask:
    def __init__(self, logger: _RecordingLogger) -> None:
        self.logger = logger

    def get_logger(self) -> _RecordingLogger:
        return self.logger


def _ready_callback(
    interval: int = 10,
) -> tuple[SeparatePlotsClearMLCallback, _RecordingLogger]:
    callback = SeparatePlotsClearMLCallback(weight_norm_interval_steps=interval)
    logger = _RecordingLogger()
    callback._initialized = True
    callback._clearml_task = _FakeTask(logger)
    return callback, logger


def test_clearml_scalars_use_independent_train_plots() -> None:
    assert clearml_scalars(
        {
            "loss": 0.5,
            "grad_norm": 1.25,
            "learning_rate": 2e-5,
            "train_runtime": 10.0,
            "ignored": "text",
            "flag": True,
        }
    ) == [
        ("train/loss", 0.5),
        ("train/grad_norm", 1.25),
        ("train/learning_rate", 2e-5),
        ("train/runtime", 10.0),
    ]


def test_clearml_scalars_use_eval_phase_for_whole_event() -> None:
    assert clearml_scalars(
        {
            "eval_loss": 0.4,
            "eval_runtime": 2.0,
            "epoch": 1.0,
            "num_input_tokens_seen": 100,
        }
    ) == [
        ("eval/loss", 0.4),
        ("eval/runtime", 2.0),
        ("eval/epoch", 1.0),
        ("eval/num_input_tokens_seen", 100.0),
    ]


def test_global_weight_l2_norm_uses_all_trainable_parameters() -> None:
    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[3.0, 4.0]]))
        model.bias.copy_(torch.tensor([12.0]))
    assert global_weight_l2_norm(model) == pytest.approx(13.0)


def test_callback_samples_weight_norm_at_first_tenth_and_final_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback, logger = _ready_callback()
    model = torch.nn.Linear(1, 1)
    norm_calls: list[torch.nn.Module] = []

    def fake_norm(received_model: torch.nn.Module) -> float:
        norm_calls.append(received_model)
        return 42.0

    monkeypatch.setattr(clearml_logging, "global_weight_l2_norm", fake_norm)
    for step, logs in (
        (1, {"loss": 1.0}),
        (2, {"loss": 0.9}),
        (10, {"loss": 0.8}),
        (10, {"eval_loss": 0.7}),
        (11, {"train_loss": 0.75}),
        (11, {"train_loss": 0.75}),
    ):
        callback.on_log(
            args=SimpleNamespace(),
            state=SimpleNamespace(global_step=step, is_world_process_zero=True),
            control=SimpleNamespace(),
            model=model,
            logs=logs,
        )

    assert norm_calls == [model, model, model]
    weight_calls = [
        call for call in logger.calls if call["title"] == "train/weight_norm"
    ]
    assert [call["iteration"] for call in weight_calls] == [1, 10, 11]
    assert {call["series"] for call in logger.calls} == {"value"}
    assert not {"train", "eval"} & {call["title"] for call in logger.calls}


def test_nonzero_rank_computes_norm_without_reporting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback, logger = _ready_callback()
    norm_calls = 0

    def fake_norm(_: torch.nn.Module) -> float:
        nonlocal norm_calls
        norm_calls += 1
        return 1.0

    monkeypatch.setattr(clearml_logging, "global_weight_l2_norm", fake_norm)
    callback.on_log(
        args=SimpleNamespace(),
        state=SimpleNamespace(global_step=1, is_world_process_zero=False),
        control=SimpleNamespace(),
        model=torch.nn.Linear(1, 1),
        logs={"loss": 1.0},
    )

    assert norm_calls == 1
    assert logger.calls == []


@pytest.mark.parametrize("value", [None, True, 0, -1, 1.5, "10"])
def test_weight_norm_interval_must_be_positive_integer(value: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        validate_weight_norm_interval(value)


def test_clearml_callback_is_absent_when_logging_is_disabled() -> None:
    sentinel = object()
    assert _build_trainer_callbacks(
        {"enabled": False, "weight_norm_interval_steps": 10},
        sentinel,
    ) == [sentinel]


def test_clearml_callback_coexists_with_early_stopping() -> None:
    sentinel = object()
    callbacks = _build_trainer_callbacks(
        {"enabled": True, "weight_norm_interval_steps": 10},
        sentinel,
    )
    assert callbacks[0] is sentinel
    assert isinstance(callbacks[1], SeparatePlotsClearMLCallback)
