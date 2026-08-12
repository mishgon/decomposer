from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from datasets import Dataset
from transformers import GenerationConfig

from training.sft.experiments import (
    INSTANCE_TYPES_BY_NUM_GPUS,
    build_train_command,
    collect_experiments,
    has_training_artifacts,
)
from training.sft.run_train_jobs import (
    _build_job_script,
    _latest_checkpoint,
    _require_latest_checkpoint,
    _validate_clearml_config,
)
from training.sft.train import (
    _apply_overlength_policy,
    _build_early_stopping_callback,
    _configure_gemma4_generation,
    _configure_sdpa_backends,
    _has_existing_run_output,
    _resolve_train_batch_config,
    _save_final_configuration,
    _summarize_trainer_state,
)

EARLY_STOPPING_TRAINING_CONFIG = {
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
}


class _FakeGemmaTokenizer:
    bos_token_id = 2
    eos_token_id = 1
    pad_token_id = 0
    unk_token_id = 3

    def __init__(self, token_ids: dict[str, int] | None = None) -> None:
        self.token_ids = token_ids or {
            "<turn|>": 106,
            "<|tool_response>": 50,
        }

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.token_ids.get(token, self.unk_token_id)

    def convert_ids_to_tokens(self, token_id: int) -> str:
        for token, candidate_id in self.token_ids.items():
            if candidate_id == token_id:
                return token
        return "<unk>"

    def save_pretrained(self, path: str | Path) -> None:
        (Path(path) / "tokenizer_config.json").write_text("{}")


class _FakeModelConfig:
    def save_pretrained(self, path: str | Path) -> None:
        (Path(path) / "config.json").write_text("{}")


def test_sft_experiments_are_unique_and_register_retained_configs() -> None:
    experiments = collect_experiments()
    assert len({experiment.name for experiment in experiments}) == len(experiments)
    assert INSTANCE_TYPES_BY_NUM_GPUS[2] == "a100plus.2gpu.80vG.24C.488G"
    assert INSTANCE_TYPES_BY_NUM_GPUS[4] == "a100plus.4gpu.80vG.48C.976G"
    assert {experiment.use_liger_kernel for experiment in experiments[:2]} == {
        False,
        True,
    }
    assert len(experiments) == 4
    e2b_four_gpu = experiments[2]
    assert e2b_four_gpu.num_gpus == 4
    assert e2b_four_gpu.use_liger_kernel is True
    assert e2b_four_gpu.pytorch_cuda_alloc_conf == "expandable_segments:True"
    e4b_gb4 = experiments[3]
    assert e4b_gb4.num_gpus == 4
    assert e4b_gb4.use_liger_kernel is True
    assert e4b_gb4.pytorch_cuda_alloc_conf == "expandable_segments:True"
    assert {experiment.name for experiment in experiments[-2:]} == {
        "gemma4-e2b-nonthinking-4gpu-liger-workplace-26b-v3",
        "gemma4-e4b-nonthinking-4gpu-liger-workplace-26b-v3",
    }


def test_build_train_command_uses_torchrun_and_explicit_liger_mode() -> None:
    native, liger = collect_experiments()[:2]
    native_command = build_train_command(
        native, workdir="/staged", output_dir="/artifacts/native"
    )
    liger_command = build_train_command(
        liger, workdir="/staged", output_dir="/artifacts/liger"
    )
    assert native_command[:3] == ["torchrun", "--standalone", "--nproc-per-node=2"]
    assert "/staged/training/sft/configs/gemma4_e2b_smoke.yaml" in native_command
    assert "--clearml" in native_command
    assert "--no-use-liger-kernel" in native_command
    assert "--use-liger-kernel" in liger_command


def test_build_train_command_can_resume_an_explicit_checkpoint() -> None:
    native = collect_experiments()[2]
    command = build_train_command(
        native,
        workdir="/staged",
        output_dir="/artifacts/native",
        resume_from_checkpoint="/artifacts/native/checkpoint-458",
    )
    assert command[-2:] == [
        "--resume-from-checkpoint",
        "/artifacts/native/checkpoint-458",
    ]


def test_job_script_captures_console_log_without_hiding_training_failure() -> None:
    script = _build_job_script(
        ["torchrun", "--standalone", "train.py"],
        workdir=Path("/staged"),
        venv=Path("/venv"),
        triton_cache=Path("/cache/triton"),
        output_dir=Path("/artifacts/run"),
    )
    assert "mkdir -p /artifacts/run" in script
    assert 'PYTHONPATH="$WORKDIR/src:$WORKDIR:' in script
    assert "bash -o pipefail -c" in script
    assert "/venv/bin/torchrun --standalone train.py" in script
    assert "tee -a /artifacts/run/console.log" in script


def test_clearml_config_must_be_private(tmp_path: Path) -> None:
    config = tmp_path / "clearml.conf"
    config.write_text("api {}")
    config.chmod(0o600)
    _validate_clearml_config(config)
    config.chmod(0o644)
    with pytest.raises(PermissionError, match="expected 0600 or stricter"):
        _validate_clearml_config(config)


def test_launcher_logs_do_not_trigger_existing_run_guard(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / "console.log").write_text("starting")
    (output / "mlspace.log").write_text("pending")
    assert not _has_existing_run_output(output)
    (output / "resolved_config.json").write_text("{}")
    assert _has_existing_run_output(output)


def test_global_batch_derives_gradient_accumulation() -> None:
    resolved, runtime = _resolve_train_batch_config(
        {
            "global_batch_size": 8,
            "per_device_train_batch_size": 1,
        },
        world_size=4,
    )
    assert resolved == {
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 2,
    }
    assert runtime == {
        "global_batch_size": 8,
        "world_size": 4,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 2,
    }


def test_global_batch_derives_e4b_gb4() -> None:
    resolved, runtime = _resolve_train_batch_config(
        {
            "global_batch_size": 4,
            "per_device_train_batch_size": 1,
        },
        world_size=4,
    )
    assert resolved["gradient_accumulation_steps"] == 1
    assert runtime["global_batch_size"] == 4


@pytest.mark.parametrize(
    ("training_config", "world_size", "message"),
    [
        (
            {
                "global_batch_size": 8,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 2,
            },
            4,
            "is derived.*must not be specified",
        ),
        (
            {
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 2,
            },
            4,
            "is derived.*must not be specified",
        ),
        (
            {"per_device_train_batch_size": 1},
            4,
            "global_batch_size is required",
        ),
        (
            {"global_batch_size": 7, "per_device_train_batch_size": 1},
            4,
            "must be divisible",
        ),
        (
            {"global_batch_size": 0, "per_device_train_batch_size": 1},
            4,
            "positive integer",
        ),
        (
            {"global_batch_size": True, "per_device_train_batch_size": 1},
            4,
            "positive integer",
        ),
        (
            {"global_batch_size": 8, "per_device_train_batch_size": "1"},
            4,
            "positive integer",
        ),
        (
            {"global_batch_size": 8, "per_device_train_batch_size": 1},
            0,
            "positive integer",
        ),
    ],
)
def test_batch_resolution_rejects_invalid_values(
    training_config: dict[str, object],
    world_size: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _resolve_train_batch_config(training_config, world_size=world_size)


def test_gemma4_generation_preserves_multi_eos_and_saves_it(tmp_path: Path) -> None:
    tokenizer = _FakeGemmaTokenizer()
    generation_config = GenerationConfig(
        bos_token_id=99,
        eos_token_id=[1, 106, 50],
        pad_token_id=99,
    )
    runtime = _configure_gemma4_generation(
        generation_config,
        tokenizer=tokenizer,
    )
    assert runtime == {
        "bos_token_id": 2,
        "eos_token_id": [1, 106, 50],
        "pad_token_id": 0,
        "required_stop_token_ids": {
            "<turn|>": 106,
            "<|tool_response>": 50,
        },
    }
    _save_final_configuration(
        tmp_path,
        model_config=_FakeModelConfig(),
        tokenizer=tokenizer,
        generation_config=generation_config,
    )
    saved = GenerationConfig.from_pretrained(tmp_path)
    assert saved.bos_token_id == 2
    assert saved.eos_token_id == [1, 106, 50]
    assert saved.pad_token_id == 0
    assert (tmp_path / "config.json").is_file()
    assert (tmp_path / "tokenizer_config.json").is_file()
    assert (tmp_path / "generation_config.json").is_file()


def test_gemma4_generation_rejects_missing_required_stop_token() -> None:
    tokenizer = _FakeGemmaTokenizer({"<turn|>": 106})
    with pytest.raises(ValueError, match="tool_response.*not an exact"):
        _configure_gemma4_generation(
            GenerationConfig(eos_token_id=1),
            tokenizer=tokenizer,
        )


def _tokenized_dataset(*lengths: int) -> Dataset:
    return Dataset.from_dict(
        {
            "id": [f"example-{index}" for index in range(len(lengths))],
            "_token_length": list(lengths),
            "_supervised_tokens": [1] * len(lengths),
        }
    )


def test_overlength_policy_errors_by_default() -> None:
    with pytest.raises(ValueError, match="explicitly enable data.exclude_overlength"):
        _apply_overlength_policy(
            _tokenized_dataset(100, 35044),
            split="train",
            max_length=32768,
            exclude_overlength=False,
            error_on_truncation=True,
        )


def test_overlength_policy_explicitly_excludes_and_records_trace() -> None:
    filtered, excluded = _apply_overlength_policy(
        _tokenized_dataset(100, 35044, 200),
        split="train",
        max_length=32768,
        exclude_overlength=True,
        error_on_truncation=True,
    )
    assert filtered["id"] == ["example-0", "example-2"]
    assert excluded == [
        {
            "id": "example-1",
            "split": "train",
            "token_length": 35044,
            "max_length": 32768,
        }
    ]


def test_overlength_policy_refuses_to_empty_split() -> None:
    with pytest.raises(ValueError, match="emptied the validation split"):
        _apply_overlength_policy(
            _tokenized_dataset(40000),
            split="validation",
            max_length=32768,
            exclude_overlength=True,
            error_on_truncation=True,
        )


def test_configure_sdpa_backends_disables_only_cudnn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {
        "cudnn": True,
        "flash": True,
        "memory_efficient": True,
        "math": True,
    }
    monkeypatch.setattr(
        torch.backends.cuda,
        "enable_cudnn_sdp",
        lambda enabled: state.__setitem__("cudnn", enabled),
    )
    monkeypatch.setattr(
        torch.backends.cuda, "cudnn_sdp_enabled", lambda: state["cudnn"]
    )
    monkeypatch.setattr(
        torch.backends.cuda, "flash_sdp_enabled", lambda: state["flash"]
    )
    monkeypatch.setattr(
        torch.backends.cuda,
        "mem_efficient_sdp_enabled",
        lambda: state["memory_efficient"],
    )
    monkeypatch.setattr(torch.backends.cuda, "math_sdp_enabled", lambda: state["math"])

    assert _configure_sdpa_backends({"disable_cudnn_sdpa": True}) == {
        "cudnn": False,
        "flash": True,
        "memory_efficient": True,
        "math": True,
    }


def test_configure_sdpa_backends_preserves_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_enable(_: bool) -> None:
        raise AssertionError("The default policy must not change cuDNN SDPA.")

    monkeypatch.setattr(torch.backends.cuda, "enable_cudnn_sdp", unexpected_enable)
    monkeypatch.setattr(torch.backends.cuda, "cudnn_sdp_enabled", lambda: True)
    monkeypatch.setattr(torch.backends.cuda, "flash_sdp_enabled", lambda: False)
    monkeypatch.setattr(torch.backends.cuda, "mem_efficient_sdp_enabled", lambda: True)
    monkeypatch.setattr(torch.backends.cuda, "math_sdp_enabled", lambda: True)

    assert _configure_sdpa_backends({}) == {
        "cudnn": True,
        "flash": False,
        "memory_efficient": True,
        "math": True,
    }


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_configure_sdpa_backends_requires_boolean(value: object) -> None:
    with pytest.raises(ValueError, match="must be a boolean"):
        _configure_sdpa_backends({"disable_cudnn_sdpa": value})


def test_latest_checkpoint_is_numeric_and_requires_trainer_state(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    for step in (9, 100):
        checkpoint = output / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "trainer_state.json").write_text("{}")
    (output / "checkpoint-1000").mkdir()
    (output / "checkpoint-invalid").mkdir()

    assert _latest_checkpoint(output) == output / "checkpoint-100"
    assert _latest_checkpoint(tmp_path / "missing") is None
    with pytest.raises(FileNotFoundError, match="No complete checkpoint-N"):
        _require_latest_checkpoint(tmp_path / "missing")


def test_early_stopping_callback_uses_epoch_patience() -> None:
    callback = _build_early_stopping_callback(
        {"early_stopping": {"patience": 2, "threshold": 0.0}},
        EARLY_STOPPING_TRAINING_CONFIG,
    )
    assert callback is not None
    assert callback.early_stopping_patience == 2
    assert callback.early_stopping_threshold == 0.0
    assert _build_early_stopping_callback({}, EARLY_STOPPING_TRAINING_CONFIG) is None


@pytest.mark.parametrize(
    ("early_stopping", "message"),
    [
        ({"patience": 0}, "positive integer"),
        ({"patience": 2, "threshold": -0.1}, "non-negative"),
        ({"patience": 2, "unknown": True}, "Unknown"),
    ],
)
def test_early_stopping_rejects_invalid_config(
    early_stopping: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _build_early_stopping_callback(
            {"early_stopping": early_stopping},
            EARLY_STOPPING_TRAINING_CONFIG,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("save_strategy", "steps", "matching"),
        ("load_best_model_at_end", False, "load_best_model_at_end"),
        ("metric_for_best_model", "accuracy", "eval_loss"),
        ("greater_is_better", True, "greater_is_better"),
    ],
)
def test_early_stopping_requires_compatible_training_config(
    field: str,
    value: object,
    message: str,
) -> None:
    training_config = {**EARLY_STOPPING_TRAINING_CONFIG, field: value}
    with pytest.raises(ValueError, match=message):
        _build_early_stopping_callback(
            {"early_stopping": {"patience": 2}},
            training_config,
        )


def test_training_state_summary_reports_early_stop_and_best_checkpoint() -> None:
    trainer = SimpleNamespace(
        state=SimpleNamespace(
            global_step=687,
            max_steps=1145,
            epoch=3.0,
            best_metric=1.5,
            best_model_checkpoint="/artifacts/checkpoint-458",
        )
    )
    assert _summarize_trainer_state(trainer, early_stopping_enabled=True) == {
        "global_step": 687,
        "max_steps": 1145,
        "completed_epochs": 3.0,
        "best_metric": 1.5,
        "best_model_checkpoint": "/artifacts/checkpoint-458",
        "early_stopped": True,
    }


@pytest.mark.parametrize(
    ("config_name", "model_tag"),
    [
        ("gemma4_e2b_smoke.yaml", "gemma-4-E2B-it"),
        (
            "gemma4_e2b_nonthinking_4gpu_liger_workplace_26b_v3.yaml",
            "gemma-4-E2B-it",
        ),
        (
            "gemma4_e4b_nonthinking_4gpu_liger_workplace_26b_v3.yaml",
            "gemma-4-E4B-it",
        ),
    ],
)
def test_sft_configs_have_clearml_project_and_model_tags(
    config_name: str,
    model_tag: str,
) -> None:
    config_path = Path("training/sft/configs") / config_name
    config = yaml.safe_load(config_path.read_text())
    clearml = config["clearml"]
    assert clearml["project"] == "decomposer"
    assert clearml["weight_norm_interval_steps"] == 10
    assert clearml["tags"] == ["sft", model_tag, "workplace-assistant"]


def test_smoke_config_evaluates_clearml_metrics_after_one_step() -> None:
    config = yaml.safe_load(
        Path("training/sft/configs/gemma4_e2b_smoke.yaml").read_text()
    )
    assert config["training"]["max_steps"] == 1
    assert config["training"]["eval_strategy"] == "steps"
    assert config["training"]["eval_steps"] == 1
    assert config["training"]["global_batch_size"] == 2
    assert "gradient_accumulation_steps" not in config["training"]


def test_training_completion_requires_summary_and_final_weights(tmp_path: Path) -> None:
    output = tmp_path / "run"
    (output / "final").mkdir(parents=True)
    (output / "training_summary.json").write_text("{}")
    assert not has_training_artifacts(output)
    (output / "final" / "model.safetensors").write_bytes(b"weights")
    assert has_training_artifacts(output)


@pytest.mark.parametrize(
    (
        "config_name",
        "global_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
    ),
    [
        (
            "gemma4_e2b_nonthinking_4gpu_liger_workplace_26b_v3.yaml",
            8,
            2,
            2.0e-5,
        ),
        (
            "gemma4_e4b_nonthinking_4gpu_liger_workplace_26b_v3.yaml",
            4,
            1,
            1.0e-5,
        ),
    ],
)
def test_workplace_26b_v3_configs_use_cleaned_data_and_32k_exclusion(
    config_name: str,
    global_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
) -> None:
    config = yaml.safe_load((Path("training/sft/configs") / config_name).read_text())
    data = config["data"]
    training = config["training"]
    assert (
        "datasets/sft/decomposer-workplace-26b-a4b-nonthinking/v3" in data["train_file"]
    )
    assert data["exclude_overlength"] is True
    assert data["error_on_truncation"] is True
    assert training["max_length"] == 32768
    assert training["global_batch_size"] == global_batch_size
    assert "gradient_accumulation_steps" not in training
    resolved, _ = _resolve_train_batch_config(training, world_size=4)
    assert resolved["gradient_accumulation_steps"] == gradient_accumulation_steps
    assert training["learning_rate"] == learning_rate
    assert training["num_train_epochs"] == 5
    assert training["eval_strategy"] == "epoch"
    assert training["save_strategy"] == "epoch"
    assert training["save_total_limit"] == 2
    assert training["use_liger_kernel"] is True
    assert training["liger_kernel_config"] == {
        "fused_linear_cross_entropy": True,
        "cross_entropy": False,
        "rms_norm": False,
        "geglu": False,
        "rope": False,
        "layer_norm": False,
    }
    assert config["run"]["expected_world_size"] == 4
    assert config["run"]["disable_cudnn_sdpa"] is True
    assert config["run"]["early_stopping"] == {
        "patience": 2,
        "threshold": 0.0,
    }
