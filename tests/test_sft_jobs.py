from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sft import train as train_module

from sft.experiments import (
    INSTANCE_TYPES_BY_NUM_GPUS,
    ExperimentConfig,
    build_train_command,
    collect_experiments,
    has_experiment_artifacts,
    has_training_artifacts,
)
from sft.run_train_jobs import (
    _archive_output_dir,
    _build_job_script,
    _latest_checkpoint,
    _require_latest_checkpoint,
    _validate_clearml_config,
)
from sft.train import (
    _attention_backend_runtime,
    _build_early_stopping_callback,
    _build_parser,
    _configure_sdpa_backends,
    _has_existing_run_output,
    _load_prepared_split,
    _qwen35_linear_attention_runtime,
    _validate_required_runtime_profile,
    _resolve_config,
    _resolve_train_batch_config,
    _summarize_trainer_state,
)
from sft.qwen35_fast_runtime import (
    EXPECTED_RUNTIME,
    HF_FA2_IMPLEMENTATION,
    PROFILE_NAME as QWEN35_FAST_PROFILE,
)

EARLY_STOPPING_TRAINING_CONFIG = {
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
}


def _prepared_loader_record(
    index: int,
    *,
    metrics: dict[str, float],
    padding: str = "",
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "id": f"record-{index}",
        "group_id": f"group-{index}",
        "messages": [
            {"role": "system", "content": "System."},
            {"role": "user", "content": "Task."},
            {
                "role": "assistant",
                "content": "Done.",
                "teacher_reasoning": "Hidden reasoning.",
            },
        ],
        "tools": [],
        "source": {
            "adapter": "fixture",
            "adapter_version": 1,
            "source_id": "fixture",
            "benchmark": "fixture",
            "environment": "fixture",
            "partition": "train",
            "teacher": "fixture",
            "task_id": str(index),
            "rollout_id": "0",
        },
        "outcome": {"success": True, "reward": 1.0, "metrics": metrics},
        "attributes": {"padding": padding},
    }


def _experiment(**overrides: object) -> ExperimentConfig:
    return ExperimentConfig(
        name="smoke",
        config_path="sft/configs/smoke.yaml",
        description="SFT smoke",
        **overrides,
    )


def test_sft_experiments_are_unique() -> None:
    experiments = collect_experiments()
    assert len({experiment.name for experiment in experiments}) == len(experiments)
    assert INSTANCE_TYPES_BY_NUM_GPUS[2] == "a100plus.2gpu.80vG.24C.488G"
    assert INSTANCE_TYPES_BY_NUM_GPUS[4] == "a100plus.4gpu.80vG.48C.976G"


def test_prepared_loader_preserves_dynamic_json_across_reader_chunks(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mixed.jsonl"
    padding = "x" * 13_000
    records = [
        _prepared_loader_record(index, metrics={"reward": 1.0}, padding=padding)
        for index in range(900)
    ]
    records.append(
        _prepared_loader_record(
            900,
            metrics={
                "reward": 1.0,
                "binary_pass": 1.0,
                "check_passed": 9.0,
                "check_total": 10.0,
                "check_pass_ratio": 0.9,
            },
        )
    )
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")

    dataset = _load_prepared_split(
        path,
        sample_limit=None,
        num_proc=1,
    )

    assert len(dataset) == 901
    assert dataset[0]["outcome"]["metrics"] == {"reward": 1.0}
    assert dataset[-1]["outcome"]["metrics"] == records[-1]["outcome"]["metrics"]
    assert "teacher_reasoning" not in dataset[0]["messages"][-1]
    assert "reasoning" not in dataset[0]["messages"][-1]


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ('{"schema_version": 1}\n{not-json}\n', r"line 2"),
        ('{"schema_version": 1}\n[]\n', r"line 2 must be an object"),
        ("\n", r"split is empty"),
    ],
)
def test_prepared_loader_rejects_invalid_jsonl(
    tmp_path: Path, content: str, match: str
) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        _load_prepared_split(
            path,
            sample_limit=None,
            num_proc=1,
        )


def test_build_train_command_uses_torchrun_and_explicit_liger_mode() -> None:
    native = _experiment()
    liger = _experiment(use_liger_kernel=True)
    native_command = build_train_command(
        native, workdir="/staged", output_dir="/artifacts/native"
    )
    liger_command = build_train_command(
        liger, workdir="/staged", output_dir="/artifacts/liger"
    )
    assert native_command[:3] == ["torchrun", "--standalone", "--nproc-per-node=2"]
    assert "/staged/sft/configs/smoke.yaml" in native_command
    assert "--clearml" in native_command
    assert "--no-use-liger-kernel" in native_command
    assert "--use-liger-kernel" in liger_command


def test_build_train_command_can_resume_an_explicit_checkpoint() -> None:
    native = _experiment()
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


def test_job_script_adds_qwen35_runtime_overlays_and_fail_closed_environment() -> None:
    script = _build_job_script(
        ["torchrun", "--standalone", "train.py"],
        workdir=Path("/staged"),
        venv=Path("/venv"),
        triton_cache=Path("/cache/triton"),
        output_dir=Path("/artifacts/run"),
        runtime_bundle=Path("/runtime/qwen35"),
    )
    assert 'PYTHONPATH="/runtime/qwen35/triton:/runtime/qwen35/causal:' in script
    assert "export FLA_TILELANG=0" in script
    assert "DECOMPOSER_SFT_RUNTIME_PROFILE=qwen35-hf-fa2-fla-v1" in script
    assert "DECOMPOSER_SFT_RUNTIME_BUNDLE=/runtime/qwen35" in script


def test_force_archive_is_atomic_and_recoverable(tmp_path: Path) -> None:
    output = tmp_path / "jobs" / "experiment"
    output.mkdir(parents=True)
    (output / "training_summary.json").write_text("{}")
    archived = _archive_output_dir(
        output,
        archive_root=tmp_path / "jobs" / "_archive" / "experiment",
        commit="1234567890abcdef",
        timestamp=datetime(2026, 8, 31, 12, 0, tzinfo=UTC),
    )
    assert archived == (
        tmp_path
        / "jobs/_archive/experiment/20260831T120000Z-1234567890ab"
    )
    assert not output.exists()
    assert (archived / "training_summary.json").is_file()


def test_force_archive_ignores_missing_or_empty_output(tmp_path: Path) -> None:
    output = tmp_path / "empty"
    assert (
        _archive_output_dir(
            output,
            archive_root=tmp_path / "archive",
            commit="abc",
        )
        is None
    )
    output.mkdir()
    assert (
        _archive_output_dir(
            output,
            archive_root=tmp_path / "archive",
            commit="abc",
        )
        is None
    )


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


def test_benchmark_cli_can_override_prepared_dataset_paths() -> None:
    args = _build_parser().parse_args(
        [
            "--config",
            "unused.yaml",
            "--train-file",
            "/candidate/train.jsonl",
            "--validation-file",
            "/candidate/validation.jsonl",
            "--manifest-file",
            "/candidate/manifest.json",
        ]
    )
    resolved = _resolve_config(
        {
            "model": {},
            "data": {},
            "training": {},
            "clearml": {},
            "run": {},
        },
        args,
    )
    assert resolved["data"] == {
        "train_file": "/candidate/train.jsonl",
        "validation_file": "/candidate/validation.jsonl",
        "manifest_file": "/candidate/manifest.json",
        "error_on_truncation": False,
    }


def test_qwen35_linear_attention_runtime_reports_bound_implementations() -> None:
    runtime = _qwen35_linear_attention_runtime()
    assert set(runtime["packages"]) == {
        "flash-linear-attention",
        "fla-core",
        "causal-conv1d",
    }
    assert isinstance(runtime["fast_path_available"], bool)
    assert {
        "causal_conv1d",
        "causal_conv1d_update",
        "chunk_gated_delta_rule",
        "fused_recurrent_gated_delta_rule",
        "gated_rms_norm",
    } <= runtime.keys()


def test_attention_backend_runtime_reports_sdpa_callable_provenance() -> None:
    runtime = _attention_backend_runtime("sdpa", resolved_implementation="sdpa")
    assert runtime["requested_implementation"] == "sdpa"
    assert runtime["resolved_implementation"] == "sdpa"
    assert runtime["packages"]["torch"] == torch.__version__
    interface = runtime["attention_interface"]
    assert interface["identity"].endswith("sdpa_attention_forward")
    assert interface["module_file"]["exists"] is True
    assert len(interface["module_file"]["sha256"]) == 64


def test_required_qwen35_runtime_profile_validates_actual_bound_kernels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FLA_TILELANG", "0")
    monkeypatch.setenv("DECOMPOSER_SFT_RUNTIME_PROFILE", QWEN35_FAST_PROFILE)
    monkeypatch.setenv("DECOMPOSER_SFT_RUNTIME_BUNDLE", "/runtime/qwen35")
    monkeypatch.setattr(
        train_module,
        "validate_runtime_bundle",
        lambda *_args, **_kwargs: {"profile": QWEN35_FAST_PROFILE},
    )
    monkeypatch.setattr(
        train_module,
        "_optional_distribution_version",
        lambda name: EXPECTED_RUNTIME["packages"].get(name),
    )
    monkeypatch.setattr(
        train_module,
        "_qwen35_linear_attention_runtime",
        lambda: {
            "fast_path_available": True,
            "causal_conv1d": "causal_conv1d.interface.causal_conv1d_fn",
            "causal_conv1d_update": "causal_conv1d.interface.causal_conv1d_update",
            "chunk_gated_delta_rule": "fla.ops.gated_delta_rule.chunk",
            "fused_recurrent_gated_delta_rule": "fla.ops.gated_delta_rule.recurrent",
            "gated_rms_norm": "fla.modules.FusedRMSNormGated",
        },
    )
    monkeypatch.setattr(
        train_module,
        "_attention_backend_runtime",
        lambda *_args, **_kwargs: {
            "resolved_callables": {
                "flash_attn_func": {
                    "identity": "_flash_attn2_cuda_f12afc9.flash_attn_func"
                }
            }
        },
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "NVIDIA H100")

    runtime = _validate_required_runtime_profile(
        {"required_runtime_profile": QWEN35_FAST_PROFILE},
        {"attn_implementation": HF_FA2_IMPLEMENTATION},
    )
    assert runtime is not None
    assert runtime["profile"] == QWEN35_FAST_PROFILE
    assert runtime["cuda_capability"] == "9.0"


def test_required_qwen35_runtime_profile_rejects_tilelang_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FLA_TILELANG", raising=False)
    with pytest.raises(RuntimeError, match="FLA_TILELANG=0"):
        _validate_required_runtime_profile(
            {"required_runtime_profile": QWEN35_FAST_PROFILE},
            {"attn_implementation": HF_FA2_IMPLEMENTATION},
        )


@pytest.mark.parametrize(
    ("flag", "expected_strategy", "expected_length_column"),
    [
        ("--group-by-length", "group_by_length", "_token_length"),
        ("--no-group-by-length", "random", None),
    ],
)
def test_benchmark_group_by_length_uses_transformers_5_sampling_api(
    flag: str,
    expected_strategy: str,
    expected_length_column: str | None,
) -> None:
    args = _build_parser().parse_args(["--config", "unused.yaml", flag])
    resolved = _resolve_config(
        {
            "model": {},
            "data": {},
            "training": {},
            "clearml": {},
            "run": {},
        },
        args,
    )
    training = resolved["training"]
    assert training["train_sampling_strategy"] == expected_strategy
    assert training.get("length_column_name") == expected_length_column


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


def test_training_completion_requires_summary_and_final_weights(tmp_path: Path) -> None:
    output = tmp_path / "run"
    (output / "final").mkdir(parents=True)
    (output / "training_summary.json").write_text("{}")
    assert not has_training_artifacts(output)
    (output / "final" / "model.safetensors").write_bytes(b"weights")
    assert has_training_artifacts(output)


def test_training_completion_accepts_sharded_safetensors_index(tmp_path: Path) -> None:
    output = tmp_path / "run"
    (output / "final").mkdir(parents=True)
    (output / "training_summary.json").write_text("{}")
    (output / "final" / "model.safetensors.index.json").write_text("{}")
    assert has_training_artifacts(output)


def test_benchmark_experiment_completion_uses_benchmark_summary(tmp_path: Path) -> None:
    smoke = _experiment(benchmark=True)
    output = tmp_path / "smoke"
    output.mkdir()
    assert not has_experiment_artifacts(smoke, output)
    (output / "benchmark_summary.json").write_text("{}")
    assert has_experiment_artifacts(smoke, output)
