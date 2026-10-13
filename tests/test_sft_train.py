from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset

# Training runs FLA on its Triton kernels; its TileLang backend is not the validated path.
os.environ.setdefault("FLA_TILELANG", "0")

from sft import train as train_module
from sft.qwen35_fast_runtime import (
    EXPECTED_RUNTIME,
    HF_FA2_IMPLEMENTATION,
    PROFILE_NAME as QWEN35_FAST_PROFILE,
)
from sft.train import (
    _attention_backend_runtime,
    _benchmark_sample_manifest,
    _bf16_config,
    _build_early_stopping_callback,
    _build_parser,
    _configure_sdpa_backends,
    _enable_qwen35_native_gva,
    _evaluation_datasets,
    _freeze_modules,
    _has_existing_run_output,
    _liger_compatible_loss_type,
    _overlength_truncations,
    _load_prepared_split,
    _qwen35_linear_attention_runtime,
    _resolve_config,
    _resolve_train_batch_config,
    _rewrite_weights_in_bf16,
    _select_longest_by_token_length,
    _select_stratified_by_environment_and_length,
    _step_timing_summary,
    _summarize_trainer_state,
    _validate_required_runtime_profile,
)


def _tiny_text_config(**overrides):
    from transformers import Qwen3_5TextConfig

    values = {
        "vocab_size": 64,
        "hidden_size": 128,
        "intermediate_size": 192,
        "num_hidden_layers": 2,
        "layer_types": ["linear_attention", "full_attention"],
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 64,
        "linear_key_head_dim": 64,
        "linear_value_head_dim": 64,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
    }
    return Qwen3_5TextConfig(**{**values, **overrides})


def _tiny_multimodal_config():
    from transformers import Qwen3_5Config

    return Qwen3_5Config(
        text_config=_tiny_text_config().to_dict(),
        vision_config={"depth": 1, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2, "out_hidden_size": 128},
    )


def test_step_timing_summary_skips_warmup_and_measures_imbalance() -> None:
    summary = _step_timing_summary(
        [30.0, 5.0, 5.0, 2.0, 2.0],
        [[1, 1], [1, 1], [1, 1], [100, 300], [200, 200]],
        skip=3,
    )
    assert summary["steady_tokens_per_second"] == 800 / 4.0
    # Largest-rank tokens (300 + 200) over mean-rank tokens ((400 + 400) / 2).
    assert summary["rank_imbalance_ratio"] == 500 / 400
    assert summary["step_seconds_p50"] == 2.0
    assert _step_timing_summary([1.0], [[5, 5]], skip=3) == {"steps": 1, "skipped": 3}


def test_freeze_modules_freezes_only_the_named_subtree() -> None:
    from transformers import Qwen3_5ForConditionalGeneration

    model = Qwen3_5ForConditionalGeneration(_tiny_multimodal_config())
    visual = sum(p.numel() for n, p in model.named_parameters() if n.startswith("model.visual."))

    assert _freeze_modules(model, ["model.visual"]) == visual
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad is not name.startswith("model.visual.")
    with pytest.raises(ValueError, match="matched no parameters"):
        _freeze_modules(model, ["model.vis"])


def test_rewrite_weights_in_bf16_casts_every_shard(tmp_path) -> None:
    from safetensors.torch import load_file
    from transformers import Qwen3_5ForConditionalGeneration

    torch.manual_seed(0)
    Qwen3_5ForConditionalGeneration(_tiny_multimodal_config()).save_pretrained(
        tmp_path, max_shard_size="200KB"
    )
    shards = sorted(tmp_path.glob("*.safetensors"))
    assert len(shards) > 1

    def load():
        tensors = {}
        for shard in shards:
            tensors.update(load_file(shard))
        return tensors

    original = load()
    assert {tensor.dtype for tensor in original.values()} == {torch.float32}
    assert _rewrite_weights_in_bf16(tmp_path) == len(original)
    rewritten = load()
    assert rewritten.keys() == original.keys()
    for name, tensor in original.items():
        torch.testing.assert_close(rewritten[name], tensor.to(torch.bfloat16), rtol=0, atol=0)
    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    assert index["metadata"]["total_size"] == sum(tensor.numel() * 2 for tensor in rewritten.values())


def test_bf16_config_records_bf16_in_every_sub_config(tmp_path) -> None:
    config = _tiny_multimodal_config()
    config.dtype = torch.float32  # as an fp32-weight run loads it
    for key in config.sub_configs:
        getattr(config, key).dtype = torch.float32

    _bf16_config(config).save_pretrained(tmp_path)

    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["dtype"] == "bfloat16"
    assert saved["text_config"]["dtype"] == saved["vision_config"]["dtype"] == "bfloat16"
    assert config.dtype == torch.float32  # the training model's config is left alone


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FLA kernels need CUDA")
def test_native_gva_matches_repeated_key_heads() -> None:
    pytest.importorskip("fla")
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    torch.manual_seed(0)
    layer = Qwen3_5GatedDeltaNet(_tiny_text_config(), layer_idx=0).cuda().to(torch.bfloat16)
    hidden = torch.randn(2, 96, 128, device="cuda", dtype=torch.bfloat16)

    def run():
        inputs = hidden.clone().requires_grad_(True)
        output = layer(inputs)
        output.float().square().mean().backward()
        return output.detach(), inputs.grad.detach()

    repeated_output, repeated_grad = run()
    assert _enable_qwen35_native_gva(layer) == 1
    assert layer.num_k_heads == layer.num_v_heads
    native_output, native_grad = run()
    torch.testing.assert_close(native_output, repeated_output, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(native_grad, repeated_grad, atol=2e-2, rtol=2e-2)


def test_stratified_benchmark_sample_preserves_environments_and_lengths() -> None:
    rows = []
    for environment, count in (("workplace", 6), ("gaia", 3), ("toolathlon", 3)):
        for index in range(count):
            rows.append(
                {
                    "id": f"{environment}-{index}",
                    "source": {"environment": environment},
                    "_token_length": (index + 1) * 100,
                    "_supervised_tokens": index + 1,
                }
            )
    selected = _select_stratified_by_environment_and_length(
        Dataset.from_list(rows),
        6,
    )
    environments = list(selected["source"])
    counts = {
        name: sum(source["environment"] == name for source in environments)
        for name in ("workplace", "gaia", "toolathlon")
    }
    assert counts == {"workplace": 3, "gaia": 2, "toolathlon": 1}
    assert len(set(selected["_token_length"])) > 2


def test_benchmark_sample_manifest_pins_order_and_lengths() -> None:
    rows = [
        {
            "id": f"record-{index}",
            "_token_length": 100 + index,
            "_supervised_tokens": 10 + index,
        }
        for index in range(3)
    ]
    sample = Dataset.from_list(rows)
    forward = _benchmark_sample_manifest(sample)
    repeated = _benchmark_sample_manifest(sample)
    reversed_sample = _benchmark_sample_manifest(sample.select([2, 1, 0]))
    assert forward == repeated
    assert forward["ordered_record_ids"] == [
        "record-0",
        "record-1",
        "record-2",
    ]
    assert len(forward["ordered_record_ids_sha256"]) == 64
    assert len(forward["ordered_records_sha256"]) == 64
    assert (
        forward["ordered_records_sha256"] != reversed_sample["ordered_records_sha256"]
    )


def test_config_cuts_at_16k_from_the_end_by_default() -> None:
    sections = {"model": {}, "data": {}, "training": {}, "clearml": {}, "run": {}}
    args = _build_parser().parse_args(["--config", "unused.yaml"])
    resolved = _resolve_config(sections, args)
    assert resolved["training"]["max_length"] == 16384

    explicit = _resolve_config({**sections, "training": {"max_length": None}}, args)
    assert explicit["training"]["max_length"] is None

    override = _resolve_config(
        sections,
        _build_parser().parse_args(["--config", "unused.yaml", "--max-length", "32768"]),
    )
    assert override["training"]["max_length"] == 32768


def test_evaluation_datasets_add_one_subset_per_gym() -> None:
    environments = ["tau2_gym", "wideseek", "tau2_gym", "workplace_assistant"]
    dataset = Dataset.from_list(
        [
            {"id": f"r{index}", "source": {"environment": environment}}
            for index, environment in enumerate(environments)
        ]
    )
    datasets = _evaluation_datasets(dataset)
    assert list(datasets) == ["all", "tau2_gym", "wideseek", "workplace_assistant"]
    assert datasets["all"] is dataset
    assert datasets["tau2_gym"]["id"] == ["r0", "r2"]
    assert datasets["wideseek"]["id"] == ["r1"]

    single = dataset.select([0, 2])
    assert list(_evaluation_datasets(single)) == ["all"]


def _tokenized_dataset(*lengths: int) -> Dataset:
    return Dataset.from_dict(
        {
            "id": [f"example-{index}" for index in range(len(lengths))],
            "_token_length": list(lengths),
            "_supervised_tokens": list(lengths),
            "assistant_masks": [[1] * length for length in lengths],
        }
    )


def test_overlength_truncations_report_the_supervised_tokens_kept() -> None:
    assert _overlength_truncations(
        _tokenized_dataset(100, 35044, 200), split="train", max_length=32768
    ) == [
        {
            "id": "example-1",
            "split": "train",
            "token_length": 35044,
            "max_length": 32768,
            "supervised_tokens": 35044,
            "supervised_tokens_kept": 32768,
        }
    ]


def test_overlength_truncations_are_empty_within_or_without_a_limit() -> None:
    dataset = _tokenized_dataset(100, 35044)
    assert _overlength_truncations(dataset, split="validation", max_length=40000) == []
    assert _overlength_truncations(dataset, split="validation", max_length=None) == []


def test_longest_sample_selection_happens_after_tokenization() -> None:
    dataset = _tokenized_dataset(100, 400, 200, 400, 300)
    selected = _select_longest_by_token_length(dataset, 4)
    assert selected["id"] == ["example-1", "example-3", "example-4", "example-2"]
    assert selected["_token_length"] == [400, 400, 300, 200]
    assert _select_longest_by_token_length(dataset, None) is dataset


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_longest_sample_selection_requires_positive_integer(limit: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        _select_longest_by_token_length(_tokenized_dataset(100), limit)  # type: ignore[arg-type]


def test_liger_uses_fused_ce_instead_of_chunked_nll() -> None:
    assert _liger_compatible_loss_type(True, "chunked_nll") is None
    assert _liger_compatible_loss_type(False, "chunked_nll") == "chunked_nll"
    assert _liger_compatible_loss_type(True, "custom") == "custom"


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


def test_global_batch_equal_to_world_size_needs_no_accumulation() -> None:
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
