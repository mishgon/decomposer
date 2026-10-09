from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
import yaml
from datasets import Dataset

# Training runs FLA on its Triton kernels; its TileLang backend is not the validated path.
os.environ.setdefault("FLA_TILELANG", "0")

from sft.train import (
    _apply_overlength_policy,
    _benchmark_sample_manifest,
    _bf16_config,
    _build_early_stopping_callback,
    _build_parser,
    _enable_qwen35_native_gva,
    _evaluation_datasets,
    _freeze_modules,
    _resolve_config,
    _rewrite_weights_in_bf16,
    _select_longest_by_token_length,
    _select_stratified_by_environment_and_length,
    _step_timing_summary,
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
    assert resolved["data"]["error_on_truncation"] is False

    explicit = _resolve_config(
        {
            **sections,
            "data": {"error_on_truncation": True},
            "training": {"max_length": None},
        },
        args,
    )
    assert explicit["training"]["max_length"] is None
    assert explicit["data"]["error_on_truncation"] is True

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


def test_overlength_policy_refuses_with_error_on_truncation() -> None:
    with pytest.raises(ValueError, match="data.error_on_truncation: false"):
        _apply_overlength_policy(
            _tokenized_dataset(100, 35044),
            split="train",
            max_length=32768,
            exclude_overlength=False,
            error_on_truncation=True,
        )


def test_overlength_policy_explicitly_excludes_and_records_trace() -> None:
    filtered, excluded, truncated = _apply_overlength_policy(
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
    assert truncated == []


def test_overlength_policy_can_leave_traces_for_trl_to_cut() -> None:
    dataset = _tokenized_dataset(100, 35044)
    kept, excluded, truncated = _apply_overlength_policy(
        dataset,
        split="train",
        max_length=32768,
        exclude_overlength=False,
        error_on_truncation=False,
    )
    assert kept is dataset
    assert excluded == []
    assert truncated == [
        {
            "id": "example-1",
            "split": "train",
            "token_length": 35044,
            "max_length": 32768,
            "supervised_tokens": 35044,
            "supervised_tokens_kept": 32768,
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


def test_qwen35_unloop_v3_lora_configs_differ_only_in_length() -> None:
    root = Path("sft/configs")
    configs = {
        length: yaml.safe_load(
            (root / f"qwen35_4b_unloop_nonthinking_mixed_v3_lora_{length}_4gpu.yaml").read_text()
        )
        for length in ("16k", "32k")
    }
    for length, max_length in (("16k", 16384), ("32k", 32768)):
        config = configs[length]
        assert config["training"]["max_length"] == max_length
        assert config["training"]["output_dir"].endswith(f"mixed-v3-lora-{length}")
        assert config["data"]["expected_fingerprint"] == (
            "50a733439db998f83c559946eb5f57fc77c811916142b787bdf2245d1fdb8df4"
        )
        assert config["data"]["expected_system_prompt_profile"] == "teacher"
        assert config["lora"] == {"r": 32, "alpha": 64, "dropout": 0.05}
        assert config["training"]["fsdp"] is False
        assert _build_early_stopping_callback(config["run"], config["training"]) is not None

    def without_length(config: dict) -> str:
        text = yaml.safe_dump(config)
        for length in ("16k", "32k", "16K", "32K", "16384", "32768"):
            text = text.replace(length, "")
        return text

    assert without_length(configs["16k"]) == without_length(configs["32k"])
