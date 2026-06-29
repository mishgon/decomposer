"""Tests for vLLM process command/env construction."""

from __future__ import annotations

import os

from claw_eval.runner.vllm_process import (
    apply_vllm_env,
    make_vllm_server_spec,
    make_vllm_subprocess_env,
)


def test_make_vllm_server_spec_executor_command_includes_reasoning_parser(monkeypatch):
    monkeypatch.setenv("VLLM_MAX_MODEL_LEN", "65536")
    spec = make_vllm_server_spec(
        role="executor",
        model_id="Qwen/Qwen3.5-4B",
        port=8001,
        gpu="1",
        host="127.0.0.1",
        gpu_memory_utilization="0.9",
        extra_args=["--dtype", "auto"],
    )

    cmd = spec.command()
    assert "--served-model-name" in cmd
    assert cmd[cmd.index("--served-model-name") + 1] == "Qwen/Qwen3.5-4B"
    assert cmd[cmd.index("--max-model-len") + 1] == "65536"
    assert cmd[cmd.index("--gpu-memory-utilization") + 1] == "0.9"
    assert "--enable-auto-tool-choice" in cmd
    assert cmd[cmd.index("--tool-call-parser") + 1] == "hermes"
    assert cmd[cmd.index("--reasoning-parser") + 1] == "deepseek_r1"
    assert cmd[cmd.index("--gdn-prefill-backend") + 1] == "triton"
    assert cmd[-2:] == ["--dtype", "auto"]


def test_make_vllm_server_spec_manager_command_includes_reasoning_parser():
    spec = make_vllm_server_spec(
        role="decomposer",
        model_id="Qwen/Qwen3.6-27B",
        port=8000,
        gpu="0",
        host="127.0.0.1",
    )

    cmd = spec.command()
    assert cmd[cmd.index("--tool-call-parser") + 1] == "hermes"
    assert cmd[cmd.index("--reasoning-parser") + 1] == "deepseek_r1"
    assert cmd[cmd.index("--gdn-prefill-backend") + 1] == "triton"


def test_make_vllm_server_spec_does_not_duplicate_user_vllm_options():
    spec = make_vllm_server_spec(
        role="model",
        model_id="Qwen/Qwen3.5-122B-A10B",
        port=8000,
        gpu="0",
        host="127.0.0.1",
        extra_args=[
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "hermes",
            "--reasoning-parser=deepseek_r1",
            "--gdn-prefill-backend=triton",
        ],
    )

    cmd = spec.command()
    assert cmd.count("--enable-auto-tool-choice") == 1
    assert cmd.count("--tool-call-parser") == 1
    assert cmd.count("--reasoning-parser") == 0
    assert cmd.count("--reasoning-parser=deepseek_r1") == 1
    assert cmd.count("--gdn-prefill-backend") == 0
    assert cmd.count("--gdn-prefill-backend=triton") == 1


def test_apply_vllm_env_sets_role_specific_vars(monkeypatch):
    for key in [
        "VLLM_ENABLED",
        "VLLM_DECOMPOSER_BASE_URL",
        "VLLM_DECOMPOSER_MODEL_ID",
        "VLLM_DECOMPOSER_API_KEY",
        "VLLM_EXECUTOR_BASE_URL",
        "VLLM_EXECUTOR_MODEL_ID",
        "VLLM_EXECUTOR_API_KEY",
    ]:
        monkeypatch.setenv(key, "")

    specs = [
        make_vllm_server_spec(
            role="decomposer",
            model_id="manager-model",
            port=8000,
            gpu="0",
            host="127.0.0.1",
        ),
        make_vllm_server_spec(
            role="executor",
            model_id="executor-model",
            port=8001,
            gpu="1",
            host="127.0.0.1",
        ),
    ]

    apply_vllm_env(specs)
    assert os.environ["VLLM_ENABLED"] == "1"
    assert os.environ["VLLM_DECOMPOSER_BASE_URL"] == "http://127.0.0.1:8000/v1"
    assert os.environ["VLLM_DECOMPOSER_MODEL_ID"] == "manager-model"
    assert os.environ["VLLM_EXECUTOR_BASE_URL"] == "http://127.0.0.1:8001/v1"
    assert os.environ["VLLM_EXECUTOR_MODEL_ID"] == "executor-model"


def test_apply_vllm_env_overwrites_flat_model_role_vars(monkeypatch):
    monkeypatch.setenv("VLLM_MODEL_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("VLLM_MODEL_ID", "stale-model")
    monkeypatch.setenv("VLLM_MODEL_API_KEY", "stale-key")

    spec = make_vllm_server_spec(
        role="model",
        model_id="flat-model",
        port=8010,
        gpu="0",
        host="127.0.0.1",
        api_key="fresh-key",
    )

    apply_vllm_env([spec])

    assert os.environ["VLLM_BASE_URL"] == "http://127.0.0.1:8010/v1"
    assert os.environ["VLLM_MODEL_BASE_URL"] == "http://127.0.0.1:8010/v1"
    assert os.environ["VLLM_MODEL_ID"] == "flat-model"
    assert os.environ["VLLM_MODEL_API_KEY"] == "fresh-key"


def test_make_vllm_subprocess_env_disables_python_int_digit_limit(monkeypatch):
    monkeypatch.delenv("PYTHONINTMAXSTRDIGITS", raising=False)
    spec = make_vllm_server_spec(
        role="executor",
        model_id="Qwen/Qwen3.5-0.8B",
        port=8001,
        gpu="1",
        host="127.0.0.1",
    )

    env = make_vllm_subprocess_env(spec)

    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env["PYTHONINTMAXSTRDIGITS"] == "0"
