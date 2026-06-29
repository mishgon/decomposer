"""Tests for vLLM config resolution."""

from __future__ import annotations

import os

import pytest

from claw_eval.config import Config, collect_vllm_wait_targets, load_config


@pytest.fixture(autouse=True)
def _clean_vllm_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("VLLM_") or key.startswith("LOCALINFERENCE_"):
            monkeypatch.delenv(key, raising=False)
    # load_config() reads .env; blank these so machine-local role settings do not
    # leak into unit tests. Individual tests can still override them.
    for key in [
        "VLLM_DECOMPOSER_BASE_URL",
        "VLLM_DECOMPOSER_MODEL_ID",
        "VLLM_MODEL_BASE_URL",
        "VLLM_MODEL_MODEL_ID",
        "VLLM_EXECUTOR_BASE_URL",
        "VLLM_EXECUTOR_MODEL_ID",
        "VLLM_JUDGE_BASE_URL",
        "VLLM_JUDGE_MODEL_ID",
        "VLLM_USER_AGENT_BASE_URL",
        "VLLM_USER_AGENT_MODEL_ID",
    ]:
        monkeypatch.setenv(key, "")


def test_vllm_overrides_local_inference(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALINFERENCE_API_KEY", "remote-key")
    monkeypatch.setenv("LOCALINFERENCE_BASE_URL", "https://remote.example/v1")
    monkeypatch.setenv("LOCALINFERENCE_MODEL_ID", "remote/model")
    monkeypatch.setenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("VLLM_MODEL_ID", "local/vllm-model")

    cfg = load_config(None)
    assert cfg.provider_mode == "vllm"
    assert cfg.model.base_url == "http://127.0.0.1:8000/v1"
    assert cfg.model.model_id == "local/vllm-model"
    assert cfg.model.api_key == "unused"


def test_vllm_per_role_executor_url(monkeypatch):
    monkeypatch.setenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("VLLM_MODEL_ID", "big-model")
    monkeypatch.setenv("VLLM_EXECUTOR_BASE_URL", "http://127.0.0.1:8001/v1")
    monkeypatch.setenv("VLLM_EXECUTOR_MODEL_ID", "small-model")

    cfg = Config()
    from claw_eval.config import _apply_provider_auto_selection

    cfg = _apply_provider_auto_selection(cfg)
    assert cfg.executor_model is not None
    assert cfg.executor_model.base_url == "http://127.0.0.1:8001/v1"
    assert cfg.executor_model.model_id == "small-model"
    assert cfg.model.base_url == "http://127.0.0.1:8000/v1"


def test_flat_config_ignores_stale_decomposer_role_env(monkeypatch):
    monkeypatch.setenv("VLLM_BASE_URL", "http://127.0.0.1:8010/v1")
    monkeypatch.setenv("VLLM_MODEL_ID", "flat-model")
    monkeypatch.setenv("VLLM_DECOMPOSER_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("VLLM_DECOMPOSER_MODEL_ID", "decomposer-model")

    cfg = load_config("config_vllm.yaml")

    assert cfg.model.base_url == "http://127.0.0.1:8010/v1"
    assert cfg.model.model_id == "flat-model"


def test_decomposer_config_uses_decomposer_role_env(monkeypatch):
    monkeypatch.setenv("VLLM_BASE_URL", "http://127.0.0.1:8010/v1")
    monkeypatch.setenv("VLLM_MODEL_ID", "flat-model")
    monkeypatch.setenv("VLLM_DECOMPOSER_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("VLLM_DECOMPOSER_MODEL_ID", "decomposer-model")
    monkeypatch.setenv("VLLM_EXECUTOR_BASE_URL", "http://127.0.0.1:8001/v1")
    monkeypatch.setenv("VLLM_EXECUTOR_MODEL_ID", "executor-model")

    cfg = load_config("config_decomposer.yaml")

    assert cfg.model.base_url == "http://127.0.0.1:8000/v1"
    assert cfg.model.model_id == "decomposer-model"
    assert cfg.executor_model is not None
    assert cfg.executor_model.base_url == "http://127.0.0.1:8001/v1"
    assert cfg.executor_model.model_id == "executor-model"


def test_collect_vllm_wait_targets_deduplicates(monkeypatch):
    monkeypatch.setenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("VLLM_MODEL_ID", "same-model")

    cfg = Config()
    from claw_eval.config import _apply_provider_auto_selection

    cfg = _apply_provider_auto_selection(cfg)
    targets = collect_vllm_wait_targets(
        cfg,
        roles=["model", "decomposer"],
        include_judge=False,
    )
    assert len(targets) == 1
    assert targets[0][2] == "same-model"


def test_collect_wait_targets_multi_server(monkeypatch):
    monkeypatch.setenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("VLLM_MODEL_ID", "big-model")
    monkeypatch.setenv("VLLM_EXECUTOR_BASE_URL", "http://127.0.0.1:8001/v1")
    monkeypatch.setenv("VLLM_EXECUTOR_MODEL_ID", "small-model")

    cfg = Config()
    from claw_eval.config import _apply_provider_auto_selection

    cfg = _apply_provider_auto_selection(cfg)
    targets = collect_vllm_wait_targets(
        cfg,
        roles=["decomposer", "executor"],
        include_judge=False,
    )
    urls = {t[0] for t in targets}
    assert urls == {"http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1"}


def test_vllm_strips_openrouter_extra_body_but_keeps_chat_template_kwargs(monkeypatch):
    monkeypatch.setenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
    cfg = Config()
    cfg.model.extra_body = {
        "reasoning": {"effort": "high"},
        "chat_template_kwargs": {"enable_thinking": False},
    }

    from claw_eval.config import _apply_provider_auto_selection

    cfg = _apply_provider_auto_selection(cfg)
    assert cfg.model.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}


def test_react_retry_defaults_disabled():
    cfg = Config()
    assert cfg.react.max_turns is None
    assert cfg.react.max_environment_tool_calls is None
    assert cfg.react.retry_empty_model_response is False
    assert cfg.react.retry_missing_required_tool is False
    assert cfg.react.retry_transitional_tool_text is False
    assert cfg.react.transitional_tool_retry_limit == 2
    assert cfg.decomposer.executor_synthetic_failure_report is True


def test_vllm_retry_settings_by_mode():
    flat_cfg = load_config("config_vllm.yaml")
    decomposer_cfg = load_config("config_decomposer.yaml")

    assert flat_cfg.react.max_turns == 352
    assert flat_cfg.react.max_environment_tool_calls == 320
    assert flat_cfg.react.retry_empty_model_response is False
    assert flat_cfg.react.retry_missing_required_tool is False
    assert flat_cfg.react.retry_transitional_tool_text is False
    assert flat_cfg.react.transitional_tool_retry_limit == 2
    assert decomposer_cfg.react.max_turns is None
    assert decomposer_cfg.react.max_environment_tool_calls is None
    assert decomposer_cfg.react.retry_empty_model_response is False
    assert decomposer_cfg.react.retry_missing_required_tool is False
    assert decomposer_cfg.react.retry_transitional_tool_text is False
    assert decomposer_cfg.react.transitional_tool_retry_limit == 2
    assert decomposer_cfg.decomposer.executor_prompt_mode == "report_wrapper"
    assert decomposer_cfg.decomposer.executor_synthetic_failure_report is False
    assert decomposer_cfg.decomposer.manager_valid_tool_guidance is False


def test_decomposer_experiment_configs_load_manager_guidance():
    retry_cfg = load_config("configs/experiments/config_decomposer_transitional_retry.yaml")
    strict_cfg = load_config("configs/experiments/config_decomposer_strict_manager_tools.yaml")
    prompt_v2_budget_cfg = load_config(
        "configs/experiments/config_decomposer_prompt_v2_strict_tools_budget16.yaml"
    )
    report_wrapper_cfg = load_config(
        "configs/experiments/config_decomposer_report_wrapper_transitional_strict_tools.yaml"
    )
    report_wrapper_min0_cfg = load_config(
        "configs/experiments/config_decomposer_report_wrapper_min0_transitional_strict_tools.yaml"
    )

    assert retry_cfg.react.retry_transitional_tool_text is False
    assert retry_cfg.decomposer.executor_prompt_mode == "flat_subtask"
    assert retry_cfg.decomposer.executor_report_mode == "strict"
    assert retry_cfg.decomposer.executor_synthetic_failure_report is False
    assert retry_cfg.decomposer.manager_valid_tool_guidance is False

    assert strict_cfg.react.retry_transitional_tool_text is False
    assert strict_cfg.decomposer.executor_prompt_mode == "flat_subtask"
    assert strict_cfg.decomposer.executor_report_mode == "strict"
    assert strict_cfg.decomposer.executor_synthetic_failure_report is False
    assert strict_cfg.decomposer.manager_valid_tool_guidance is True

    assert prompt_v2_budget_cfg.react.retry_transitional_tool_text is False
    assert prompt_v2_budget_cfg.decomposer.executor_prompt_mode == "flat_subtask"
    assert prompt_v2_budget_cfg.decomposer.max_delegations == 16
    assert prompt_v2_budget_cfg.decomposer.max_decomposer_turns == 32
    assert prompt_v2_budget_cfg.decomposer.executor_synthetic_failure_report is False
    assert prompt_v2_budget_cfg.decomposer.manager_valid_tool_guidance is True

    assert report_wrapper_cfg.react.retry_transitional_tool_text is False
    assert report_wrapper_cfg.decomposer.executor_prompt_mode == "report_wrapper"
    assert report_wrapper_cfg.decomposer.executor_report_mode == "strict"
    assert report_wrapper_cfg.decomposer.executor_min_tool_calls == 1
    assert report_wrapper_cfg.decomposer.executor_synthetic_failure_report is False
    assert report_wrapper_cfg.decomposer.manager_valid_tool_guidance is True

    assert report_wrapper_min0_cfg.react.retry_transitional_tool_text is False
    assert report_wrapper_min0_cfg.decomposer.executor_prompt_mode == "report_wrapper"
    assert report_wrapper_min0_cfg.decomposer.executor_report_mode == "strict"
    assert report_wrapper_min0_cfg.decomposer.executor_min_tool_calls == 0
    assert report_wrapper_min0_cfg.decomposer.executor_synthetic_failure_report is False
    assert report_wrapper_min0_cfg.decomposer.manager_valid_tool_guidance is True


def test_no_retry_comparison_decomposer_configs_load_budget_and_prompt_modes():
    cases = [
        (
            "configs/experiments/config_decomposer_no_retry_flat_subtask_no_strict_budget16_min0.yaml",
            "flat_subtask",
            False,
        ),
        (
            "configs/experiments/config_decomposer_no_retry_flat_subtask_strict_tools_budget16_min0.yaml",
            "flat_subtask",
            True,
        ),
        (
            "configs/experiments/config_decomposer_no_retry_report_wrapper_no_strict_budget16_min0.yaml",
            "report_wrapper",
            False,
        ),
        (
            "configs/experiments/config_decomposer_no_retry_report_wrapper_strict_tools_budget16_min0.yaml",
            "report_wrapper",
            True,
        ),
    ]

    for path, prompt_mode, strict_tools in cases:
        cfg = load_config(path)
        assert cfg.react.retry_empty_model_response is False
        assert cfg.react.retry_missing_required_tool is False
        assert cfg.react.retry_transitional_tool_text is False
        assert cfg.react.max_turns is None
        assert cfg.react.max_environment_tool_calls is None
        assert cfg.decomposer.max_delegations == 16
        assert cfg.decomposer.max_decomposer_turns == 32
        assert cfg.decomposer.executor_max_turns == 20
        assert cfg.decomposer.executor_max_environment_tool_calls == 20
        assert cfg.decomposer.executor_min_tool_calls == 0
        assert cfg.decomposer.executor_prompt_mode == prompt_mode
        assert cfg.decomposer.executor_report_mode == "strict"
        assert cfg.decomposer.executor_synthetic_failure_report is False
        assert cfg.decomposer.executor_evidence_mode == "none"
        assert cfg.decomposer.manager_valid_tool_guidance is strict_tools
