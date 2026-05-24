import pytest

from sampling import (
    QWEN3_NON_THINKING_SAMPLING_PARAMS,
    QWEN3_THINKING_SAMPLING_PARAMS,
    QWEN36_CODE_THINKING_SAMPLING_PARAMS,
    QWEN36_NON_THINKING_SAMPLING_PARAMS,
    QWEN36_QA_THINKING_SAMPLING_PARAMS,
    build_sampling_params,
)


def test_sampling_for_qwen3_hf_model() -> None:
    assert (
        build_sampling_params("Qwen/Qwen3-32B", True, task="general")
        == QWEN3_THINKING_SAMPLING_PARAMS
    )
    assert (
        build_sampling_params("Qwen/Qwen3-4B", False, task="code")
        == QWEN3_NON_THINKING_SAMPLING_PARAMS
    )


def test_sampling_for_qwen3_openrouter_model() -> None:
    assert (
        build_sampling_params("qwen/qwen3-32b", True, task="code")
        == QWEN3_THINKING_SAMPLING_PARAMS
    )
    assert (
        build_sampling_params("qwen/qwen3-0.6b", False, task="general")
        == QWEN3_NON_THINKING_SAMPLING_PARAMS
    )


def test_sampling_for_qwen36_uses_task_specific_thinking_params() -> None:
    assert (
        build_sampling_params("Qwen/Qwen3.6-27B", True, task="code")
        == QWEN36_CODE_THINKING_SAMPLING_PARAMS
    )
    assert (
        build_sampling_params("Qwen/Qwen3.6-27B", True, task="general")
        == QWEN36_QA_THINKING_SAMPLING_PARAMS
    )
    assert (
        build_sampling_params("qwen/qwen3.6-27b", False, task="code")
        == QWEN36_NON_THINKING_SAMPLING_PARAMS
    )
    assert (
        build_sampling_params("qwen/qwen3.6-27b", False, task="general")
        == QWEN36_NON_THINKING_SAMPLING_PARAMS
    )


def test_unknown_sampling_task_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown sampling task"):
        build_sampling_params("Qwen/Qwen3.6-27B", True, task="classification")


def test_qwen35_does_not_match_qwen3() -> None:
    with pytest.raises(ValueError, match="No sampling parameters configured"):
        build_sampling_params("Qwen/Qwen3.5-9B", True, task="general")
